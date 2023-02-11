import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.distributions as dist


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Planar(nn.Module):
    def __init__(self,size:int=1,init_sigma:float=0.01)->None:
        super().__init__()

        self.u = nn.Parameter(torch.randn(1,size).normal_(0,init_sigma))
        self.w = nn.Parameter(torch.randn(1,size).normal_(0,init_sigma))
        self.b = nn.Parameter(torch.zeros(1))
        self.h = nn.Tanh()
        self.h_prime = lambda x:1-torch.tanh(x)**2


    def constraint_u(self)->Tensor:
        """
        Needed for invertibility condition.
        See Appendix A.1
        Rezende et al. Variational Inference with Normalizing Flows
        https://arxiv.org/pdf/1505.05770.pdf
        """
        m = lambda x:-1 + torch.log(1+torch.exp(x))
        wtu = torch.matmul(self.w,self.u.t())
        norm = self.w / torch.norm(self.w + 1e-15)
        return self.u + (m(wtu) - wtu)*norm

    def psi(self,z:Tensor)->Tensor:

        """
        ψ(z) =h′(w^tz+b)w
        See eq(11)
        Rezende et al. Variational Inference with Normalizing Flows
        https://arxiv.org/pdf/1505.05770.pdf
        """

        return self.h_prime(z@self.w.t() + self.b)*self.w

    def forward(self,z:Tensor)->Tuple[Tensor,float]:

        if isinstance(z,tuple):
            z,log_det_jac = z
        else:
            z,log_det_jac = z,0

        psi = self.psi(z)
        u = self.constraint_u()

        #determinant of jacobian
        det = (1 + psi@u.t())

        #log |det Jac|
        ldj = torch.log(torch.abs(det) + 1e-6)

        wzb = z@self.w.t() + self.b
        fz = z + (u * self.h(wzb))
        return fz,log_det_jac + ldj



class Flow(nn.Module):
    def __init__(self,dim:int=2,n_flows:int=10)->None:
        super().__init__()

        self.flow = nn.Sequential(*[
            Planar(dim) for _ in range(n_flows)])
        self.mu = nn.Parameter(torch.randn(dim,).normal_(1,0.01))
        self.log_var = nn.Parameter(torch.randn(dim,).normal_(1,0.01))

    def forward(self,shape):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(shape)  # unit gaussian
        z0 = self.mu + eps * std

        zk, ldj = self.flow(z0)
        return z0, zk, ldj, self.mu, self.log_var


def det_loss(mu, log_var, z_0, z_k, ldj, beta):
    # assume uniform prior here.
    # So P(z) is constant and not modelled in this loss function
    batch_size = z_0.size(0)

    # Qz0
    log_qz0 = dist.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z_0)
    # Qzk = Qz0 + sum(log det jac)
    log_qzk = log_qz0.sum() - ldj.sum()
    # P(x|z)
    nll = -torch.log(target_density(z_k) + 1e-7).sum() * beta
    return (log_qzk + nll) / batch_size


def target_density(z):
    z1, z2 = z[..., 0], z[..., 1]
    norm = (z1**2 + z2**2)**0.5
    exp1 = torch.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)


def train_flow(flow, shape, epochs=1000):
    optim = torch.optim.Adam(flow.parameters(), lr=1e-2)
    
    for i in range(epochs):
        z0, zk, ldj, mu, log_var = flow(shape=shape)
        loss = det_loss(mu=mu,
                        log_var=log_var,
                        z_0=z0,
                        z_k=zk,
                        ldj=ldj,
                        beta=1)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 1000 == 0:
            print(loss.item())


