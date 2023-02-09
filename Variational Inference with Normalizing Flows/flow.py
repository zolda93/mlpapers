# https://arxiv.org/pdf/1505.05770.pdf

import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import math

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_gaussian(z:Tensor)->Tensor:
    return -0.5*(torch.log(torch.tensor([math.pi*2] ,device=z.device)) + z**2).sum(1)


class Flow(nn.Module):
    def __init__(self,D:int)->None:
        super().__init__()
        
        # refer to equation (10) in paper
        self.u = nn.Parameter(torch.rand(D))
        self.w = nn.Parameter(torch.rand(D))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()

    def condition(self)->Tensor:
        """
        Condition the parameter u to ensure invertibility
        """
        wTu = torch.matmul(self.w.T,self.u)
        return self.u + (-1 + torch.log(1+torch.exp(wTu)) + wTu)*(self.w /(torch.norm(self.w)**2 + 1e-15))

    def forward(self,z:Tensor)->Tuple[Tensor,Tensor]:
        u = self.condition()
        hidden = torch.matmul(self.w.T,z.T) + self.b
        f_z = z + u.unsqueeze(0)*self.h(hidden).unsqueeze(-1)
        qsi = (1-self.h(hidden)**2).unsqueeze(0)*self.w.unsqueeze(-1) # equation(11)
        log_det = torch.log((1+torch.matmul(u.T,qsi)).abs() + 1e-15)
        return f_z,log_det



class NormalizingFlow(nn.Module):
    def __init__(self,flow_len:int,D:int)->None:
        super().__init__()

        self.layers = nn.Sequential(
                *(Flow(D) for _ in range(flow_len))
                )

    def forward(self,z:Tensor)->Tuple[Tensor,float]:
        log_jacobians = 0
        for l in self.layers:
            z,log_jacobian = l(z)
            log_jacobians += log_jacobian
        return z,log_jacobians


def train(flow,optimizer,epochs,log_density,batch_size,data_dim):
    training_loss = []

    for e in tqdm(range(epochs)):
        #generate new samples from flow
        z0 = torch.randn(batch_size,data_dim).to(device)
        zk,log_jacobian = flow(z0)

        #evaluate the exact and approx densities
        flow_log_density = log_gaussian(z0) - log_jacobian
        exact_log_density = log_density(zk).to(device)


        # Compute the loss
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
    return training_loss


def plot_flow_density(flow, ax, lims=np.array([[-4, 4], [-4, 4]]), cmap="coolwarm", title=None,
                      nb_point_per_dimension=1000):
    # Sample broadly from the latent space
    latent_space_boundaries = np.array([[-15, 15], [-15, 15]]);
    xx, yy = np.meshgrid(
        np.linspace(latent_space_boundaries[0][0], latent_space_boundaries[0][1], nb_point_per_dimension),
        np.linspace(latent_space_boundaries[1][0], latent_space_boundaries[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1), dtype=torch.float)
    # Generate data points and evaluate their densities
    zk, log_jacobian = flow(z.to(device))
    final_log_prob = log_gaussian(z) - log_jacobian.cpu()
    qk = torch.exp(final_log_prob)

    ax.set_xlim(lims[0][0], lims[0][1]); ax.set_ylim(lims[1][0], lims[1][1])
    ax.pcolormesh(
        zk[:, 0].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension),
        zk[:, 1].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension) * -1,
        qk.detach().data.reshape(nb_point_per_dimension, nb_point_per_dimension),
        cmap=cmap,
        rasterized=True,
    )
    if title is not None:
        plt.title(title, fontsize=22)


def plot_exact_density(ax, exact_log_density, lims=np.array([[-4, 4], [-4, 4]]), nb_point_per_dimension=100,
                       cmap="coolwarm", title=None):
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], nb_point_per_dimension),
                         np.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1))
    density = torch.exp(exact_log_density(z)).reshape(nb_point_per_dimension, nb_point_per_dimension)
    ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap=cmap)
    if title is not None:
        plt.title(title, fontsize=22)

