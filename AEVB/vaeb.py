# implementation of https://arxiv.org/abs/1312.6114

import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import scipy.io
from typing import Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and prepare training set
img_size = (28, 20)
img_data = scipy.io.loadmat('Data/frey_rawface.mat')["ff"]
img_data = img_data.T.reshape((-1, img_size[0], img_size[1]))
trainX = torch.tensor(img_data[:int(0.8 * img_data.shape[0])], dtype=torch.float)/255.


def get_minibatch(batch_size, device='cpu'):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return trainX[indices].reshape(batch_size, -1).to(device)

class Model(nn.Module):

    """
    Model p(y|x) as N(mu, sigma) where mu and sigma are Neural Networks
    """
    def __init__(self,
                data_dim:int=2,
                context_dim:int=2,
                hidden_dim:int=200,
                constrain:bool=False)->None:

        super().__init__()

        self.h = nn.Sequential(
                nn.Linear(context_dim,hidden_dim),
                nn.Tanh()
                )

        self.log_var = nn.Sequential(
                nn.Linear(hidden_dim,data_dim)
                )

        if constrain:
            self.mu = nn.Sequential(
                    nn.Linear(hidden_dim,data_dim),
                    nn.Sigmoid()
                    )
        else:
            self.mu = nn.Sequential(
                    nn.Linear(hidden_dim,data_dim)
                    )

    def mean_log_var(self,x:Tensor)->Tuple[Tensor,Tensor]:
        h = self.h(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu,log_var

    def compute_log_density(self,y:Tensor,x:Tensor):

        # Compute log p(y|x)

        mu,log_var = self.mean_log_var(x)
        log_density = -.5 * (torch.log(2 * torch.tensor(np.pi)) + log_var + (((y-mu)**2)/(torch.exp(log_var) + 1e-10))).sum(dim=1)
        return log_density

    def KL(self,x):

        # Assume that p(x) is a normal gaussian distribution; N(0, 1)

        mu,log_var = self.mean_log_var(x)
        return -.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)

    def forward(self,eps,x):

        # Sample y ~ p(y|x) using the reparametrization trick
        mu, log_var = self.mean_log_var(x)
        sigma = torch.sqrt(torch.exp(log_var))
        return eps * sigma + mu



def train(encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        epochs,
        M=100,
        L=1,
        latent_dim=1):

    losses = []

    for e in tqdm(range(epochs)):
        x = get_minibatch(M,device=device)
        eps = torch.normal(torch.zeros(M*L,latent_dim),torch.ones(latent_dim)).to(device)

        #loss
        z = encoder(eps,x)
        log_ll = decoder.compute_log_density(x,z)
        kl = encoder.KL(x)
        loss = (kl - log_ll.view(-1, L).mean(dim=1)).mean()

        encoder.zero_grad()
        decoder.zero_grad()

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        losses.append(loss.item())

    return losses






