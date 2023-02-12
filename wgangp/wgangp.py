# implementation https://arxiv.org/abs/1704.00028

from torch_snippets import*
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize,Lambda
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_transforms = Compose([
    ToTensor(),
    Normalize(mean=(0.5),std=(0.5)),
    Lambda(lambda x:x.to(device))
])

training_data = MNIST('./mnist/',transform = img_transforms,train=True,download=False)

class Generator(nn.Module):
    def __init__(self,latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim,128),
            nn.LeakyReLU(0.2,inplace = True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,28*28),
            nn.Tanh(),)

    def forward(self,x):
        x = self.model(x)
        x = x.reshape(len(x),1,28,28)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1))
    
    def forward(self,x):
        return self.model(x)



class WGANGP(nn.Module):
    def __init__(self,D,G,latent_dim=100,n_critic=5,clip_value=0.01,gamma=10):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.gamma = gamma

        self.generator = G
        self.discriminator = D

        self.data_loader = DataLoader(training_data,batch_size = 64,shuffle=True)

        self.Goptimizer = optim.Adam(self.generator.parameters(), lr=0.0001,betas=(0.,0.9))
        self.Doptimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001,betas=(0.,0.9))

    def build_wgangp(self):
        self.discriminator.to(device)
        self.generator.to(device)
        from torchsummary import summary
        summary(self.discriminator,torch.zeros(2,28*28))
        summary(self.generator,torch.zeros(2,self.latent_dim))

    def gradient_penalty(self,real_data,fake_data):

        eps = torch.randn(len(real_data),1).to(device)

        inter = (eps*real_data + ((1-eps)*fake_data)).requires_grad_(True)

        d_inter = self.discriminator(inter)
        fake = torch.ones(len(real_data),1).to(device)

        gradient = torch.autograd.grad(
            outputs=d_inter,
            inputs=inter,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty



    def train_discriminator(self,real,fake):

        self.Doptimizer.zero_grad()

        pred_real = self.discriminator(real)
        pred_fake = self.discriminator(fake)
        gradient_penalty = self.gradient_penalty(real,fake)

        d_loss = -torch.mean(pred_real) + torch.mean(pred_fake) + self.gamma * gradient_penalty

        d_loss.backward()
        self.Doptimizer.step()


        return d_loss

    def train_generator(self,fake):
        self.Goptimizer.zero_grad()

        fake_data = self.generator(fake)
        fake_data = fake_data.reshape(len(fake_data),28*28)
        pred_fake = self.discriminator(fake_data)
        g_loss = -torch.mean(pred_fake)

        g_loss.backward()
        self.Goptimizer.step()
        return g_loss

    @torch.no_grad()
    def generate_samples(self):
        z = torch.randn(32,self.latent_dim).to(device)
        samples_images = self.generator(z).detach().cpu()
        grid = make_grid(samples_images.view(32,1,28,28), nrow=8, normalize=True)
        show(grid.cpu().detach().permute(1,2,0),title='Generate images')

    def train_wgangp(self):
        num_epochs = 30
        log = Report(num_epochs)

        for e in range(num_epochs):
            N = len(self.data_loader)

            for idx,(images,_) in enumerate(self.data_loader):
                images = images.reshape(len(images),28*28)
                fake_imgs = self.generator(torch.randn(len(images),self.latent_dim).to(device)).to(device).detach()
                fake_imgs = fake_imgs.reshape(len(fake_imgs),28*28)

                d_loss = self.train_discriminator(images,fake_imgs)

                if idx % self.n_critic == 0:
                    fake = torch.randn(len(images),self.latent_dim).to(device)
                    g_loss = self.train_generator(fake)

                log.record(e+(1+idx)/N, d_loss=d_loss.detach(), g_loss=g_loss.detach(), end='\r')
            log.report_avgs(e+1)
            self.generate_samples()
