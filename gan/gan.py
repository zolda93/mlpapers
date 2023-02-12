# implementation https://arxiv.org/abs/1406.2661

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import make_grid
from torch_snippets import*


device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),std=(0.5,))
])

training_data = datasets.MNIST('./mnist/',transform = image_transform,train=True,download = False)

class Discriminator(nn.Module):
    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid())


    def forward(self,x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,28*28),
            nn.Tanh(),)
        
    
    def forward(self,x):
        return self.model(x)


class GAN(nn.Module):

    def __init__(self,discr,gener):

        super().__init__()

        self.discriminator = discr
        self.generator = gener
        self.data_loader = DataLoader(training_data,batch_size=128,shuffle=True)

        self.Doptimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.Goptimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.loss = nn.BCELoss()

    def build_model(self):

        self.discriminator.to(device)
        self.generator.to(device)

        from torchsummary import summary
        summary(self.discriminator,torch.zeros(1,28*28))
        summary(self.generator,torch.zeros(1,100))

    def train_discriminator(self,real_data,fake_data):
        # reset gradient
        self.Doptimizer.zero_grad()

        #train discriminator with real data labele as one
        pred_real = self.discriminator(real_data)
        error_real = self.loss(pred_real,torch.ones(len(real_data),1).to(device))
        error_real.backward()

        #train discriminator with fake data labeled as zero
        pred_fake = self.discriminator(fake_data)
        error_fake = self.loss(pred_fake,torch.zeros(len(fake_data),1).to(device))
        error_fake.backward()
        #update
        self.Doptimizer.step()

        return error_real + error_fake

    def train_generator(self,fake_data,real_data):

        self.Goptimizer.zero_grad()

        pred = self.discriminator(fake_data)
        error = self.loss(pred,torch.ones(len(real_data),1).to(device))
        error.backward()

        self.Goptimizer.step()
        return error

    def train_gan(self):
        num_epochs = 50
        log = Report(num_epochs)

        for e in range(num_epochs):
            N = len(self.data_loader)

            for idx,(images,_) in enumerate(self.data_loader):
                real = images.view(len(images),-1).to(device)
                fake = self.generator(torch.randn(len(real),100).to(device)).to(device)
                fake = fake.detach()

                d_loss = self.train_discriminator(real,fake)
                fake = self.generator(torch.randn(len(real),100).to(device)).to(device)
                g_loss = self.train_generator(fake,real)
                log.record(e+(1+idx)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')
            log.report_avgs(e+1)
        log.plot_epochs(['d_loss', 'g_loss'])
