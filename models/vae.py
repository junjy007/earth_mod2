import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from utils.common import *

class ChlVAE(nn.Module):        
    def __init__(self, cfg: Config):
        super(ChlVAE, self).__init__()
        self.cfg = cfg
        c = cfg # short cut
        self.latent_dim = c.latent_dim
        
        ## encoder
        # input 20 x 20
        kw = dict(kernel_size=3, stride=2, padding=1)
        self.conv_lay1 = nn.Conv2d(in_channels= 1, out_channels=16, **kw) # 100 -> 50
        self.bn1 = nn.BatchNorm2d(16)
        self.conv_lay2 = nn.Conv2d(in_channels=16, out_channels=32, **kw) # -> 25
        self.bn2 = nn.BatchNorm2d(32)
        self.conv_lay3 = nn.Conv2d(in_channels=32, out_channels=64, **kw) # -> 13
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_lay4 = nn.Conv2d(in_channels=64, out_channels=128, **kw) # -> 7
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_lay5 = nn.Conv2d(in_channels=128, out_channels=256, **kw) # -> 4
        self.bn5 = nn.BatchNorm2d(256)
        self.conv_lay6 = nn.Conv2d(in_channels=256, out_channels=512, **kw) # -> 2
        self.bn6 = nn.BatchNorm2d(512)
        
        self.inner_fmap_dim = 512
        self.inner_fmap_size = 2
        self.feature_vec_dim = self.inner_fmap_dim * (self.inner_fmap_size ** 2)
        
        # z-distribution
        self.fc_mu = nn.Linear(self.feature_vec_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.feature_vec_dim, self.latent_dim)
        
        ## decoder
        def get_kw(opad):
            return dict(kernel_size=3, stride=2, padding=1, output_padding=opad) 
        self.decoder_input = nn.Linear(self.latent_dim, self.feature_vec_dim)
        self.deconv6 = nn.ConvTranspose2d(512, 256, **get_kw(1)) # 2 -> 4
        self.bn6d = nn.BatchNorm2d(num_features=256)
        self.deconv5 = nn.ConvTranspose2d(256, 128, **get_kw(0)) # -> 7
        self.bn5d = nn.BatchNorm2d(num_features=128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, **get_kw(0)) # -> 13
        self.bn4d = nn.BatchNorm2d(num_features=64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, **get_kw(0)) # -> 25
        self.bn3d = nn.BatchNorm2d(num_features=32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, **get_kw(1)) # -> 50
        self.bn2d = nn.BatchNorm2d(num_features=16)
        self.deconv1 = nn.ConvTranspose2d(16, 16, **get_kw(1)) # -> 100
        self.bn1d = nn.BatchNorm2d(num_features=16)
        self.out_conv = nn.Conv2d(16, out_channels=1, kernel_size=3, padding=1)
      
        
    def encode(self, x: Tensor, msk: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        tmp = F.leaky_relu(self.bn1(self.conv_lay1(torch.masked_fill(x, msk, 0))))
        tmp = F.leaky_relu(self.bn2(self.conv_lay2(tmp)))
        tmp = F.leaky_relu(self.bn3(self.conv_lay3(tmp)))
        tmp = F.leaky_relu(self.bn4(self.conv_lay4(tmp)))
        tmp = F.leaky_relu(self.bn5(self.conv_lay5(tmp)))
        tmp = F.leaky_relu(self.bn6(self.conv_lay6(tmp)))
        
        feat = torch.flatten(tmp, start_dim=1)
        mu = self.fc_mu(feat)
        log_var = self.fc_var(feat)
        return [mu, log_var]  
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        tmp = self.decoder_input(z)
        sz = self.inner_fmap_size
        tmp = tmp.view(-1, self.inner_fmap_dim, sz, sz)
        tmp = F.leaky_relu(self.bn6d(self.deconv6(tmp)))
        tmp = F.leaky_relu(self.bn5d(self.deconv5(tmp)))
        tmp = F.leaky_relu(self.bn4d(self.deconv4(tmp)))
        tmp = F.leaky_relu(self.bn3d(self.deconv3(tmp)))
        tmp = F.leaky_relu(self.bn2d(self.deconv2(tmp)))
        tmp = F.leaky_relu(self.bn1d(self.deconv1(tmp)))
        x_hat = torch.tanh(self.out_conv(tmp))
        return x_hat
                                       
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
        
    def forward(self, x: Tensor, msk: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x, msk)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var, z
    
    def loss_function(
        self,
        results: List[Tensor], # "forward" output
        targets: List[Tensor], # x-in and mask
        **kwargs) -> dict:
        """
        Computes the VAE loss function.
        $$
        KL(N(\mu, \sigma), N(0, 1)) = 
          \log \frac{1}{\sigma} 
          + \frac{\sigma^2 + \mu^2}{2} 
          - \frac{1}{2}
        $$
        """
        x_hat, mu, log_var = results[:3] # z not used
        x_in, msk = targets
        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N'] 

        mx = torch.masked_fill(x_in, msk, 0)
        err = torch.masked_fill(x_hat - mx, msk, 0)
        recons_loss = (err ** 2).mean()

        kld = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
        kld_loss = kld.sum(dim=1).mean(dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 
            'Reconstruction_Loss':recons_loss, 
            'KLD':-kld_loss}
    
    
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim) # from N(0, 1) NOT any posterior !!
        z = z.to(current_device)
        samples = self.decode(z)
        return samples