import torch
from base import BaseVAE
from torch import nn
from torch.nn import functional as F
from types_ import *
from vit_components import Attention, FeedForward
from einops.layers.torch import Rearrange
from pos3d import *

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    #y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    y = y.to(torch.float32)
    x = x.to(torch.float32)
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                                Attention(dim, heads = heads, dim_head = dim_head),
                                FeedForward(dim, mlp_dim)
                                ]))
    def forward(self, x):
        for attn, ff in self.layers:
             x = attn(x) + x
             x = ff(x) + x
        return x







class VITVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 depth : int = 2,
                 mlp_dims: int = 512,
                 heads : int = 8,
                 dim_head = 64,
                 patch_size = 32,
                 _3d = False,
                 **kwargs) -> None:
        super(VITVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.fc_mu = nn.Linear(mlp_dims, latent_dim)
        self.fc_var = nn.Linear(mlp_dims, latent_dim)


        self._3d = _3d
        image_height = image_width = kwargs['im_size']
        if _3d:
            patch_dim = in_channels * patch_size ** 3
        else:
            patch_dim = in_channels * patch_size ** 2

        if _3d:
            ## channel last for position embedding 3d
            ## currently only 2D is used for the challenge
            self.to_patch_embedding = nn.Sequential(
                    Rearrange("b c (d p0) (h p1) (w p2) -> b (d h w) (p0 p1 p2 c)", p0=patch_size ,p1=patch_size, p2=patch_size),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    )
        else:
            self.to_patch_embedding = nn.Sequential(
                    Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, latent_dim),
                    nn.LayerNorm(latent_dim),

                )

        #if _3d:
        #    posemb = PositionalEncoding3D( patch_dim )
        #    self.pos_embedding = posemb
        #else:
        self.pos_embedding = posemb_sincos_2d(
                h = image_height // patch_size,
                w = image_width // patch_size,
                dim = latent_dim,
            ) 

        self.encoder_transformer = Transformer(latent_dim, depth, heads, dim_head, mlp_dims)

        self.decoder_transformer = Transformer(latent_dim, depth, heads, dim_head, latent_dim)

        if _3d:
            self.depatch = nn.Sequential(
                    nn.Linear( latent_dim, patch_dim),
                    nn.LayerNorm(patch_dim),
                    Rearrange("b (d h w) (p0 p1 p2 c) -> b c (d p0) (h p1) (w p2) ",p0 = patch_size, p1=patch_size, p2=patch_size, c=in_channels,
                                h=image_height // patch_size, w=image_height // patch_size),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid()
                )
        else:
            self.depatch = nn.Sequential(
                    nn.Linear( latent_dim, patch_dim),
                    nn.LayerNorm(patch_dim),
                    Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2) ", p1=patch_size, p2=patch_size, c=in_channels, h=image_height // patch_size),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid()
                )
                                                            

    def encode(self, img: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        result = self.encoder_transformer(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_transformer(z)
        result = self.depatch(result)
        return result

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

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        ## Each patch is consider a sample, the latent of each patch q(z | x_patch_i, position) | P(z)
        # we average across the batch, but kld_loss can be average across patch 
        kld_loss = kld_loss.mean()

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

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
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
