import torch
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
import numpy as np

# likelihood
def store_NLL(x, recon, mu, logvar, z, patch_fn):
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        #recon = (recon * 255).long() #added
        recon = patch_fn(recon)

        recon = recon.contiguous()


        #target = patch_fn((x* 255).long())
        #recon = F.one_hot(recon, num_classes=256) #aded
        #log_p_x_z = -F.cross_entropy(recon.permute(0,-1,1,2).float(), target, reduction='none').sum(-1) #ce is postive = negative loglikelihood, so add a negative to become llk
        target = patch_fn(x)
        #sig2 = ((recon - recon.mean(-1, keepdim=True))**2).mean(-1,keepdim=True)
        sig2 = torch.std(target , dim=-1, keepdim=True)
        #sig2 = torch.std(target, dim=[1,2], keepdim=True)
        eps = 1e-6
        log_p_x_z = (-(( target - recon)**2) / (2* sig2**2 + eps)).sum(-1)

        # latent Z|X and Z chose to be multivariate gaussian, therefore its likelihood calculated as sum of Z'inv(S)Z but sigma S is assumed identity matrix
        # https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,2) #sum at the last dim , (b p d) -> (b p )

        z_eps = (z - mu)/sigma
        #z_eps = z_eps.view(opt.repeat,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 2) # sum at last dim
                
        weights = log_p_x_z+log_p_z-log_q_z_x
                
    return weights
#{'log_p_x_z': log_p_x_z, 'log_p_z':log_p_z , 'log_q_z_x':log_q_z_x}



def compute_NLL(weights):
        
    with torch.no_grad():
        max_w = weights.max(dim=-1, keepdim=True)[0]
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - max_w))) + max_w)
        #NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        #NLL_loss = -torch.mean(weights, dim=-1, keepdim=True)
                
    return NLL_loss
