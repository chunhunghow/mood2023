from pathlib import Path
import argparse
import os
import numpy as np
import copy
#import pytorch_lightning as pl
#from pytorch_lightning.trainer import Trainer
import torch
from torch import nn
#from mood_dataloader import LoadMOOD
from torch.utils.data import DataLoader
#from pytorch_lightning.loggers import WandbLogger
import sys
from einops import rearrange
from vit_vae import VITVAE
from einops.layers.torch import Rearrange
from tqdm import tqdm
from utils import *
import cv2
import sys


class Model(nn.Module):
    def __init__(self, 
                 model='vitvae', 
                 _3d = False,
                 max_batch_size = 64,
                 im_size = 256,
                 in_channels = 1,
                 threshold=1000,
                 patch_size = 32,
                 ):

        super().__init__()


        self.learning_rate = 1e-4
        self.patch_size = patch_size
        self.NUM_ITER_TEST = 50
        self.regret = []
        self._3d = _3d
        self.max_batch_size = max_batch_size
        self.img_size = im_size

        self.thres = threshold
        self.second_thres = threshold//4

        #VIT_VAE
        if model == 'vitvae':
            self.net = VITVAE(in_channels= in_channels,
                          latent_dims = 256,
                          depth = 4,
                          mlp_dims = 256,
                          im_size = im_size,
                          patch_size = self.patch_size,
                          _3d = _3d
                )


        elif model == 'transunet':
            #Traunnet
            raise NotImplementedError
            #img_size = 256
            #vit_patches_size = self.patch_size
            #vit_name = 'R50-ViT-B_16'

            #config_vit = CONFIGS_ViT_seg[vit_name]
            #config_vit.n_classes = 1 #for mse loss
            #config_vit.n_skip = 3 # not important
            #if vit_name.find('R50') != -1:
            #    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
            #self.net = transunet_vae(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()

        self.patching = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        self.depatching = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2) ", p1=self.patch_size, p2=self.patch_size, c=in_channels, h=im_size // patch_size)

    def forward(self, im ):

        return self.net(im)



    def training_step(self, batch, batch_idx):
        if self._3d:
            im = batch 
        else:
            im = rearrange(batch, 'b c d h w -> (b c d) 1 h w')

        if im.shape[0] > self.max_batch_size:
            pick = np.random.choice(im.shape[0], self.max_batch_size, replace=False)
            im = im[pick]


        
        h = w = im.shape[-1] // self.patch_size
        c = im.shape[1] # in channels
        b = im.shape[0]
        out_args = self(im)
        loss = self.net.loss_function(*out_args,M_N=0.001)
        recon, _, _ ,_ = out_args
        


        #==========================================================================
        #        Draw
        #==========================================================================
        #if (self.global_step % 2000 == 0) & (self.global_step > 1):
        #if (self.global_step % 10000 == 0) :

        #    # Image Patching
        #    #============================
        #    #im = rearrange(im, "(b c h w) 1 p1 p2 -> b c (h p1) (w p2)" ,b=b, c=c, h=h, w=w)           
        #    #recon = rearrange(recon, "(b c h w) 1 p1 p2 -> b c (h p1) (w p2)" ,b=b, c=c, h=h, w=w)           


        #    #============================
        #    grid = []
        #    for i in [0,1]:
        #        grid_ori = (im[i].cpu() * 255).to(torch.uint8)
        #        grid_recon = (recon[i].cpu() *255).to(torch.uint8)
        #        grid += [grid_ori, grid_recon]

        #    grid = make_grid(grid, nrow= 2).float()
        #    self.logger.experiment.log({'Plot' : wandb.Image(grid)})
        #==========================================================================
        #==========================================================================


        return loss



        
    #@torch.inference_mode(False)
    def test_step(self, batch,batch_idx):

        """
        The test is to use the same dataset but artificially noised, only for evaluating the method
        Does not work for test time.
        """
        pass
        #return torch.cat(cls_label,0).view(-1), regret.view(-1)

    def on_test_epoch_end(self):
        pass





    def configure_optimizers(self):
        opt_gen = torch.optim.AdamW( self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        #opt_disc = torch.optim.Adam(self.discriminator.parameters(),
        #                                            lr=self.learning_rate, betas=(0.5, 0.9))
        
        #return opt_gen, opt_disc
        return opt_gen


    def predict(self,nimg ):

        '''
        Test nifti image is a 3d scan, we will run the slices in batches. 

        '''
        assert isinstance(nimg, np.ndarray), 'Input must be numpy array.'

        #prefer using gpu
        device_properties = torch.cuda.get_device_properties('cuda')
        vram = device_properties.total_memory / 1024**3 #consider size (max batch size) and inference time limit (600s)
        # if sample slice is 256x256x256, running 100 iter is within 400s , occupied 14gb 
        if nimg.shape[0] <= 256:
            if vram < 12: #11 gb 
                self.max_batch_size = 8
                self.NUM_ITER_TEST = 50
            elif vram <17: #17gb
                self.max_batch_size = 32
                self.NUM_ITER_TEST = 80
            else:
                pass

        else: #assume 512
            if vram < 12: #9 gb 
                self.max_batch_size = 8
                self.NUM_ITER_TEST = 50
            elif vram <17: #17gb
                self.max_batch_size = 32
                self.NUM_ITER_TEST = 40
            else:
                pass

        #==================================================================
        #resizing to 256
        #==================================================================

        slic_num = nimg.shape[-3]
        h0, w0 = nimg.shape[-2:]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_stack = []
            interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            for idx in range(slic_num):
                sli = nimg[idx]
                sli = cv2.resize(sli, ( self.img_size, self.img_size), interpolation=interp)
                img_stack += [sli[np.newaxis,...]]
            nimg = np.concatenate(img_stack,0)
        #==================================================================
        #==================================================================


        result = []
        it = nimg.shape[0] // self.max_batch_size


        for i in range(it):
            im = nimg[(i * self.max_batch_size):(i+1)* self.max_batch_size]
            im = torch.from_numpy(im).float()[:,None,...]
            im = im.to('cuda' if torch.cuda.is_available() else 'cpu')

            h = w = im.shape[-1] // self.patch_size
            c = im.shape[1] # in channels
            b = im.shape[0]
            self.net.eval() # for dropout in the network
            # compute nll loss of original vae
            with torch.no_grad():
                weights_agg = []
                for it in range(5):
                    mu, log_var = self.net.encode(im)
                    z = self.net.reparameterize(mu,log_var)
                    recon = self.net.decode(z)


                    weights = store_NLL(im, recon, mu, log_var, z, self.patching )
                    weights_agg += [weights]
                weights_agg = torch.stack(weights_agg,-1)
                NLL_loss_before = compute_NLL(weights_agg)


            with torch.enable_grad():
                # short optimization
                copy_net = copy.deepcopy(self.net)
                copy_net.train()
                params = [{'params': copy_net.encoder_transformer.parameters()},
                          {'params': copy_net.to_patch_embedding.parameters()},
                          {'params': copy_net.fc_mu.parameters()},
                          {'params': copy_net.fc_var.parameters()},
                        ]
                optimizer = torch.optim.AdamW( params, lr=self.learning_rate, betas=(0.9, 0.95))
                pbar = tqdm(range(self.NUM_ITER_TEST), total=self.NUM_ITER_TEST)
                pbar.set_description('Optimizing Posterior..')
                for it in pbar:
                    out = copy_net.to_patch_embedding(im)
                    out += copy_net.pos_embedding.to(im.device, dtype=im.dtype)
                    out = copy_net.encoder_transformer(out)
                    mu = copy_net.fc_mu(out)
                    log_var = copy_net.fc_var(out)

                    z = copy_net.reparameterize(mu,log_var)
                    recon = copy_net.decode(z)

                    loss = copy_net.loss_function(*[recon, im, mu, log_var], M_N=0.001)
                    optimizer.zero_grad()
                    loss['loss'].backward()
                    optimizer.step()

            # compute loss after optimization
            with torch.no_grad():
                copy_net.eval()
                weights_agg = []
                for it in range(5):
                    mu, log_var = copy_net.encode(im)
                    z = copy_net.reparameterize(mu,log_var)
                    recon = copy_net.decode(z)

                    weights = store_NLL(im, recon, mu, log_var, z, self.patching)
                    weights_agg += [weights]
                weights_agg = torch.stack(weights_agg,-1)
                NLL_loss_after = compute_NLL(weights_agg)

            regret = NLL_loss_before - NLL_loss_after

            result += [regret]


        # Depatching patch into image 
        regret = torch.cat(result, 0)
        regret = torch.maximum(regret.detach().cpu(), torch.zeros_like(regret).cpu())

        patch_q = self.patching(torch.from_numpy(nimg)[:,None]).quantile(0.9,dim=-1)
        regret[patch_q < 0.01] = 0



        #==============================================
        ## For the patches and slices  adjacent to detected 
        # outlier patch, measure with a lower threshold
        #==============================================
        sz = regret.shape
        regret = regret.view(sz[0], int(np.sqrt(sz[1])), int(np.sqrt(sz[1])), -1)
        regret_thres = torch.zeros_like(regret)
        indx = np.where(regret > self.thres)[:3]

        # patch +- 1 step, slice +- 5 slices
        z_range = np.append(np.arange(-5,0), np.arange(1,6))
        y_range = np.append(np.arange(-1,0), 1)
        x_range = np.append(np.arange(-1,0), 1)

        grid_z , grid_y = np.meshgrid(z_range,y_range, indexing='xy')   
        gzy = list(zip(grid_z.flatten(), grid_y.flatten()))
        _, gx = np.meshgrid(gzy, x_range)

        grids=[]
        for (tup, i, j ) in list(zip(gzy, *gx)): grids += [list(tup) + [i] ]; grids += [list(tup) + [j] ] 
        for (z,y,x) in zip(*indx):
            regret_thres[z,y,x,:] = True
            for g in grids: 
                try:
                    regret_thres[z+g[0], y+g[1], x+g[2], :] = regret[z+g[0], y+g[1], x+g[2], :] > self.second_thres
                except IndexError:
                    pass
        regret_thres = regret_thres.to(int)
        regret_thres = regret_thres.view(sz[0], sz[1], -1)

        #==============================================
        #==============================================
        

        mask = regret_thres.repeat(1,1,self.patch_size**2)
        mask = self.depatching(mask)[:,0].float().detach().cpu().numpy()

        #mask = (mask > self.thres).astype(float)

        #resize to original
        if r!= 1:
            slic_num = mask.shape[-3]
            img_stack = []
            interp = cv2.INTER_NEAREST
            for idx in range(slic_num):
                sli = mask[idx]
                sli = cv2.resize(sli, ( h0, w0), interpolation=interp)
                img_stack += [sli[np.newaxis,...]]
            mask = np.concatenate(img_stack,0)

        return ((regret > self.thres).sum() > 0).cpu().int().numpy()  , (mask>0).astype(int)


  




def random_patch( im_size):
    center = im_size // 2
    n_box = 1
    mask = torch.zeros(im_size, im_size)
    #loc = [ torch.clamp(torch.randn(2),-1,1) for _ in range(n_box)]
    loc = [ torch.randn(2) for _ in range(n_box)]
    ratio = [torch.rand(2) for _ in range(n_box)]
    for (x_delt,y_delt),(w,h) in zip(loc,ratio):
        x,y = int(center + x_delt * center//3), int(center+ y_delt * center//3)
        w = int(w * (center//3))
        h = int(h * (center//3))
        mask[max((y-w),0): max((y+w),0),max((x-w),0):max((x+w),0)] = max(torch.rand(1),0.2)
    return mask



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, help='')
    parser.add_argument('--wandb', action='store_true', help='')
    parser.add_argument('--m', type=str, default='', help='')
    parser.add_argument('--batch_size', type=int, default=3, help='')
    parser.add_argument('--max_batch_size', type=int, default=64, help='To fit in gpu')
    parser.add_argument('--modality', type=str, default='brain', help='abdom , brain')
    parser.add_argument('--is3d', action='store_true', help='Train data and model in 3D')
    parser.add_argument('--test', action='store_true', help='')
    parser.add_argument('--model', type=str , default='vitvae' )
    parser.add_argument('--ckpt_dir', type=str, help='The wandb run ID')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


#if __name__ == '__main__':
#
#    args = parse_opt()
#
#    path = lambda x: f'../data/{x}/{x}_train'
#
#    dataset = LoadMOOD(path(args.modality))
#    batch_size = args.batch_size
#    model = Model(model=args.model, 
#                  _3d=args.is3d,
#                  max_batch_size=args.max_batch_size
#
#
#                  )
#
#    train_loader = DataLoader(
#        dataset,
#        batch_size=batch_size,
#        num_workers=4,
#        drop_last=False)
#    logger = None
#
#    project = 'MOOD'
#    logdir = Path('./wandb') / project
#    if not os.path.exists(logdir): os.makedirs(logdir)
#    default_logger_cfgs = {
#              "project" : project,
#              "name": '',
#              'entity': '',
#              "save_dir": logdir,
#              "offline": not args.wandb ,
#                     }
#
#    logger = WandbLogger(**default_logger_cfgs)
#    trainer = Trainer(accelerator="gpu", devices=[int(args.device)], logger=logger, max_epochs=10000) #single gpu
#
#
#    if not (args.test or args.predict):
#        mode = 'train'
#    elif args.test and not args.predict:
#        mode = 'test'
#    else:
#        mode = 'predict'
#
#
#    if mode == 'train':
#        trainer.fit(model, train_loader)
#    elif mode == 'test' :
#        dataset = LoadMOOD(f'../data/{args.modality}/toy') #your directory of datset
#        test_dataset = dataset
#        test_loader = DataLoader(
#            test_dataset,
#            batch_size=batch_size,
#            num_workers=4,
#            shuffle=False,
#            drop_last=False)
#        #assert args.ckpt is not None
#        if args.ckpt_dir is not None:
#            ckpt = glob.glob(f'wandb/MOOD/MOOD/{args.ckpt_dir}/checkpoints/*')[0] #your checkpoint
#            print(f'Loading checkpoint from {ckpt}..')
#            model.load_state_dict(torch.load(ckpt,map_location='cpu')['state_dict'])
#            print('Loaded successfully!')
#        trainer.test(model, test_loader)
#
#    else:
#        train_loader = DataLoader(
#            dataset,
#            batch_size=1,
#            num_workers=4,
#            shuffle=False,
#            drop_last=False)
#        if args.ckpt_dir is not None:
#            ckpt = glob.glob(f'wandb/MOOD/MOOD/{args.ckpt_dir}/checkpoints/*')[0] #your checkpoint
#            print(f'Loading checkpoint from {ckpt}..')
#            model.load_state_dict(torch.load(ckpt,map_location='cpu')['state_dict'])
#            print('Loaded successfully!')
#        trainer.predict(model, train_loader)
#        np.save(f'{args.modality}_regret.npy',model.regret)
#
