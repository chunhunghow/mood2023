
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile
import pylibjpeg
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import numpy as np
import nrrd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import re
import cv2
import nibabel as nib

class LoadMOOD():
    def __init__(self, 
                 path,
                 ):
        '''
        Args:
        path: path to folder of abnormal 
        _2d: Load the data (b d h w) to (bxd, h, w)
        '''
        self.images = glob.glob(path + '/*')
        self.img_size = 256
        self.transform_labeled = A.Compose([
              A.HorizontalFlip(p=0.5),              
              A.Affine(scale=(0.99,1.0),
                       translate_percent=(0.,0.1),
                       rotate=(-15,15),
                       shear=(-2,2)
                  ),
              #A.RandomResizedCrop(height=img_size,
              #                    width=img_size, 
              #                    scale=(0.85,1.0),
              #                    p=0.7),
              A.RandomBrightnessContrast(),
              A.GaussNoise(),
              ToTensorV2(), #not normalized
              ])



    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        """

        Returns:
                    
        """
        nifti = nib.load(self.images[idx])
        im = nifti.get_fdata()
        im = im.transpose(2,1,0)
        affine_matrix = nifti.affine

        d = im.shape[-3]
        h0, w0 = im.shape[-2:]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_stack = []
            interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            for idx in range(d):
                sli = im[idx]
                sli = cv2.resize(sli, ( self.img_size, self.img_size), interpolation=interp)
                img_stack += [sli[np.newaxis,...]]
            im = np.concatenate(img_stack,0)

        return torch.from_numpy(im).to(torch.float32)[None,...]


        #healthy
        im = (im - im.min())/np.ptp(im)
        return torch.from_numpy(im).to(torch.float32)[:,None,...]



