import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from skimage import measure
import math
import random

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname != 'BNReLUConv': #filtered for MemNet: BNReLUConv, ResidualBlock, MemoryBlock
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            # nn.init.uniform(m.weight.data, 1.0, 0.02)
            m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    if math.isnan(PSNR):
        import pdb; pdb.set_trace()
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        if Img.shape[1] > 1: #if color image, swap color channels to last dimension and use multi-channel ssim
            np.moveaxis(Img[i,:],0,-1)
            SSIM += measure.compare_ssim(np.moveaxis(Iclean[i,:,:,:],0,-1), np.moveaxis(Img[i,:,:,:],0,-1), data_range=data_range, multichannel=True)
        else:
            SSIM += measure.compare_ssim(Iclean[i,0,:,:], Img[i,0,:,:], data_range=data_range)
    return (SSIM/Img.shape[0])


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        
        return img

    return [_augment(_l) for _l in l]
