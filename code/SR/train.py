import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import functools
from functools import partial
import numpy as np
import glob
import cv2
import os
import torch.nn as nn
import math
import argparse

from model.MPNCOV.MPNCOV import CovpoolLayer
from model.common import save_checkpoint, Hdf5Dataset, adjust_learning_rate
from model.RRDB import RRDBNet
from model.vgg_feature_extractor import VGGFeatureExtractor
from model.GAN import Discriminator_VGG_128, GANLoss
from model.SRFBN import SRFBN
from model.SAN import SAN
from model.RDN import RDN
from model.RCAN import RCAN

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--nEpochs", type=int, default=49, help="Number of training epochs")
parser.add_argument("--start_epoch", type=int, default=0, help='Starting Epoch')
parser.add_argument("--net", type=str, default="ours", help="RCAN, ESRGAN, SAN, RDN, EPSR, SRFAN, ours")
parser.add_argument("--lr_h5", type=str, default="../../net_data/avg400_64_32.h5", help='path of LR h5 file')
parser.add_argument("--hr_h5", type=str, default="../../net_data/sim_128_64.h5", help='path of HR h5 file')
parser.add_argument("--ngpu", type=int, default=1, help='number of GPUs')
parser.add_argument("--batch_size", type=int, default='16', help='number of GPUs')
parser.add_argument("--resume", type=str, default="", help='restart training checkpoint path')
parser.add_argument('--extra_model_name', type=str, default='', help='addition name for path')


opt = parser.parse_args()

num_workers = 1
batch_size = opt.batch_size
initial_lr = 0.0001
train_set = Hdf5Dataset(lrname=opt.lr_h5, hrname=opt.hr_h5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("===> Building model")
if (opt.net == 'ours'):
    model = RRDBNet()
    mse_factor = 0.5
    feature_factor = 0.05
    texture_factor = 0.05
    vgg = VGGFeatureExtractor(device=device, feature_layer=[2, 7, 16, 25, 34], use_bn=False, use_input_norm=True)
    vgg = nn.DataParallel(vgg,device_ids=range(opt.ngpu))
    vgg.to(device)
if (opt.net == 'RCAN'):
    model = RCAN()
if (opt.net == 'SRFBN'):
    model = SRFBN()
if (opt.net == 'SAN'):
    model = SAN()
elif (opt.net == 'RDN'):
    model = RDN()
elif (opt.net == 'EPSR'):
    model = EDSR()
    GANcriterion = GANLoss('ragan', 1.0, 0.0)
    l1_factor = 0
    mse_factor = .05
    feature_factor = 1
    gan_factor = 0.04
elif (opt.net == 'ESRGAN'):
    model = RRDBNet(nb=23)
    GANcriterion = GANLoss('ragan', 1.0, 0.0)
    l1_factor = 0.01
    mse_factor = 0
    feature_factor = 1.
    gan_factor = 0.005
if (opt.net == 'EPSR' or opt.net == 'ESRGAN'):
    gan = Discriminator_VGG_128()
    gan = nn.DataParallel(gan,device_ids=range(opt.ngpu))
    gan.to(device)
    vgg = VGGFeatureExtractor(device=device, feature_layer=34, use_bn=False, use_input_norm=True)
    vgg = nn.DataParallel(vgg,device_ids=range(opt.ngpu))
    vgg.to(device)
    optimizerD = optim.Adam(gan.parameters(), lr=initial_lr, weight_decay=1e-5)
    training_data_loader_gan = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

model = nn.DataParallel(model, device_ids=range(opt.ngpu))
if (len(opt.resume) > 0):
    model.load_state_dict(torch.load(opt.resume)['model'].state_dict())

model.to(device)
MSEcriterion = nn.MSELoss()
L1criterion = nn.L1Loss()

training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
for epoch in range(opt.start_epoch, opt.nEpochs):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    lr = adjust_learning_rate(initial_lr, optimizer, epoch)
    if (opt.net == 'ESRGAN' or opt.net == 'EPSR'):
        lr_gan = adjust_learning_rate(initial_lr, optimizerD, epoch)
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        x_data, z_data = Variable(batch[0].float()).cuda(), Variable(batch[1].float()).cuda()
        output = model(z_data)
        
        if (opt.net == 'SRFBN'):
            loss = L1criterion(output[0], x_data) + L1criterion(output[1], x_data) + L1criterion(output[2], x_data) + L1criterion(output[3], x_data)
            mseloss = MSEcriterion(output[3], x_data)
        else:
            loss = MSEcriterion(output, x_data)
            mseloss = MSEcriterion(output, x_data)
        
        if (opt.net == 'ours'):
            color_gt = torch.cat((x_data,x_data,x_data),1).cuda()
            color_output = torch.cat((output,output,output),1).cuda()
            
            vgg_gt = vgg(color_gt)
            vgg_output = vgg(color_output)
            for i in range(5):
                loss = loss + 0.2*feature_factor*MSEcriterion(vgg_gt[i], vgg_output[i])
                if (i==4):
                    loss += 0.2*feature_factor*MSEcriterion(CovpoolLayer(vgg_gt[i]),CovpoolLayer(vgg_output[i]))
          
        if (opt.net == 'ESRGAN' or opt.net == 'EPSR'):
            x_data_gan, z_data_gan = next(iter(training_data_loader_gan))
            x_data_gan = Variable(x_data_gan.float()).cuda()
            pred_d_real = gan(x_data_gan).detach()
            pred_g_fake = gan(output)
            if opt.net == 'ESRGAN':
                l_g_gan = gan_factor * (
                    GANcriterion(pred_d_real - torch.mean(pred_g_fake), False) +
                    GANcriterion(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            if opt.net == 'EPSR':
                l_g_gan = gan_factor * (GANcriterion(pred_g_fake - torch.mean(pred_d_real), True))
            color_gt = torch.cat((x_data,x_data,x_data),1)
            color_output = torch.cat((output,output,output),1)
            loss = mse_factor * mseloss + l1_factor * loss + feature_factor * L1criterion(vgg(color_output), vgg(color_gt)) + l_g_gan
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (opt.net == 'ESRGAN' or opt.net == 'EPSR'):
            optimizerD.zero_grad()
            pred_d_fake = gan(output).detach()
            pred_d_real = gan(x_data_gan)
            l_d_real = GANcriterion(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
            pred_d_fake = gan(output.detach())
            l_d_fake = GANcriterion(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
            D_loss = l_d_real + l_d_fake
            D_loss.backward()
            optimizerD.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): MSELoss: {:.10f}".format(epoch, iteration, len(training_data_loader), mseloss.item()))
            #save_checkpoint(model, epoch, 'RRDB')
    if (epoch % 10 == 9):
        save_checkpoint('../../net_data/trained_srs/' + opt.net + opt.extra_model_name + '/', model, epoch, opt.net)
save_checkpoint('../../net_data/trained_srs/' + opt.net + opt.extra_model_name + '/', model, epoch, opt.net)
