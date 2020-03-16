import torch
import torch.nn as nn
import os

from RIDmodel import ops
from RIDmodel import common

##########################################################################################################################################################
# Noise CNN and Fusion Layers

class NoiseCNN(nn.Module):
    def __init__(self, channels, num_of_layers=5):
        super(NoiseCNN, self).__init__()
        kernel_size = 52
        padding = 2
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.noisecnn = nn.Sequential(*layers)
        self.sigmoid_mapping = nn.Sigmoid()

    def forward(self, x):
        noise_level = self.noisecnn(x)
        noise_level = self.sigmoid_mapping(noise_level)

        return noise_level



class FinalFusionLayers(nn.Module):
    def __init__(self, channels):
        super(FinalFusionLayers, self).__init__()
        kernel_size = 3
        padding = 1
        features = 16
        dilation = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=5*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation))
        
        self.fusion_layers = nn.Sequential(*layers)

    def forward(self, a, b, c):
        noisy_input = a
        prior = b
        noise_level = c

        channel_0 = noisy_input
        channel_1 = prior
        channel_2 = noise_level
        channel_3 = noisy_input * (1-noise_level)
        channel_4 = prior * noise_level

        x = torch.cat((channel_0, channel_1, channel_2, channel_3, channel_4), 1)
        fused_out = self.fusion_layers(x)
        
        return fused_out
    
##########################################################################################################################################################
# DnCNN and BUIFD (DnCNN-based):
class DnCNN_RL(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_RL, self).__init__()

        self.dncnn = DnCNN(channels=channels, num_of_layers=num_of_layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise
    
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise


class DnCNN_BUIFD(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_BUIFD, self).__init__()

        self.dncnn = DnCNN(channels=channels, num_of_layers=num_of_layers)

        self.noisecnn = NoiseCNN(channels=channels)
        self.FinalFusionLayers = FinalFusionLayers(channels=channels)

    def forward(self, x):
        noisy_input = x
        
        # PRIOR:
        noise = self.dncnn(x)
        prior = noisy_input - noise
        
        # NOISE LVL:
        noise_level = self.noisecnn(x)

        # FUSION:
        denoised_image = self.FinalFusionLayers(noisy_input, prior, noise_level)
        noise_out = noisy_input - denoised_image

        return noise_out, noise_level


##########################################################################################################################################################



####################################################################################################
# MemNet and its BUIFD version:

class MemNet(nn.Module):
    def __init__(self, in_channels, channels=20, num_memblock=6, num_resblock=4):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual
        
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(nn.Module):
    """ResidualBlock
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


# class BNReLUConv(nn.Sequential):
#     def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
#         super(BNReLUConv, self).__init__()
#         self.add_module('bn', nn.BatchNorm2d(in_channels))
#         self.add_module('relu', nn.ReLU(inplace=inplace))
#         self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

class BNReLUConv(nn.Module):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv = nn.Conv2d(in_channels, channels, k, s, p, bias=False)
    
    def forward(self,x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

        
class MemNet_BUIFD(nn.Module):
    def __init__(self, in_channels):
        super(MemNet_BUIFD, self).__init__()

        self.memnet = MemNet(in_channels=in_channels, channels=20, num_memblock=6, num_resblock=4)

        self.noisecnn = NoiseCNN(channels=in_channels)
        self.FinalFusionLayers = FinalFusionLayers(channels=in_channels)

    def forward(self, x):
        noisy_input = x
        
        # PRIOR:
        noise = self.memnet(x)
        prior = noisy_input - noise
        
        # NOISE LVL:
        noise_level = self.noisecnn(x)

        # FUSION:
        denoised_image = self.FinalFusionLayers(noisy_input, prior, noise_level)
        noise_out = noisy_input - denoised_image

        return noise_out, noise_level

####################################################################################################


####################################################################################################
# START(RIDNet)
def make_model(args, parent=False):
    return RIDNET(args)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = ops.BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        #self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        #g = self.g(r3)
        out = self.ca(r3)

        return out
        


class RIDNET(nn.Module):
    def __init__(self, in_channels):
        super(RIDNET, self).__init__()
        
        n_feats = 64
        kernel_size = 3
        reduction = 16
        rgb_range = 1
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        if in_channels == 1:
            rgb_mean = sum(rgb_mean)/3.0
            rgb_std = 1.0
        

        # Global avg pooling needed layers
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, in_channels=in_channels)       
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, sign=1, in_channels=in_channels)

        self.head = ops.BasicBlock(in_channels, n_feats, kernel_size, 1, 1)

        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)

        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size, 1, 1, 1)

    def forward(self, x):

        s = self.sub_mean(x)
        h = self.head(s)

        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)

        res = self.tail(b_out)
        out = self.add_mean(res)
        f_out = out + x 

        return f_out

# END(RIDNet)
####################################################################################################