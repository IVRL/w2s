import math
import torch
import torch.nn as nn
import torchvision
import functools
from torch.nn import init
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=False):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 1, 0, bias))  # kernal_size is 1
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 1, 0, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def default_conv(in_channels, out_channels, kernel_size, padding, bias=False, init_scale=0.1):
    basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
    basic_conv.weight.data *= init_scale
    if basic_conv.bias is not None:
        basic_conv.bias.data.zero_()
    return basic_conv

def default_Linear(in_channels, out_channels, bias=False):
    basic_Linear = nn.Linear(in_channels, out_channels, bias=bias)
    # nn.init.xavier_normal_(basic_Linear.weight.data)
    nn.init.kaiming_normal_(basic_Linear.weight.data, a=0, mode='fan_in')
    basic_Linear.weight.data *= 0.1
    if basic_Linear.bias is not None:
        basic_Linear.bias.data.zero_()
    return basic_Linear

class TuningBlock(nn.Module):
    def __init__(self, input_size):
        super(TuningBlock, self).__init__()
        self.conv0 = default_conv(in_channels=input_size, out_channels=input_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=input_size, out_channels=input_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        return out


class TuningBlockModule(nn.Module):
    def __init__(self, channels=64, num_blocks=5, task_type='sr', upscale=4):
        super(TuningBlockModule, self).__init__()
        self.num_channels = channels
        self.task_type = task_type
        # define control variable
        self.control_alpha = nn.Sequential(
            default_Linear(512, 256, bias=False),
            default_Linear(256, 128, bias=False),
            default_Linear(128, channels, bias=False)
        )
        self.adaptive_alpha = nn.ModuleList(
            [nn.Sequential(
                default_Linear(channels, channels, bias=False),
                default_Linear(channels, channels, bias=False)
            ) for _ in range(num_blocks)]
        )
        self.tuning_blocks = nn.ModuleList(
            [TuningBlock(channels) for _ in range(num_blocks)]
        )
        if self.task_type == 'sr':
            self.tuning_blocks.append(nn.Sequential(
                default_conv(in_channels=channels, out_channels=channels,
                             kernel_size=3, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, channels, bias=False, act=False),
            ))
            self.adaptive_alpha.append(nn.Sequential(
                default_Linear(channels, channels, bias=False),
                default_Linear(channels, channels, bias=False)
            ))

    def forward(self, x, alpha, number=0):
        input_alpha = self.control_alpha(alpha)
        tuning_f = self.tuning_blocks[number](x)
        ad_alpha = self.adaptive_alpha[number](input_alpha)
        ad_alpha = ad_alpha.view(-1, self.num_channels, 1, 1)
        return tuning_f * ad_alpha, torch.ones_like(ad_alpha).cuda()-ad_alpha

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, padding=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x

        return res

class MainNet(nn.Module):
    def __init__(self, n_colors, out_nc, num_channels, num_blocks, task_type='sr', upscale=4):
        super(MainNet, self).__init__()

        self.task_type = task_type
        # define head
        self.head = default_conv(in_channels=n_colors, out_channels=num_channels,
                                 kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.body = nn.ModuleList(
            [ResBlock(default_conv,
                      n_feats=num_channels, kernel_size=3, act=nn.ReLU(True), res_scale=1
                      ) for _ in range(num_blocks)]
        )

        if self.task_type == 'sr':
            self.tail = nn.Sequential(
                default_conv(in_channels=num_channels, out_channels=num_channels,
                             kernel_size=3, padding=1, bias=False, init_scale=0.1),
                Upsampler(default_conv, upscale, num_channels, act=False),
            )

        self.end = default_conv(in_channels=num_channels, out_channels=out_nc,
                                kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        output = self.head(x)
        head_f = output
        for mbody in self.body:
            output = mbody(output)
        if self.task_type == 'sr':
            output = self.tail(output)
        output = self.end(output + head_f)
        return output

class CFSNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_channels=64, num_main_blocks=30, num_tuning_blocks=30, upscale=2,task_type='sr'):
        super(CFSNet, self).__init__()
        self.num_main_blocks = num_main_blocks
        self.num_tuning_blocks = num_tuning_blocks
        self.task_type = task_type
        self.main = MainNet(in_channel, out_channel, num_channels, num_main_blocks, task_type=task_type, upscale=upscale)
        self.tuning_blocks = TuningBlockModule(channels=num_channels, num_blocks=num_tuning_blocks,task_type=task_type, upscale=upscale)

    def forward(self, x, control_vector=None):
        out = self.main.head(x)
        head_f=out
        if (control_vector == None):
            control_vector = torch.ones(x.shape[0], 512) * 1.
        for i, body in enumerate(self.main.body):
            tun_out, tun_alpha = self.tuning_blocks(x=out, alpha= control_vector.cuda(),number=i)
            out = body(out) * tun_alpha + tun_out

        if self.task_type == 'sr':
            tun_out,tun_alpha = self.tuning_blocks(x=out+head_f, alpha= control_vector.cuda(), number=-1)
            out = self.main.tail(out+head_f)
            out = self.main.end(out*tun_alpha + tun_out)
        else:
            out = self.main.end(out+head_f)
        return out