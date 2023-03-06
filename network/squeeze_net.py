# coding:utf-8
import os
import torch
import torch.nn
import warnings
# import torchvision
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
# import torchvision.transforms as transforms
# from torchvision import datasets, models, transforms

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')


class Fire(torch.nn.Module):
    def __init__(self, inchn, sqzout_chn, exp1x1out_chn, exp3x3out_chn):
        super(Fire, self).__init__()
        self.inchn = inchn
        self.squeeze = torch.nn.Conv2d(inchn, sqzout_chn, kernel_size=1)
        self.squeeze_act = torch.nn.ReLU(inplace=True)
        self.bn1 = torch.nn.BatchNorm2d(exp1x1out_chn)
        self.expand1x1 = torch.nn.Conv2d(sqzout_chn, exp1x1out_chn, kernel_size=1)
        self.expand1x1_act = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(sqzout_chn, exp3x3out_chn, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(exp3x3out_chn)
        self.expand3x3_act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
            self.bn1(self.expand1x1_act(self.expand1x1(x))),
            self.bn2(self.expand3x3_act(self.expand3x3(x)))
        ], 1)


class Sqznet(torch.nn.Module):
    def __init__(self, num_class=6):
        super(Sqznet, self).__init__()
        self.num_class = num_class
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        final_conv = torch.nn.Conv2d(512, self.num_class, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            final_conv,
            torch.nn.ReLU(inplace=True),
            # torch.nn.AvgPool2d(13)
            torch.nn.AdaptiveAvgPool2d(1)
        )
        # parameters initial
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m is final_conv:
                    torch.nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x, im_info=None):
        x = self.features(x)
        x = self.classifier(x)
        return self.sf(x.view(x.size(0), self.num_class))


if __name__ == '__main__':
    net = Sqznet(6)
    print net
