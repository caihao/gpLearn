# 残差网络(ResNet)
# 适用于粒子分辨 train_type=particle

import torch
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet_old(nn.Module):
    def __init__(self,input_channel,input_size_x,input_size_y,output_size,init_weights=False):
        super().__init__()
        def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
            blk=[]
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk
        
        b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 1, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 1))
        b4 = nn.Sequential(*resnet_block(128, 256, 1))
        b5 = nn.Sequential(*resnet_block(256, 512, 1))

        self.features=nn.Sequential(
            b1,b2,b3,b4,b5,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        x=torch.randn(1,input_channel,input_size_x,input_size_y)
        linear_input_size=self.features(x).size(-1)
        # print(self.features(x).shape)
        self.classifer=nn.Sequential(
            nn.Linear(linear_input_size,output_size)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x=self.features(x)
        x=self.classifer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
