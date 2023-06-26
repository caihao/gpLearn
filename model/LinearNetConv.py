import torch
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False, strides=1):
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

class ResNet_Block(nn.Module):
    def __init__(self,input_channel,hidden_size):
        super().__init__()
        def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
            blk=[]
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk
        
        b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3,  padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.features=nn.Sequential(
            b1,b2,b3,b4,b5,
            nn.AdaptiveAvgPool2d((hidden_size,hidden_size)),
            nn.Flatten()
        )
        # x=torch.randn(1,input_channel,input_size_x,input_size_y)
        # linear_input_size=self.features(x).size(-1)
        # self.classifer=nn.Sequential(
        #     nn.Linear(linear_input_size,hidden_size)
        # )

    def forward(self,x):
        x=self.features(x)
        # x=self.classifer(x)
        return x

class LinearNetConv(nn.Module):
    def __init__(self,input_channel,input_size_x,input_size_y,input_size_info,hidden_size_pic,hidden_size_info,output_size):
        super().__init__()
        self.single_conv=nn.Sequential(
            ResNet_Block(input_channel,hidden_size_pic)
        )
        self.single_linear=nn.Sequential(
            nn.Linear(input_size_info,hidden_size_info),
            nn.Flatten()
        )
        x=torch.rand((1,input_channel,input_size_x,input_size_y))
        conv_size=self.single_conv(x).size(-1)
        self.joint=nn.Sequential(
            nn.Linear(4*(conv_size+hidden_size_info),conv_size+hidden_size_info),nn.ReLU(),
            nn.Linear(conv_size+hidden_size_info,output_size)
        )

    def forward(self,X,Y):
        x1,x2,x3,x4=torch.split(X,1,dim=1)
        y1,y2,y3,y4=torch.split(Y,1,dim=1)
        return self.joint(torch.cat((
            self.single_conv(x1),self.single_conv(x2),self.single_conv(x3),self.single_conv(x4),
            self.single_linear(y1),self.single_linear(y2),self.single_linear(y3),self.single_linear(y4)
        ),dim=1))

