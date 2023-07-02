import torch
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=1,stride=2)
        )

        # self.classifer=nn.Sequential(
        #     nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1,stride=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.features=nn.Sequential(
        #     nn.Conv2d(input_channels,out_channels,kernel_size=1)
        # )

    def forward(self,X):
        Y=F.relu(self.conv1(X))
        Y=self.conv2(Y)
        X=self.conv3(X)
        Y=Y+X
        return F.relu(Y)

        # Y1=self.classifer(X)
        # Y2=self.features(X)
        # return  F.relu(Y1+Y2)

class ResNet_custom(nn.Module):
    def __init__(self,input_channels,input_size_x,input_size_y,output_size):
        super().__init__()
        self.classifer=nn.Sequential(
            # nn.Conv2d(input_channels,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            Residual(input_channels,64),
            Residual(64,128),
            Residual(128,256),
            Residual(256,512),
            Residual(512,1024),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        x=torch.rand((1,input_channels,input_size_x,input_size_y))
        linear_size=self.classifer(x).size(-1)
        self.features=nn.Sequential(
            nn.Linear(linear_size,256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,output_size)
            # nn.Linear(linear_size,output_size)
        )

    def forward(self,X):
        Y=self.classifer(X)
        Y=self.features(Y)
        return Y
