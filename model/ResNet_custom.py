import torch
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.classifer=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.features=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=1)
        )

    def forward(self,X):
        Y1=self.classifer(X)
        Y2=self.features(X)
        return  F.relu(Y1+Y2)

class ResNet_custom(nn.Module):
    def __init__(self,input_channels,input_size_x,input_size_y,output_size):
        super().__init__()
        self.classifer=nn.Sequential(
            Residual(input_channels,16),
            Residual(16,64),
            Residual(64,256),
            Residual(256,1024),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten()
        )
        x=torch.rand((1,input_channels,input_size_x,input_size_y))
        linear_size=self.classifer(x).size(-1)
        self.features=nn.Sequential(
            nn.Linear(linear_size,2048),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,output_size)
        )

    def forward(self,X):
        Y=self.classifer(X)
        Y=self.features(Y)

        # Y=Y[:,0]/Y[:,1]
        # print(Y)
        # return Y.unsqueeze(1)
    
        return Y
