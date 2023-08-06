import torch
import torch.nn as nn
from torch.nn import functional as F

class AngleNet(nn.Module):
    def __init__(self,input_channel,input_size_x,input_size_y,output_size):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv3d(in_channels=input_channel,out_channels=96,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(96),
            nn.Conv3d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(192),
            nn.Conv3d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(384),
            nn.Conv3d(in_channels=384,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(512),
            nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.AdaptiveAvgPool3d((4,4,4)),
            nn.Flatten()
        )
        x=torch.rand((1,4,input_size_x,input_size_y,4))
        linear_size=self.features(x).size(-1)
        self.classifer=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_size,256),
            nn.Dropout(0.5),
            nn.Linear(256,64),
            nn.Dropout(0.5),
            nn.Linear(64,output_size),
        )

    def forward(self,X):
        Y=self.features(X)
        return self.classifer(Y)



class Residual3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResAngleNet(nn.Module):
    def __init__(self, input_channel, input_size_x, input_size_y, output_size):
        super().__init__()
        self.features = nn.Sequential(
            Residual3D(input_channel, 96),
            Residual3D(96, 192),
            Residual3D(192, 384),
            Residual3D(384, 512),
            Residual3D(512, 512),
            nn.AdaptiveAvgPool3d((4,4,4)),
            nn.Flatten()
        )
        x = torch.rand((1, input_channel, input_size_x, input_size_y, 4))
        linear_size = self.features(x).size(-1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_size, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        y = self.features(x)
        return self.classifier(y)

class Residual2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResAngleNet2D(nn.Module):
    def __init__(self, input_channel, input_size_x, input_size_y, output_size):
        super().__init__()
        self.features = nn.Sequential(
            Residual2D(input_channel, 96),
            Residual2D(96, 192),
            Residual2D(192, 384),
            Residual2D(384, 512),
            Residual2D(512, 512),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten()
        )
        x = torch.rand((1, input_channel, input_size_x, input_size_y))
        linear_size = self.features(x).size(-1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_size, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        y = self.features(x)
        return self.classifier(y)
