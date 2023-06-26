import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# class PointNet(nn.Module):
#     def __init__(self,input_channels,input_size_x,input_size_y,output_size):
#         super().__init__()
#         self.features=nn.Sequential(
#             nn.Conv2d(input_channels,4,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(4),nn.ReLU(),
#             # nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
#             # nn.Conv2d(16,128,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
#             nn.Conv2d(4,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
#             nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
#             nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4,4)),
#             nn.Flatten()
#         )
#         x=torch.rand((1,input_channels,input_size_x,input_size_y),dtype=torch.float32)
#         linear_size=self.features(x).size(-1)
#         self.classifer=nn.Sequential(
#             nn.Linear(linear_size,512),nn.ReLU(),
#             nn.Linear(512,128),nn.ReLU(),
#             nn.Linear(128,output_size)
#         )

#     def forward(self,X):
#         x=self.features(X)
#         x=self.classifer(x)
#         return x


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

class AngleNet(nn.Module):
    def __init__(self,input_channels,input_size_x,input_size_y):
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
            nn.Linear(linear_size,2048),nn.ReLU(),
            nn.Linear(2048,512),nn.ReLU(),
            nn.Linear(512,128),nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,X):
        Y=self.classifer(X)
        Y=self.features(Y)
        # print(Y)
        return Y


        Y=Y[:,0]/Y[:,1]
        return Y.unsqueeze(1)

        Y=torch.atan2(Y[:,0],Y[:,1])*180/math.pi
        Y=Y.unsqueeze(1)
        return Y






# class Residual(nn.Module):
#     def __init__(self,input_channels,num_channels,use_1x1conv=False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(num_channels)
#         self.bn2 = nn.BatchNorm2d(num_channels)

#     def forward(self,X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)

# class PointNet(nn.Module):
#     def __init__(self,input_channel,input_size_x,input_size_y,output_size,init_weights=False):
#         super().__init__()
#         def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
#             blk=[]
#             for i in range(num_residuals):
#                 if i == 0 and not first_block:
#                     blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
#                 else:
#                     blk.append(Residual(num_channels, num_channels))
#             return blk
        
#         b1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3,  padding=1))
#         b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
#         b3 = nn.Sequential(*resnet_block(64, 128, 2))
#         b4 = nn.Sequential(*resnet_block(128, 256, 2))
#         b5 = nn.Sequential(*resnet_block(256, 512, 2))

#         self.features=nn.Sequential(
#             b1,b2,b3,b4,b5,
#             nn.AdaptiveAvgPool2d((4,4)),
#             nn.Flatten()
#         )
#         x=torch.randn(1,input_channel,input_size_x,input_size_y)
#         linear_input_size=self.features(x).size(-1)
#         self.classifer=nn.Sequential(
#             nn.Linear(linear_input_size,1024),nn.ReLU(),
#             nn.Linear(1024,128),nn.ReLU(),
#             nn.Linear(128,output_size)
#         )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self,x):
#         x=self.features(x)
#         x=self.classifer(x)
#         return x
