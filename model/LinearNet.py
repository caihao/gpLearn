# 线性网络(LinearNet)

import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearNet(nn.Module):
    def __init__(self,input_channel,input_size_x,input_size_y,output_size,init_weights=False):
        super().__init__()
        # x=torch.randn(1,input_channel,input_size_x,input_size_y)
        input_size=input_channel*input_size_x*input_size_y
        self.features=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,int(0.5*input_size)),nn.ReLU(),
            nn.Linear(int(0.5*input_size),int(0.25*input_size)),nn.ReLU(),
            nn.Linear(int(0.25*input_size),output_size),nn.ReLU()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        return self.features(x)

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

