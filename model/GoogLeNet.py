# 并行网络(GoogLeNet)
# 适用于能量分辨 train_type=energy

import torch
import torch.nn as nn
from torch.nn import functional as F

class Single(nn.Module):
    def __init__(self,input_size_x:int,input_size_y:int,hidden_x_output:int,hidden_y_output:int,output_size:int):
        super().__init__()
        self.features_x=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size_x,hidden_x_output),
            nn.ReLU()
        )
        self.features_y=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size_y,hidden_y_output),
            nn.ReLU()
        )
        self.classifer=nn.Sequential(
            nn.Linear(hidden_x_output+hidden_y_output,output_size),
            nn.ReLU()
        )
        
    def forward(self,X):
        hidden_x=self.features_x(X[0])
        hidden_y=self.features_y(X[1])
        hidden=torch.cat((hidden_x,hidden_y),dim=1)
        return self.classifer(hidden)
    
class Joint(nn.Module):
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        super().__init__()
        self.features=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU()
        )
        self.classifer=nn.Sequential(
            nn.Linear(4*hidden_size,hidden_size),nn.ReLU(),
            nn.Linear(hidden_size,int(0.25*hidden_size)),nn.ReLU(),
            nn.Linear(int(0.25*hidden_size),output_size)
        )

    def forward(self,X):
        hidden=torch.cat((self.features(X[0]),self.features(X[1]),self.features(X[2]),self.features(X[3])),dim=1)
        return self.classifer(hidden)


class GoogLeNet(nn.Module):
    def __init__(self,input_size_x:int,input_size_y:int,input_size_info:int,output_size:int):
        super().__init__()
        self.features=nn.Sequential(
            Single(input_size_x*input_size_y,input_size_info,4096,16,2048)
        )
        self.classifer=nn.Sequential(
            Joint(2048,1024,output_size)
        )

    def forward(self,X,Y):
        x1,x2,x3,x4=torch.split(X,1,dim=1)
        y1,y2,y3,y4=torch.split(Y,1,dim=1)
        a=self.features([x1,y1])
        b=self.features([x2,y2])
        c=self.features([x3,y3])
        d=self.features([x4,y4])
        return self.classifer([a,b,c,d])
