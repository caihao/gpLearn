import torch
import torch.nn as nn

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

# 定义残差网络
class LinearResNet(nn.Module):
    def __init__(self, input_channal, input_size_x, input_size_y, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        feature=nn.Sequential(self.flatten)
        x=torch.rand(1,input_channal,input_size_x,input_size_y)
        input_size=feature(x).size(-1)
        hidden_size=int(input_size*0.5)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.residual1 = ResidualBlock(hidden_size, hidden_size)
        self.residual2 = ResidualBlock(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.fc2(out)
        return out

