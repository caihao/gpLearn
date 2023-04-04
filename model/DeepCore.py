import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepCore(nn.Module):
    def __init__(self):
        super(DeepCore, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.features=nn.Sequential(
            self.conv1,nn.ReLU(),self.pool1,
            self.conv2,nn.ReLU(),self.pool2,
            self.conv3,nn.ReLU(),self.pool3,
            self.conv4,nn.ReLU(),self.pool4,
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        x=torch.rand(1,1,128,128)
        input_size=self.features(x).size(-1)
        
        self.fc1 = nn.Linear(input_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.classifer=nn.Sequential(
            self.fc1,self.fc2
        )


    def forward(self, x):
        x=self.features(x)
        x=self.classifer(x)
        return x
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 15 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# model=DeepCore()
# x=torch.randn(2,1,128,128)
# y_hat=model(x)
# print(y_hat.shape)