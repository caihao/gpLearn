import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            ))

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class MSDNet(nn.Module):
    def __init__(self, input_channel=1, input_size_x=128, input_size_y=128, growth_rate=32, n_layers=4, n_blocks=4, num_classes=2, init_weights=False):
        super(MSDNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1, bias=False)
        self.dense_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()
        self.growth_rate = growth_rate
        in_channels = 64
        for i in range(n_blocks):
            self.dense_blocks.append(DenseBlock(in_channels, growth_rate, n_layers))
            in_channels += n_layers * growth_rate
            if i != n_blocks - 1:
                out_channels = int(in_channels / 2)
                self.transition_blocks.append(TransitionBlock(in_channels, out_channels))
                in_channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x=torch.randn(1,input_channel,input_size_x,input_size_y)
        self.fc = nn.Linear(120, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        for block, trans in zip(self.dense_blocks, self.transition_blocks):
            x = block(x)
            x = trans(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
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
