import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, down_sample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, stride, 1)
        self.bn = nn.BatchNorm2d(channels_out)
        self.down_sample = down_sample

    def forward(self, x):
        prev = x
        x = F.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        if self.down_sample is not None:
            prev = self.down_sample(prev)
        x += prev
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.residual_blocks = 2
        self.channels_in = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self.make_layer(64, 1)
        self.layer2 = self.make_layer(128, 2)
        self.layer3 = self.make_layer(256, 2)
        self.layer4 = self.make_layer(512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 200)
        self.softmax = nn.Softmax(dim=1)

    def make_layer(self, channels_out, stride, down_sample=None):
        layers = []

        if stride != 1:
            down_sample = nn.Sequential(nn.Conv2d(
                self.channels_in, channels_out, 3, stride, 1), nn.BatchNorm2d(channels_out))

        layers.append(Block(self.channels_in, channels_out, stride, down_sample))
        self.channels_in = channels_out

        for i in range(self.residual_blocks - 1):
            layers.append(Block(self.channels_in, channels_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        #x = self.softmax(x)
        return x
