import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding_size = config.IM_EmbeddingSize

        # contracting path
        self.conv1 = ConvBlock(4, 64)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(256, 512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(512, 1024)

        # expanding path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(128, config.SegClasses, kernel_size=1, padding=0)

        # image embedding: deep layers
        shape = config.FeatureShape
        self.linear = nn.Linear(in_features=(shape[0] * shape[1] * shape[2]), out_features=self.embedding_size)

    def forward(self, x):   # BRATS image: [1, 4, 240, 240]
        # contracting
        x_240 = self.conv1(x)
        x_120 = self.mp1(x_240)
        x_120 = self.conv2(x_120)
        x_60 = self.mp2(x_120)
        x_60 = self.conv3(x_60)
        x_30 = self.mp3(x_60)
        x_30 = self.conv4(x_30)
        x_15 = self.mp4(x_30)
        x_15 = self.conv5(x_15)

        # expanding
        _x_30 = torch.cat([self.up1(x_15), x_30], dim=1)
        _x_30 = self.deconv1(_x_30)
        _x_60 = torch.cat([self.up2(_x_30), x_60], dim=1)
        _x_60 = self.deconv2(_x_60)
        _x_120 = torch.cat([self.up3(_x_60), x_120], dim=1)
        _x_120 = self.deconv3(_x_120)
        _x_240 = torch.cat([self.up4(_x_120), x_240], dim=1)
        output = self.outconv(_x_240)

        feature = x_15
        embedding = self.linear(feature.view(-1))
        return embedding.view(1, -1), output