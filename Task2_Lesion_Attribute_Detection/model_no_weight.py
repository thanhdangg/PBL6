import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SqueezeExcite, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1)

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        return conv1, conv2, conv3, pool3

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.up6 = nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2)
        self.adjust_channels = nn.Conv2d(256, 512, kernel_size=1)  
        self.conv6 = ConvBlock(512, 256)
        self.se6 = SqueezeExcite(256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(256, 128)
        self.se7 = SqueezeExcite(128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(128, 64)
        self.se8 = SqueezeExcite(64)

    def forward(self, conv1, conv2, conv3, drop4_3):
        drop4_3 = self.adjust_channels(drop4_3)          
        up6 = self.up6(drop4_3)
        merge6 = torch.cat([up6, conv3], dim=1)
        conv6 = self.conv6(merge6)
        conv6 = self.se6(conv6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv2], dim=1)
        conv7 = self.conv7(merge7)
        conv7 = self.se7(conv7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv1], dim=1)
        conv8 = self.conv8(merge8)
        conv8 = self.se8(conv8)

        return conv8

class SEDU_Net_D3(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(SEDU_Net_D3, self).__init__()
        self.encoder = Encoder(in_channels)
        self.dense_block = DenseBlock(256, 256)
        self.decoder = Decoder(512)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        conv1, conv2, conv3, pool3 = self.encoder(x)
        conv4_1 = self.dense_block(pool3)
        conv4_2 = self.dense_block(conv4_1)
        conv4_3 = self.dense_block(conv4_2)
        drop4_3 = F.dropout(conv4_3, p=0.5)

        conv8 = self.decoder(conv1, conv2, conv3, drop4_3)
        output = self.final_conv(conv8)
        return torch.sigmoid(output)