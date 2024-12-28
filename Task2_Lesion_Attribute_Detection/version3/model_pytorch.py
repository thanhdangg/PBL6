import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        x = self.conv2(x)
        if self.activation == 'relu':
            x = F.relu(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, filters, kernel_size=(2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal'):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding)

    def forward(self, x):
        x = self.upconv(x)
        return x

class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTMBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.conv_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=batch_first, bias=bias)

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Apply ConvLSTM
        output, (hn, cn) = self.conv_lstm(x)
        
        if self.return_all_layers:
            return output, (hn, cn)
        else:
            return output[-1], (hn[-1], cn[-1])