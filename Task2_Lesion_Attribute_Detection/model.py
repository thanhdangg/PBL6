import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_out = self.conv_block(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.conv_block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)  # Concatenated output will have in_channels * 2
    
    def forward(self, x, enc_output):
        x = self.upconv(x)
        # Concatenate along the channel axis (dim=1)
        x = torch.cat((x, enc_output), dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(UNet, self).__init__()

        # Encoder blocks
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 1024)

        # Decoder blocks
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        enc1, pool1 = self.encoder1(x)
        enc2, pool2 = self.encoder2(pool1)
        enc3, pool3 = self.encoder3(pool2)
        enc4, pool4 = self.encoder4(pool3)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder path
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        
        out = self.final_conv(dec1)
        return out
    
# def create_unet_model(in_channels=3, out_channels=1, init_features=32):
#     model = UNet(in_channels, out_channels, init_features)
#     return model


# if __name__ == "__main__":
#     # Create the U-Net model
#     model = UNet(in_channels=3, out_channels=1)

#     # Test with a random input tensor
#     input_image = torch.randn((1, 3, 256, 256))  # Example input (batch_size=1, channels=3, height=256, width=256)
#     output_mask = model(input_image)

#     print(f"Output shape: {output_mask.shape}")  # Expected output shape: [1, 1, 256, 256]
