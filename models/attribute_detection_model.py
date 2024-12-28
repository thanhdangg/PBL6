import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import requests
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()
# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super().__init__()
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        super().__init__()
        self.encoder = models.vgg16(pretrained=(pretrained == 'vgg')).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # Encoder Layers
        self.conv1 = self.encoder[:4]
        self.conv2 = self.encoder[5:9]
        self.conv3 = self.encoder[10:16]
        self.conv4 = self.encoder[17:23]
        self.conv5 = self.encoder[24:30]

        # Decoder Layers
        self.center = DecoderBlock(512, num_filters * 16, num_filters * 8)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 16, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 16, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 8, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 4, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, conv1], dim=1))
        return self.final(dec1)
def getting_attribute_detection_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet16(num_classes=5, pretrained='vgg')  # Assuming UNet16 model class is defined
    model = nn.DataParallel(model)
    return model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def parse_image_from_url(img_url):
    response = requests.get(img_url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def process_and_upload_mask(mask, attr_type):
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)
    upload_result = cloudinary.uploader.upload(buffer, resource_type="image")
    return attr_type, upload_result["url"]

def get_masks_from_model(image, model, device="cpu"):
    image = image.to(device)
    with torch.inference_mode():
        output = model(image)
        return (output > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

def detect_attributes(img_url, model, device="cpu"):
    image = preprocess_image(parse_image_from_url(img_url)).to(device)
    masks = get_masks_from_model(image, model, device)
    attr_types = ['Pigment Network', 'Negative Network', 'Streaks', 'Milia-like Cyst', 'Globules']
    
    results = {}
    for i, attr_type in enumerate(attr_types):
        _, url = process_and_upload_mask(masks[i], attr_type)
        results[attr_type] = url
    
    return results
