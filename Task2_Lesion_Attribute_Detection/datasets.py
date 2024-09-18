import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, ground_truth_dir, target_size = (256,256)):
        self.image_dir = image_dir
        self.ground_truth_dir = ground_truth_dir
        self.target_size = target_size
        self.attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        # Define transformations for input images
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # Define augmentation (optional)
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        
        self.to_pil = transforms.ToPILImage()
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_id = image_filename.split('.')[0]  # Extract image ID (e.g., ISIC_<image_id>)
        
        # Load the input image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
        
        # Load the ground truth masks for each attribute
        masks = []
        for attribute in self.attributes:
            mask_filename = f"{image_id}_attribute_{attribute}.png"
            mask_path = os.path.join(self.ground_truth_dir, mask_filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')  # Load mask as grayscale
                mask = mask.resize(self.target_size)       # Resize the mask
                mask = transforms.ToTensor()(mask)         # Convert mask to tensor
                masks.append(mask)
            else:
                # If mask does not exist, create an empty mask
                masks.append(torch.zeros(1, *self.target_size))
        
        # Stack masks along channel dimension
        masks = torch.cat(masks, dim=0)
        
        return image, masks
