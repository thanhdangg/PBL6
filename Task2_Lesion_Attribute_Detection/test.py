import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import UNet
from loss import HybridLoss
import numpy as np
import matplotlib.pyplot as plt

class TestSkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size, attributes):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.attributes = attributes
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        masks = []
        for attribute in self.attributes:
            # mask_filename = image_filename.replace('.jpg', f'_attribute_{attribute}.png').replace('.png', f'_attribute_{attribute}.png')
            mask_filename = image_filename.replace('.jpg', f'_attribute_{attribute}.png')

            # mask_filename = image_filename
            mask_path = os.path.join(self.mask_dir, mask_filename)
            mask = Image.open(mask_path).convert('L')
            mask = self.transform(mask)
            mask = (mask > 0).float()  # Convert mask to binary
            masks.append(mask)

        masks = torch.cat(masks, dim=0)  # Concatenate masks along the channel dimension

        return image, masks, image_filename

def evaluate_model(model, dataloader, device, output_dir):
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    criterion = HybridLoss()
    running_loss = 0.0

    os.makedirs(output_dir, exist_ok=True)
    
    attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]


    with torch.no_grad():  # Disable gradient computation
        for images, masks, filenames in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Save predicted masks
            for i in range(outputs.size(0)):
                for j, attribute in enumerate(attributes):
                    pred_mask = outputs[i, j].cpu().numpy().squeeze()
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
                    pred_image = Image.fromarray(pred_mask)
                    pred_image.save(os.path.join(output_dir, f"{filenames[i]}_{attribute}.png"))

    avg_loss = running_loss / len(dataloader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Test a U-Net model for lesion attribute detection.")
    arg = parser.add_argument

    arg("--input_folder", type=str, default="./data/ISIC2018_Task1-2_Test_Input", help="Path to the folder containing input images.")
    arg("--mask_folder", type=str, default="./data/ISIC2018_Task1-2_Test_Masks", help="Path to the folder containing mask images.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg("--batch_size", type=int, default=16, help="Batch size for testing.")
    arg("--model_path", type=str, default="./models/multi_task_unet.h5", help="Path to the trained model.")
    arg("--output_dir", type=str, default="./predictions", help="Directory to save the predicted masks.")
    
    args = parser.parse_args()
    
    attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
    dataset = TestSkinLesionDataset(image_dir=args.input_folder, mask_dir=args.mask_folder, target_size=args.size, attributes=attributes)
    print('Loading dataset...', dataset)
    print("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Loading dataloader...", dataloader)
    
    model = UNet()
    model.load_state_dict(torch.load(args.model_path))
    print("Model loaded from", args.model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    avg_loss = evaluate_model(model, dataloader, device, args.output_dir)
    print(f"Average Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()