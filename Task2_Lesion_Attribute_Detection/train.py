import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from image_preprocess import ImagePreprocessor
from model import create_unet_model
from loss import HybridLossDice


def train_model(args):
    # Initialize the image preprocessor
    processor = ImagePreprocessor(args.input_folder, args.input_folder, target_size=tuple(args.size))
    
    # Create dataloader
    dataset = processor.get_dataloader(batch_size=args.batch_size)
    
    # Initialize the model
    model = create_unet_model(in_channels=3, out_channels=1, init_features=32).to('cuda')
    
    # Define the loss function and optimizer
    criterion = HybridLossDice(weight_bce=1.0, weight_dice=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataset:
            images = images.to('cuda')
            masks = masks.to('cuda')
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(dataset)}")
    
    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
def main():
    parser = argparse.ArgumentParser(description="Train a U-Net model for lesion attribute detection.")
    arg = parser.add_argument

    arg("--input_folder", type=str, default="/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input", help="Path to the folder containing input images.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg("--epochs", type=int, default=50, help="Number of training epochs.")
    arg("--batch_size", type=int, default=16, help="Batch size for training.")
    arg("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    arg("--output_model", type=str, default="./models/multi_task_unet.h5", help="Path to save the trained model.")
    
    args = parser.parse_args()
    train_model(args)

    
if __name__ == "__main__":
    main()