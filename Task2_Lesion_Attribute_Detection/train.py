import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import SkinLesionDataset
from image_preprocess import ImagePreprocessor
from loss import HybridLoss, HybridLossDice
from tqdm import tqdm  # For progress bar
import torch.optim as optim
from model import UNet



def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    """
    Train the model for the given number of epochs.
    
    Arguments:
    - model: the U-Net model
    - dataloader: DataLoader for the training dataset
    - criterion: loss function (HybridLoss or HybridLossDice)
    - optimizer: optimizer (e.g., Adam, SGD)
    - device: 'cuda' or 'cpu'
    - num_epochs: number of epochs to train
    
    Returns:
    - model: trained model
    - training_losses: list of average training losses per epoch
    """
    model = model.to(device)
    training_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        i = 0
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            i += 1
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            print(f"\nBatch {i}, Loss: {loss.item():.4f}")

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Update the progress bar description
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        training_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return model, training_losses

def main():
    parser = argparse.ArgumentParser(description="Train a U-Net model for lesion attribute detection.")
    arg = parser.add_argument

    arg("--input_folder", type=str, default="./data/ISIC2018_Task1-2_Training_Input", help="Path to the folder containing input images.")
    arg("--output_dir", type=str,default ="./data/Input_Processed", help="Path to the output folder to save processed images.")
    arg("--ground_truth_dir", type=str, default="./data/ISIC2018_Task1-2_Training_GroundTruth", help="Path to the ground truth masks directory.")   
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg("--epochs", type=int, default=5, help="Number of training epochs.")
    arg("--batch_size", type=int, default=16, help="Batch size for training.")
    arg("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    arg("--output_model", type=str, default="./models/multi_task_unet.h5", help="Path to save the trained model.")
    
    args = parser.parse_args()
       
    dataset = SkinLesionDataset(image_dir=args.input_folder, ground_truth_dir=args.ground_truth_dir, target_size=args.size)
    print('Loading dataset...', dataset)
    print("Creating DataLoader...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Loading dataloader...", dataloader)
    model = UNet()
    print("Creating Model...")
    print("Model: ", model)
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model, training_losses = train_model(model, dataloader, criterion, optimizer, device, num_epochs=args.epochs)
    print("Training complete! with loss: ",training_losses)
    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()