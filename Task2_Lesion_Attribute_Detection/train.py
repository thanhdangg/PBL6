import argparse
import os
from matplotlib import pyplot as plt
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



def train_model(model, train_loader,val_loader, criterion, optimizer, device, num_epochs=25, output_model_path="./models/multi_task_unet.h5"):
    """
    Train the model for the given number of epochs.
    
    Arguments:
    - model: the U-Net model
    - train_loader: DataLoader for the training dataset
    - val_loader: DataLoader for the validation dataset
    - criterion: loss function (HybridLoss or HybridLossDice)
    - optimizer: optimizer (e.g., Adam, SGD)
    - device: 'cuda' or 'cpu'
    - num_epochs: number of epochs to train
    - output_model_path: path to save the trained model

    Returns:
    - model: trained model
    - training_losses: list of average training losses per epoch
    """
    model = model.to(device)
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Update the progress bar description
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        print("Evaluating model on validation set...")
        # Validate the model
        val_loss = evaluate_model(model, val_loader, criterion, device)
        validation_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = os.path.dirname(output_model_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), output_model_path)
            print(f"Best model saved to {output_model_path}")
    
    return model, training_losses, validation_losses

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a validation or test dataset.
    
    Arguments:
    - model: trained U-Net model
    - dataloader: DataLoader for the validation or test dataset
    - criterion: loss function (HybridLoss or HybridLossDice)
    - device: 'cuda' or 'cpu'
    
    Returns:
    - avg_loss: average loss on the validation set
    """
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Update running loss
            running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    # print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss

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
    arg("--val_input_folder", type=str, default="./data/ISIC2018_Task1-2_Validation_Input", help="Path to the folder containing validation input images.")
    arg("--val_ground_truth_dir", type=str, default="./data/ISIC2018_Task1-2_Validation_GroundTruth", help="Path to the validation ground truth masks directory.")
    arg("--plot_output_dir", type=str, default="./plots", help="Directory to save the loss plot.")

    
    args = parser.parse_args()
       
    train_dataset = SkinLesionDataset(image_dir=args.input_folder, ground_truth_dir=args.ground_truth_dir, target_size=args.size)
    val_dataset = SkinLesionDataset(image_dir=args.val_input_folder, ground_truth_dir=args.val_ground_truth_dir, target_size=args.size)

    print('Loading dataset...')
    print("Creating DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Loading dataloader...")
    model = UNet()
    print("Creating Model...")
    print("Model: ", model)
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model, training_losses, validation_losses = train_model(model, train_loader,val_loader, criterion, optimizer, device, num_epochs=args.epochs, output_model_path=args.output_model)
    print("Training complete! with loss: ",training_losses)
    
    # Ensure the plot output directory exists
    if not os.path.exists(args.plot_output_dir):
        os.makedirs(args.plot_output_dir)
    
    plot_path = os.path.join(args.plot_output_dir, 'loss_plot.png')

    # Plot the training and validation losses
    plt.figure()
    plt.plot(range(1, args.epochs + 1), training_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss plot saved to {plot_path}")

    
if __name__ == "__main__":
    main()