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

def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) score for binary masks.
    Args:
        pred (torch.Tensor): The predicted mask of shape [batch, 1, H, W].
        target (torch.Tensor): The ground truth mask of shape [batch, 1, H, W].
        threshold (float): Threshold to binarize predicted masks.
        smooth (float): Small value to avoid division by zero.
    
    Returns:
        iou (float): IoU score.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()  # Binarize the prediction

    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute the Dice score for binary masks.
    Args:
        pred (torch.Tensor): The predicted mask of shape [batch, 1, H, W].
        target (torch.Tensor): The ground truth mask of shape [batch, 1, H, W].
        threshold (float): Threshold to binarize predicted masks.
        smooth (float): Small value to avoid division by zero.
    
    Returns:
        dice (float): Dice score.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def is_blank_mask(pred_mask, threshold=0.5):
    """
    Check if mask is blank (all pixels are 0).
    pred_mask: model prediction tensor 
    threshold: threshold to classify pixels as 0 or 1
    """
    # Convert tensor to numpy array
    pred_mask_np = pred_mask.cpu().detach().numpy()
    
    # Convert  to binary mask  (0 or 1)
    binary_mask = (pred_mask_np > threshold).astype(np.uint8)
    
    # Check if all is 0
    return np.all(binary_mask == 0)

def train_model(model, train_loader,val_loader, criterion, optimizer, device, num_epochs=25, output_model_path="./models/multi_task_unet.h5", output_checkpoint_dir= "/content/drive/MyDrive/best_model.h5"):
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
    - output_checkpoint_dir: path to save the checkpoint model

    Returns:
    - model: trained model
    - training_losses: list of average training losses per epoch
    """
    model = model.to(device)
    training_losses = []
    validation_losses = []
    
    training_iou = []
    vaidation_iou = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_iou = 0.0


        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
                    
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # check if mask is blank image then skip
            valid_masks = []
            valid_outputs = []
            
            for i, mask in enumerate(masks):
                mask_filename = train_loader.dataset.image_filenames[i]
                if "blank" in mask_filename: 
                    continue
                valid_masks.append(mask)
                valid_outputs.append(outputs[i])
            
            if len(valid_masks) > 0:
                valid_masks = torch.stack(valid_masks)
                valid_outputs = torch.stack(valid_outputs)
                
                # Compute loss
                loss = criterion(valid_outputs, valid_masks)
                loss.backward()
                optimizer.step()      
             
                # Update running loss
                running_loss += loss.item() * images.size(0)
                
                # Compute IoU for training batch
                batch_iou = 0.0
                for i in range(outputs.size(0)):
                    batch_iou += iou_score(outputs[i], masks[i])
                
                running_iou += batch_iou / outputs.size(0)


            # Update the progress bar description
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader)

        training_losses.append(epoch_loss)
        training_iou.append(epoch_iou)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, iou: {epoch_iou:.4f}")
        
        print("Evaluating model on validation set...")
        # Validate the model
        val_loss, val_iou = evaluate_model(model, val_loader, criterion, device)
        validation_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}, Validation iou: {val_iou:.4f}")
        
        output_dir = os.path.dirname(output_model_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_checkpoint_dir = os.path.dirname(output_checkpoint_dir)
        if not os.path.exists(output_checkpoint_dir):
            os.makedirs(output_checkpoint_dir)
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)
            torch.save(model.state_dict(), output_checkpoint_dir)
            
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
    running_iou = 0.0

    with torch.no_grad():  # Disable gradient computation
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            
            valid_masks = []
            valid_outputs = []
            for i, mask in enumerate(masks):
                if not is_blank_mask(mask):  # Check if mask is not blank
                    valid_masks.append(mask)
                    valid_outputs.append(outputs[i])
            
            if len(valid_masks) > 0:
                valid_masks = torch.stack(valid_masks)
                valid_outputs = torch.stack(valid_outputs)
                
                # Compute loss
                loss = criterion(valid_outputs, valid_masks)
                running_loss += loss.item()
                
                # Compute IoU for validation batch
                batch_iou = 0.0
                for i in range(outputs.size(0)):
                    batch_iou += iou_score(outputs[i], masks[i])
                running_iou += batch_iou / outputs.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    avg_iou = running_iou / len(dataloader)
    # print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss, avg_iou

def main():    
    parser = argparse.ArgumentParser(description="Train a U-Net model for lesion attribute detection.")
    arg = parser.add_argument

    # train datasets
    arg("--input_folder", type=str, default="./data/Processed_Train_Input", help="Path to the folder containing input images.")
    arg("--ground_truth_dir", type=str, default="./data/Processed_Train_GroundTruth", help="Path to the ground truth masks directory.")   
    
    # validation datasets
    arg("--val_input_folder", type=str, default="./data/Processed_Val_Input", help="Path to the folder containing validation input images.")
    arg("--val_ground_truth_dir", type=str, default="./data/Processed_Val_GroundTruth", help="Path to the validation ground truth masks directory.")

    # output train
    arg("--output_model", type=str, default="./models/multi_task_unet.h5", help="Path to save the trained model.")
    arg("--output_model_checkpoint_drive", type=str, default="/content/drive/MyDrive/best_model.h5", help="Path to save checkpoint the trained model.")
    arg("--plot_output_dir", type=str, default="./plots", help="Directory to save the loss plot.")
    arg("--resume_model", type=str, default=None, help="Path to a saved model to resume training.")

    # hyperparameter
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg("--epochs", type=int, default=2, help="Number of training epochs.")
    arg("--batch_size", type=int, default=16, help="Batch size for training.")
    arg("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    
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
    
    # Load the saved model if resume_model is provided
    if args.resume_model is not None:
        if os.path.isfile(args.resume_model):
            print(f"Loading model from {args.resume_model}")
            model.load_state_dict(torch.load(args.resume_model))
        else:
            print(f"No model found at {args.resume_model}, starting training from scratch.")

    
    model, training_losses, validation_losses = train_model(model, train_loader,val_loader, criterion, optimizer, device, num_epochs=args.epochs, output_model_path=args.output_model, output_checkpoint_dir = args.output_checkpoint_dir)
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