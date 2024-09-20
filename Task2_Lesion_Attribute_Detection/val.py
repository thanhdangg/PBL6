import argparse
import torch
from torch.utils.data import DataLoader
from datasets import SkinLesionDataset
from loss import HybridLoss, HybridLossDice
from model import UNet

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
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Validate a U-Net model for lesion attribute detection.")
    arg = parser.add_argument

    arg("--input_folder", type=str, default="./data/ISIC2018_Task1-2_Validation_Input", help="Path to the folder containing input images.")
    arg("--ground_truth_dir", type=str, default="./data/ISIC2018_Task1-2_Validation_GroundTruth", help="Path to the ground truth masks directory.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg("--batch_size", type=int, default=16, help="Batch size for validation.")
    arg("--model_path", type=str, default="./models/multi_task_unet.h5", help="Path to the trained model.")
    
    args = parser.parse_args()
    
    dataset = SkinLesionDataset(image_dir=args.input_folder, ground_truth_dir=args.ground_truth_dir, target_size=args.size)
    print('Loading dataset...', dataset)
    print("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print("Loading dataloader...", dataloader)
    
    model = UNet()
    model.load_state_dict(torch.load(args.model_path))
    print("Model loaded from", args.model_path)
    
    criterion = HybridLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    avg_loss = evaluate_model(model, dataloader, criterion, device)
    print(f"Average Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()