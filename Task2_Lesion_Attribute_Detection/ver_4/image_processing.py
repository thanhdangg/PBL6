import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, ground_truth_dir, output_mask_dir, target_size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_mask_dir = output_mask_dir
        self.target_size = target_size

        # Define mean and std for normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def process_image(self, image_path, is_mask=False):
        """
        Process an individual image or mask: load, resize, normalize (for images), and return as a numpy array.
        """
        # Load image or mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Process image
        if not is_mask:
            # Normalize image (scale to [0, 1] and apply mean/std normalization)
            image = image / 255.0
            image = (image - self.mean) / self.std

        return image

    def save_image(self, image, output_path):
        """
        Save the processed image or mask as a PNG image.
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8) if image.ndim == 2 else (image * 255).astype(np.uint8))

        # Save image
        image.save(output_path)

    def is_blank_image(self, mask):
        """
        Check if the mask is blank (all pixels are 0).
        """
        return np.all(mask == 0)

    def process_and_save(self, image_filename):
        """
        Process an image file and its corresponding masks, save the processed image, and mark blank masks.
        """
        # Extract the image ID (e.g., ISIC_<image_id>) from the filename
        image_id = image_filename.split('.')[0]  # Assuming filenames are like 'ISIC_<image_id>.jpg'
        input_path = os.path.join(self.input_dir, image_filename)
        output_path = os.path.join(self.output_dir, image_filename)

        # Process the input image
        image = self.process_image(input_path)
        self.save_image(image, output_path)

        # Process corresponding mask images for each attribute
        attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
        for attribute in attributes:
            mask_filename = f"{image_id}_attribute_{attribute}.png"
            mask_path = os.path.join(self.ground_truth_dir, mask_filename)
            if os.path.exists(mask_path):
                mask = self.process_image(mask_path, is_mask=True)
                if self.is_blank_image(mask):
                    # print(f"Skipping blank mask: {mask_filename}")
                    continue
                mask_output_path = os.path.join(self.output_mask_dir, mask_filename)
                self.save_image(mask, mask_output_path)
            else:
                print(f"Mask not found for {image_filename} with attribute {attribute}.")

    def process_all_images(self):
        """
        Process all images in the input directory and save them to the output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.output_mask_dir):
            os.makedirs(self.output_mask_dir)

        for image_filename in os.listdir(self.input_dir):
            if image_filename.endswith('.jpg'):  # Process only JPEG images
                print(f"Processing {image_filename}")
                self.process_and_save(image_filename)

parser = argparse.ArgumentParser(description="Process and augment images.")
arg = parser.add_argument

arg("--input_dir", type=str,default ="./data/ISIC2018_Task1-2_Training_Input", help="Path to the input folder containing images.")
arg("--output_dir", type=str,default ="./data/Processed_Train_Input", help="Path to the output folder to save processed images.")
arg("--ground_truth_dir", type=str, default="./data/ISIC2018_Task1-2_Training_GroundTruth", help="Path to the ground truth masks directory.")
arg("--output_mask_dir", type=str, default="./data/Processed_Train_GroundTruth", help="Path to the output folder to save processed masks.")

arg("--input_val_dir", type=str, default="./data/ISIC2018_Task1-2_Validation_Input", help="Path to the validation folder containing images.")
arg("--ground_truth_val_dir", type=str, default="./data/ISIC2018_Task1-2_Validation_GroundTruth", help="Path to the ground truth masks for validation directory")
arg("--output_val_dir", type=str, default="./data/Processed_Val_Input", help="Processed Val Input")
arg("--output_mask_val_dir", type=str, default="./data/Processed_Val_GroundTruth", help="Processed Val GroundTruth")

arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")

args = parser.parse_args()
        
print("Processing for train")
processor = ImagePreprocessor(args.input_dir, args.output_dir, args.ground_truth_dir, args.output_mask_dir, target_size=tuple(args.size))
processor.process_all_images()