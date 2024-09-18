import argparse
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, target_size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size

        # Define transformations
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

    def process_image(self, image_path):
        """
        Process an individual image: load, resize, normalize, and return as tensor.
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        return image

    def augment_image(self, image):
        """
        Augment an individual image with random transformations.
        """
        augmented_image = self.augmentation(image)
        return augmented_image

    def save_image(self, image_tensor, output_path):
        """
        Save the processed image tensor back as a PNG image.
        """
        # Convert the tensor back to PIL Image
        image = transforms.ToPILImage()(image_tensor)

        # Save image
        image.save(output_path)

    def process_and_save(self, image_filename, augment=False):
        """
        Process an image file, save the processed image, and optionally save augmented versions.
        """
        input_path = os.path.join(self.input_dir, image_filename)
        output_path = os.path.join(self.output_dir, image_filename)

        # Process image
        image = self.process_image(input_path)

        # Save the processed image
        self.save_image(image, output_path)

        # Optional: Save augmented images
        if augment:
            for i in range(3):  # Save 3 augmented versions
                augmented_image = self.augment_image(image)
                augmented_output_path = os.path.join(self.output_dir, f"aug_{i}_{image_filename}")
                self.save_image(augmented_image, augmented_output_path)

    def process_all_images(self, augment=False):
        """
        Process all images in the input directory and save them to the output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for image_filename in os.listdir(self.input_dir):
            if image_filename.endswith('.jpg'):  # Process only JPEG images
                print(f"Processing {image_filename}")
                self.process_and_save(image_filename, augment)




def main():
    parser = argparse.ArgumentParser(description="Process and augment images.")
    arg = parser.add_argument

    arg("--input_dir", type=str,default ="./data/ISIC2018_Task1-2_Training_Input", help="Path to the input folder containing images.")
    # arg("--input_dir", type=str,default ="/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input", help="Path to the input folder containing images.")
    arg("--output_dir", type=str,default ="./data/Input_Processed", help="Path to the output folder to save processed images.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    arg('--augment', action='store_true', help='If set, apply data augmentation')

    args = parser.parse_args()
    
    processor = ImagePreprocessor(args.input_dir, args.output_dir, target_size=tuple(args.size))
    
    # Process all images with optional augmentation
    processor.process_all_images(augment=args.augment)

if __name__ == "__main__":
    main()
