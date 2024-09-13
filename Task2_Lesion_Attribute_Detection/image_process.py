import argparse
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def normalize_image(image):
    return image / 255.0

def resize_image(image, size):
    return cv2.resize(image, size)

def augment_image(image):
    flipped_image = cv2.flip(image, 1)
    rotated_image = cv2.rotate(flipped_image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def process_image(input_path, output_path, size):
    image = cv2.imread(input_path)
    
    image = normalize_image(image)
    
    image = resize_image(image, size)
    
    image = augment_image(image)
    
    cv2.imwrite(output_path, image * 255)

    
def process_images_in_folder(input_folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path, size)

def main():
    parser = argparse.ArgumentParser(description="Process and augment images.")
    arg = parser.add_argument

    arg("--input_folder", type=str,default ="./data/ISIC2018_Task1-2_Training_Input", help="Path to the input folder containing images.")
    arg("--output_folder", type=str,default="./data/image_processed", help="Path to save the processed images.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    
    args = parser.parse_args()
    
    process_images_in_folder(args.input_folder, args.output_folder, tuple(args.size)) 

if __name__ == "__main__":
    main()
