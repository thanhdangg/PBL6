import argparse
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


def normalize_image(image):
    return image / 255.0

def resize_image(image, size):
    return cv2.resize(image, size)

def augment_image(image):
    flipped_image = cv2.flip(image, 1)
    rotated_image = cv2.rotate(flipped_image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def process_image(input_path, size):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")
    
    image = normalize_image(image)
    image = resize_image(image, size)
    image = augment_image(image)    
    return image
    
def process_images_in_folder(input_folder, size):   
    images = []
    filenames = [] 
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        try:
            image = process_image(input_path, size)
            images.append(image)
            filenames.append(filename)
        except ValueError as e:
            print(e)
    
    images = np.array(images)
    return images, filenames


def main():
    parser = argparse.ArgumentParser(description="Process and augment images.")
    arg = parser.add_argument

    arg("--input_folder", type=str,default ="./data/ISIC2018_Task1-2_Training_Input", help="Path to the input folder containing images.")
    arg("--size", type=int, nargs=2, default=(256, 256), help="Size to resize the images to (width height).")
    
    args = parser.parse_args()
    
    images, filenames = process_images_in_folder(args.input_folder, tuple(args.size))
    print(f"Processed {len(images)} images.")

if __name__ == "__main__":
    main()
