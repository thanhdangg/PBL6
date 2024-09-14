import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from image_process import process_images_in_folder
from model import MultiTaskUNet 

def load_data(input_folder, image_size):
    # Load processed images
    images, masks, _ = process_images_in_folder(input_folder, image_size)
    return images, masks

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

    # Load data
    images, labels = load_data(args.input_folder, tuple(args.size))

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Initialize the model
    model = MultiTaskUNet(input_shape=(args.size[0], args.size[1], 3))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy, precision, recall, AUC"])

    # Set up checkpointing
    checkpoint = ModelCheckpoint(args.output_model, monitor='val_loss', save_best_only=True, mode='min')
    
    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    main()