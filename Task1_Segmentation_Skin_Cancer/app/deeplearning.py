import os
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import keras

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def jacard(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection/union

def bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    dice = dice_coef(y_true, y_pred)
    return bce - K.log(dice)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss_log(y_true, y_pred):
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    dice = dice_coef(y_true, y_pred)
    return bce + 1 - dice

get_custom_objects().update({"dice_coef": dice_coef})
get_custom_objects().update({"bce_dice_loss_log": bce_dice_loss_log})

model = tf.keras.models.load_model("model/saved_model_29_09_24.keras")

def parse_image(img_path, size = (256, 256)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size)
    return img

def predict(img_path, file_name):
    if img_path and file_name:
        print(file_name)
        img = parse_image(img_path)
        predicted_mask = model.predict(img[np.newaxis, ...], verbose=0)[0]
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        print("Predict Maskkkkkkkkkkkkkk: ",predicted_mask.shape)
        
        predicted_mask = predicted_mask[..., 0]

        plt.imsave("static/results/{}".format(file_name), predicted_mask, cmap='gray')
        return True
    return False








