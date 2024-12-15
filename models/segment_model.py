'''
author: Trong Hoang Cong, Cuong Do
date: 2024-09-12
'''
from os import environ

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import cloudinary
import cloudinary.uploader
import requests
from dotenv import load_dotenv

# Configuration       
cloudinary.config( 
    cloud_name = environ.get("CLOUDINARY_CLOUD_NAME","dsnqfhnyx"),
    api_key = environ.get("CLOUDINARY_API_KEY","917939399672721"),
    api_secret = environ.get("CLOUDINARY_API_SECRET","RW67llhv4tYluEGj58liY5fpRYg"),
    secure=environ.get("CLOUDINARY_SECURE",True)
)

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    '''
    Dice coefficient for binary image segmentation

    Args:
    y_true: tf.Tensor, true labels
    y_pred: tf.Tensor, predicted labels

    Returns:
    dice coefficient
    '''
    smooth = 1e-7
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def jacard(y_true, y_pred):
    '''
    Jacard coefficient for binary image segmentation
    :param y_true: tf.Tensor, true labels
    :param y_pred: tf.Tensor, predicted labels
    :return: Jacard coefficient
    '''
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union

def bce_dice_loss(y_true, y_pred):
    '''
    Loss function combining binary cross-entropy and dice loss
    :param y_true: tf.Tensor, true labels
    :param y_pred: tf.Tensor, predicted labels
    :return: bce_dice_loss
    '''
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    dice = dice_coef(y_true, y_pred)
    return bce - K.log(dice)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss_log(y_true, y_pred):
    '''
    Loss function combining binary cross-entropy and dice loss
    :param y_true: tf.Tensor, true labels
    :param y_pred: tf.Tensor, predicted labels
    :return: bce_dice_loss
    '''
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    dice = dice_coef(y_true, y_pred)
    return bce + 1 - dice

def getting_segmet_model():
    '''
    Load the model
    :return: model loaded from the saved file
    '''
    get_custom_objects().update({"dice_coef": dice_coef})
    get_custom_objects().update({"bce_dice_loss_log": bce_dice_loss_log})
    model = tf.keras.models.load_model("/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/models/saved_model_v3.keras")
    return model

def parse_image_from_url(img_url, size=(256, 256)):
    '''
    Parse image from URL
    :param img_url:
    :param size:
    :return: image tensor
    '''
    response = requests.get(img_url)
    img = tf.image.decode_jpeg(response.content, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size)
    return img

def predict(img_url, file_name,model_segment):
    '''
    Predict the mask of the image
    :param img_url: image URL
    :param file_name: file name masked
    :param model_segment: model for segmenting
    :return: URL of the masked image
    '''
    img = parse_image_from_url(img_url)
    predicted_mask = model_segment.predict(img[np.newaxis, ...], verbose=0)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    print("Predict Mask shape: ", predicted_mask.shape)
    predicted_mask = predicted_mask[..., 0]
    plt.imsave("/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/static/results/{}".format(file_name), predicted_mask, cmap='gray')
    upload_result = cloudinary.uploader.upload("/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/static/results/{}".format(file_name))
    return upload_result['url']

