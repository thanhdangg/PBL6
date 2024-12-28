from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def jaccard_loss(y_true, y_pred, class_weights, smooth=1e-6):
    # Convert class_weights to tensor
    class_weights_tensor = tf.convert_to_tensor(list(class_weights.values()), dtype=tf.float32)

    # Flatten y_true and y_pred using tf.reshape
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))

    # Calculate the intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    # Calculate the Jaccard index and apply the class weights
    jaccard_index = (intersection + smooth) / (union + smooth)
    weighted_jaccard = jaccard_index * class_weights_tensor

    # Return the loss (1 - Jaccard Index)
    return 1 - tf.reduce_mean(weighted_jaccard)



# Dice Loss với trọng số lớp
@register_keras_serializable()
def dice_loss(y_true, y_pred, class_weights, smooth=1e-10):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    # Tính toán Dice Coefficient
    intersection = K.sum(y_true * y_pred)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    
    # Tính toán trọng số lớp cho từng pixel
    weights = K.gather(class_weights, K.cast(y_true, dtype=tf.int32))
    
    # Áp dụng trọng số vào loss
    loss = (1 - dice_coeff) * weights
    return K.mean(loss)

# Hybrid Loss với Jaccard và BCE
@register_keras_serializable()
class HybridLossJaccard(tf.keras.losses.Loss):
    def __init__(self, weight_bce=1.0, weight_jaccard=1.0, class_weights=None, name="hybrid_loss_jaccard", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bce_loss = BinaryCrossentropy()
        self.weight_bce = weight_bce
        self.weight_jaccard = weight_jaccard
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Tính toán Binary Cross-Entropy Loss
        bce = self.bce_loss(y_true, y_pred)
        
        # Tính toán Jaccard Loss với trọng số lớp
        jaccard = jaccard_loss(y_true, y_pred, self.class_weights)
        
        # Kết hợp các loss
        return self.weight_bce * bce + self.weight_jaccard * jaccard

# Hybrid Loss với Dice và BCE
@register_keras_serializable()
class HybridLossDice(tf.keras.losses.Loss):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, class_weights=None, name="hybrid_loss_dice", **kwargs):
        super().__init__(name=name,**kwargs)
        self.bce_loss = BinaryCrossentropy()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Tính toán Binary Cross-Entropy Loss
        bce = self.bce_loss(y_true, y_pred)
        
        # Tính toán Dice Loss với trọng số lớp
        dice = dice_loss(y_true, y_pred, self.class_weights)
        
        # Kết hợp các loss
        return self.weight_bce * bce + self.weight_dice * dice

from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-10):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)
    
@register_keras_serializable()
def dice_metric(y_true, y_pred, smooth=1e-10):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)
def precision_metric(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = tf.keras.backend.sum(y_true * y_pred)
    fp = tf.keras.backend.sum((1 - y_true) * y_pred)
    return tp / (tp + fp + tf.keras.backend.epsilon())

def recall_metric(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = tf.keras.backend.sum(y_true * y_pred)
    fn = tf.keras.backend.sum(y_true * (1 - y_pred))
    return tp / (tp + fn + tf.keras.backend.epsilon())

def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())



custom_objects = {
    "HybridLossJaccard": HybridLossJaccard,
    "HybridLossDice": HybridLossDice,
    "iou_metric": iou_metric,
    "dice_metric": dice_metric
    
}
# Load the model from file
model_load = load_model('/kaggle/working/model_detection.keras', custom_objects =custom_objects)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size):
    # Load ảnh
    img = load_img(image_path, target_size=target_size)
    # Chuyển ảnh sang array
    img_array = img_to_array(img)
    # Chuẩn hóa pixel về [0, 1]
    img_array = img_array / 255.0
    # Thêm chiều batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Đường dẫn ảnh
image_path = '/kaggle/input/isic-2018-task-2-lesion-attribute-detection/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input/ISIC_0012169.jpg'

# Kích thước ảnh đầu vào của mô hình (ví dụ: 256x256)
target_size = (256, 256)

# Xử lý ảnh
input_image = preprocess_image(image_path, target_size)

# Bước 2: Dự đoán
predictions = model_load.predict(input_image)

# Bước 3: Tách 5 kênh đầu ra
# Kích thước predictions: (1, height, width, 5)
predicted_masks = predictions[0]  # Bỏ chiều batch, còn (height, width, 5)

# Các thuộc tính
attributes = [
    "Pigment Network",
    "Negative Network",
    "Streaks",
    "Milia-like Cyst",
    "Globules",
]

# Bước 4: Hiển thị từng ảnh thuộc tính
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(1, 5, i + 1)  # Tạo 5 subplots
    plt.imshow(predicted_masks[:, :, i], cmap='gray')
    plt.title(attributes[i])
    plt.axis('off')

plt.tight_layout()
plt.show()