"""
author: Cuong Do
date: 2024-09-12
"""

from tensorflow.keras.models import load_model
import cv2


def getting_classify_model():
    """
    Load the pre-trained model

    Returns:
    model: keras.Model, pre-trained model
    """
    classify_model = load_model(
        "/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/models/skin-cancer-mnist-ham10000.keras"
    )
    return classify_model


# Define the classes
classes = {
    4: ("nv", "melanocytic nevi"),
    6: ("mel", "melanoma"),
    2: ("bkl", "benign keratosis-like lesions"),
    1: ("bcc", "basal cell carcinoma"),
    5: ("vasc", "pyogenic granulomas and hemorrhage"),
    0: ("akiec", "Actinic keratoses and intraepithelial carcinomae"),
    3: ("df", "dermatofibroma"),
}


def predict_image(image, model):
    """
    Predict the class of the given image

    Args:
    image: np.array, input image
    model: keras.Model, trained model

    Returns:
    class_name: str, name of the predicted class
    result: np.array, prediction probabilities
    """
    img_resized = cv2.resize(image, (28, 28))
    result = model.predict(img_resized.reshape(1, 28, 28, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind][1]
    return class_name, result
