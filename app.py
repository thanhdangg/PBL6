import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import get_custom_objects
from keras import backend as K
from keras.layers import Layer, Multiply, Concatenate, Activation
import keras.layers as kl
from myCustomLayer.SoftAttention import SoftAttention

classes = {
    4: ('nv', 'melanocytic nevi'), 
    6: ('mel', 'melanoma'), 
    2: ('bkl', 'benign keratosis-like lesions'), 
    1: ('bcc', 'basal cell carcinoma'), 
    5: ('vasc', 'pyogenic granulomas and hemorrhage'), 
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  
    3: ('df', 'dermatofibroma')
}

# Register the custom laye
get_custom_objects().update({'SoftAttention': SoftAttention})

# Build the model
inputs = Input(shape=(28, 28, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)

# Use SoftAttention without reshaping
x, attention_maps = SoftAttention(ch=7, m=16)(x)  # Adjust channels to match Conv2D output
x = Flatten()(x)  # Flatten for the Dense layer
outputs = Dense(len(classes), activation='softmax')(x)

model = Model(inputs, outputs)

# Load your pre-trained model
model.load_weights('models/best_model.h5')

# Class mapping

# Prediction function
def predict_image(image):
    img = cv2.resize(image, (28, 28))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]  
    confidence = np.max(pred) 
    return classes[class_idx], confidence

# Streamlit UI
st.title("Skin Cancer Classification")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Prediction
    label, confidence = predict_image(image_np)

    # Display the results
    st.write(f"Prediction: **{label[1]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
