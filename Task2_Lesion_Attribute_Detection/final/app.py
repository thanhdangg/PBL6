import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from model import UNet16
import numpy as np


# Load model
@st.cache_resource
def load_model(model_weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet16(num_classes=5, pretrained='vgg')  # Assuming UNet16 model class is defined
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()
    model.to(device)
    return model

# Process image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict and post-process
def predict_and_postprocess(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output, _, _ = model(image)  # Assuming model returns masks as the first output
        masks = output.squeeze(0).cpu().numpy()
        masks = (masks > 0.5).astype(np.uint8)  # Binarize
    return masks

# Main App
def main():
    st.title("Lesion Attribute Detection")
    st.write("Upload an image to detect attributes.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model
        st.write("Loading model...")
        model = load_model("model.pt")  # Path to your trained model weights

        # Preprocess image
        st.write("Processing image...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocessed_image = preprocess_image(image)

        # Predict masks
        st.write("Predicting attributes...")
        masks = predict_and_postprocess(model, preprocessed_image, device)
        attr_types = ['Pigment Network', 'Negative Network', 'Streaks', 'Milia-like Cyst', 'Globules']

        # Display results
        st.write("Prediction Results:")
        for i, attr_type in enumerate(attr_types):
            mask = masks[i]
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to image
            st.image(mask_image, caption=f"{attr_type} Mask", use_column_width=True)

if __name__ == "__main__":
    main()
