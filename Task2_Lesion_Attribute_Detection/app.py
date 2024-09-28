import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
from model import UNet  # Make sure to import your UNet model definition
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the model
model_path = "models/multi_task_unet.h5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define the Streamlit app
st.title("Lesion Attribute Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Run the model
    with torch.no_grad():
        outputs = model(input_image)

    # Process the outputs
    attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
    output_images = outputs.squeeze().cpu().numpy()  # Shape: (5, 256, 256)

    # Apply threshold to create binary masks
    binary_images = (output_images > 0.5).astype(np.uint8)  # Threshold at 0.5

    # Display the results
    st.write("Results:")
    fig, axes = plt.subplots(1, 6, figsize=(20, 10))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    for i in range(5):
        axes[i+1].imshow(binary_images[i], cmap='gray')
        axes[i+1].set_title(attributes[i])
    st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Streamlit app is running...")