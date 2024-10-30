import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNet

# Define attributes for lesion detection
attributes = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]

# Define image transformations (resize, to tensor, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Function to load and preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels (RGB)
    image = transform(image)  # Resize and convert to tensor
    return image.unsqueeze(0)  # Add batch dimension

# Function to post-process and return mask image
def postprocess_mask(pred_mask):
    pred_mask_np = pred_mask.cpu().detach().numpy()
    binary_mask = (pred_mask_np > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    mask_image = Image.fromarray(binary_mask.squeeze(), mode="L")
    return mask_image

# Function to make predictions
def predict_attributes(model, image_tensor, device):
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Return each predicted mask for the attributes
        return outputs

# Streamlit interface
st.title("Lesion Attribute Detection")
st.write("Upload an image to detect skin lesion attributes.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = UNet()  # Initialize the UNet model
    model_path = "./models/multi_task_unet.keras"  # Path to your trained model
    model.load_state_dict(torch.load(model_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess the uploaded image
    image_tensor = preprocess_image(image)

    # Run the model and get predicted masks for the attributes
    st.write("Predicting lesion attributes...")
    predicted_masks = predict_attributes(model, image_tensor, device)

    # Create 5 columns to display the images in a row
    cols = st.columns(5)

    # Display each mask in a separate column
    for i, attribute in enumerate(attributes):
        with cols[i]:
            mask_image = postprocess_mask(predicted_masks[0, i])
            st.image(mask_image, caption=attribute, use_column_width=True)
