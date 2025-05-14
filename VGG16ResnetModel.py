#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import os

# Directory to store the model
MODEL_PATH = "models/DesnetDualHead.pth"
os.makedirs("models", exist_ok=True)

# Initialize the model
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 2)  # Adjust based on your model's output
)

# Step 1: Check if model already exists locally
if os.path.exists(MODEL_PATH):
    st.success("Model found locally. Loading the model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
else:
    # Step 2: Ask for upload if not found
    st.warning("Model not found locally. Please upload the model.")
    model_file = st.file_uploader("Upload DesnetDualHead.pth Model", type=["pth"])
    if model_file is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(model_file.read())
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        st.success("Model successfully uploaded and loaded.")

# Upload image for prediction
uploaded_image = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    model.eval()
    with torch.no_grad():
        age_logits, gender_logits = model(image)
        age_pred = torch.argmax(age_logits, dim=1).item()
        gender_pred = torch.argmax(gender_logits, dim=1).item()
    
    # Display the results
    st.write(f"**Predicted Age Group:** {age_pred}")
    st.write(f"**Predicted Gender:** {'Male' if gender_pred == 1 else 'Female'}")

# In[ ]:
