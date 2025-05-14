#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import os

# Initialize the model
model = models.densenet121(pretrained=False)

# Add custom layers if needed (e.g., dual head for age and gender)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 2)  # Adjust based on your model's output
)

# Upload model file
model_file = st.file_uploader("Upload DesnetDualHead.pth Model", type=["pth"])
if model_file is not None:
    with open("DesnetDualHead.pth", "wb") as f:
        f.write(model_file.read())
    st.success("Model successfully uploaded and saved locally.")

    # Load the model
    try:
        state_dict = torch.load("DesnetDualHead.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")

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
