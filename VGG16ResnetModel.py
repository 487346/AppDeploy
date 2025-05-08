#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import torch
from transformers import ConvNextForImageClassification, ConvNextFeatureExtractor
from PIL import Image
import numpy as np

# Define age brackets
age_brackets = ['0-10', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# Load the pre-trained ConvNeXT model and feature extractor
model_name = 'facebook/convnext-large-22k-224'  # Replace with your custom model if needed
model = ConvNextForImageClassification.from_pretrained(model_name)
feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_name)

# Move the model to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit App Title and Description
st.title("Age and Gender Classification with ConvNeXT")
st.write("""
    This app allows you to upload an image, and then it classifies the gender (Male or Female) and age group (e.g., 0-10, 10-19, 20-29, etc.) of the person in the image.
    The model is based on ConvNeXT architecture, pre-trained on a large dataset for image classification.
    """)

# Streamlit File Uploader
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# If an image is uploaded, process it
if uploaded_image is not None:
    # Open the image
    img = Image.open(uploaded_image)
    
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    inputs = feature_extractor(images=img, return_tensors="pt").to(device)
    
    # Perform inference to predict gender and age
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Gender classification (Assuming first two logits are for gender: Male or Female)
        gender_class = torch.argmax(probs[:, :2], dim=-1).item()  # 0: Male, 1: Female
        
        # Age classification (Assuming the next logits are for age groups: 0-10, 10-19, etc.)
        age_class = torch.argmax(probs[:, 2:], dim=-1).item()  # Modify based on your model's age output
        
        # Gender and Age labels
        gender_labels = ["Male", "Female"]
        
        # Display predictions
        st.write(f"Predicted Gender: {gender_labels[gender_class]}")
        st.write(f"Predicted Age Group: {age_brackets[age_class]}")

# Footer
st.write("Made with 💻 by [Your Name]")

st.write(f'Age Bracket: {age_labels[age_prediction]} (Confidence: {age_probs[0][age_prediction] * 100:.2f}%)')



# In[ ]:
