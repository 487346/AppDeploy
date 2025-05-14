#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import numpy as np

# Define the modified DenseNet model with two heads (age and gender)
class DenseNetDualHead(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2):
        super(DenseNetDualHead, self).__init__()
        self.base_model = models.densenet121(pretrained=True)
        # Replace the classifier with a custom head
        self.base_model.classifier = nn.Identity()
        # Age head (classifier)
        self.age_head = nn.Linear(1024, num_age_classes)
        # Gender head (classifier)
        self.gender_head = nn.Linear(1024, num_gender_classes)

    def forward(self, x):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)
        # Age and Gender predictions
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age_logits, gender_logits

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the model and load it onto the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNetDualHead(num_age_classes=9, num_gender_classes=2).to(device)
model.eval()

# Define labels
age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
gender_labels = ['Male', 'Female']

# Streamlit app setup
st.title('Gender and Age Prediction App')
st.write('Upload an image, and the model will predict the gender and age bracket.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Resize image for display (5 inches = 5 * 100 pixels)
    display_image = image.resize((400, 400))
    
    # Predict using the model
    with torch.no_grad():
        # Apply the necessary transformations and add a batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)
        # Forward pass through the model
        age_logits, gender_logits = model(image_tensor)
        age_probs = torch.nn.functional.softmax(age_logits, dim=1)
        gender_probs = torch.nn.functional.softmax(gender_logits, dim=1)
        age_prediction = torch.argmax(age_probs, dim=1).item()
        gender_prediction = torch.argmax(gender_probs, dim=1).item()

    #  Display image and predictions side by side with 10px padding
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="center-align">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image (5x5 inches)', use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="center-align">', unsafe_allow_html=True)
        st.subheader('Predictions:')
        st.write(f'**Gender:** {gender_labels[gender_prediction]}')
        st.write(f'**Age Bracket:** {age_labels[age_prediction]}')
        st.markdown('</div>', unsafe_allow_html=True)
# In[ ]:
