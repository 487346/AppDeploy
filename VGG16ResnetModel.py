#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# Define the modified ResNet-50 model with two heads (age and gender)
class ResNet50DualHead(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2): 
        super(ResNet50DualHead, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.fc = nn.Identity()  # Remove existing FC layer

        # Age classifier
        self.age_head = nn.Linear(2048, num_age_classes)

        # Gender classifier
        self.gender_head = nn.Linear(2048, num_gender_classes)

    def forward(self, x):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)

        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)

        return age_logits, gender_logits

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the model and load it onto the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = ResNet50DualHead(num_age_classes=9, num_gender_classes=2).to(device)

# Set model to evaluation mode
resnet_model.eval()

# Labels
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
gender_labels = ['Male', 'Female']

# Streamlit app
st.title('Gender and Age Prediction App')
st.write('Upload an image, and the ResNet-50 model will predict gender and age bracket.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        age_logits, gender_logits = resnet_model(image_tensor)
        age_probs = torch.nn.functional.softmax(age_logits, dim=1)
        gender_probs = torch.nn.functional.softmax(gender_logits, dim=1)

        age_prediction = torch.argmax(age_probs, dim=1).item()
        gender_prediction = torch.argmax(gender_probs, dim=1).item()

    # Display results
    st.subheader('Predictions:')
    st.write(f'Gender: {gender_labels[gender_prediction]} (Confidence: {gender_probs[0][gender_prediction] * 100:.2f}%)')
    st.write(f'Age Bracket: {age_labels[age_prediction]} (Confidence: {age_probs[0][age_prediction] * 100:.2f}%)')



# In[ ]:
