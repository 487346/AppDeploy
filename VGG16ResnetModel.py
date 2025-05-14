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

# Define the modified ResNet-50 model with two heads (age and gender)
# Define the modified ResNet-50 model with two heads (age and gender)
class ResNet50DualHead(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2):
        super(ResNet50DualHead, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the fully connected layer for both age and gender classification
        self.base_model.fc = nn.Identity()  # Remove the existing FC layer

        # Age head (classifier)
        self.age_head = nn.Linear(2048, num_age_classes)
        
        # Gender head (classifier)
        self.gender_head = nn.Linear(2048, num_gender_classes)

    def forward(self, x):
        features = self.base_model(x)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten the output
        
        # Age and Gender predictions
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        
        return age_logits, gender_logits

# Define the modified VGG-16 model with two heads (age and gender)
# Define the modified VGG-16 model with two heads (age and gender)
class VGG16DualHead(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2):
        super(VGG16DualHead, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier with a custom head
        self.base_model.classifier = nn.Identity()  # Remove the existing classifier
        
        # Age head (classifier)
        self.age_head = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_age_classes)
        )
        
        # Gender head (classifier)
        self.gender_head = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_gender_classes)
        )

    def forward(self, x):
        features = self.base_model.features(x)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten the output
        
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

# Instantiate the models and load them onto the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = ResNet50DualHead(num_age_classes=9, num_gender_classes=2).to(device)
vgg_model = VGG16DualHead(num_age_classes=9, num_gender_classes=2).to(device)

# Set both models to evaluation mode
resnet_model.eval()
vgg_model.eval()

# Define labels
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
gender_labels = ['Male', 'Female']

# Streamlit app setup
st.title('Gender and Age Prediction App')
st.write('Upload an image, and the models will predict the gender and age bracket.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict using both models
    with torch.no_grad():
        age_logits_r, gender_logits_r = resnet_model(image_tensor)
        age_logits_v, gender_logits_v = vgg_model(image_tensor)
        
        # Average the predictions for better accuracy
        age_probs = (torch.nn.functional.softmax(age_logits_r, dim=1) + torch.nn.functional.softmax(age_logits_v, dim=1)) / 2
        gender_probs = (torch.nn.functional.softmax(gender_logits_r, dim=1) + torch.nn.functional.softmax(gender_logits_v, dim=1)) / 2
        
        age_prediction = torch.argmax(age_probs, dim=1).item()
        gender_prediction = torch.argmax(gender_probs, dim=1).item()
    
    # Display results with confidence
    st.subheader('Predictions:')
    st.write(f'**Gender:** {gender_labels[gender_prediction]} (Confidence: {gender_probs[0][gender_prediction] * 100:.2f}%)')
    st.write(f'**Age Bracket:** {age_labels[age_prediction]} (Confidence: {age_probs[0][age_prediction] * 100:.2f}%)')

# In[ ]:
