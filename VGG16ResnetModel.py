#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import streamlit as st
from time import sleep

# Age brackets
age_brackets = ['0-1', '2-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ResNet-50 Model
class ResNet50DualHead(nn.Module):
    def __init__(self, num_age_classes=10, num_gender_classes=2):
        super(ResNet50DualHead, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()
        self.age_head = nn.Linear(2048, num_age_classes)
        self.gender_head = nn.Linear(2048, num_gender_classes)

    def forward(self, x):
        features = self.base_model(x)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age_logits, gender_logits

# VGG-16 Model
class VGG16DualHead(nn.Module):
    def __init__(self, num_age_classes=10, num_gender_classes=2):
        super(VGG16DualHead, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.age_head = nn.Linear(25088, num_age_classes)
        self.gender_head = nn.Linear(25088, num_gender_classes)

    def forward(self, x):
        features = self.base_model.features(x)
        features = features.view(features.size(0), -1)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age_logits, gender_logits

# Initialize models
resnet_model = ResNet50DualHead().to(device)
vgg_model = VGG16DualHead().to(device)

# Streamlit App
st.title('Age and Gender Prediction App')
st.write('Upload an image to detect age group and gender using two models: ResNet-50 and VGG-16.')

uploaded_file = st.file_uploader('Upload an Image...', type=['jpg', 'png', 'jpeg'])

def predict(image, model):
    image_tensor = transform(image).unsqueeze(0).to(device)
    age_logits, gender_logits = model(image_tensor)
    age_confidences = torch.softmax(age_logits, dim=1)
    gender_confidences = torch.softmax(gender_logits, dim=1)
    age_index = torch.argmax(age_confidences, dim=1).item()
    gender_index = torch.argmax(gender_confidences, dim=1).item()
    return age_brackets[age_index], age_confidences[0][age_index].item(), 'Male' if gender_index == 0 else 'Female', gender_confidences[0][gender_index].item()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner('Processing the image...'):
            sleep(2)
            resnet_age, resnet_age_conf, resnet_gender, resnet_gender_conf = predict(image, resnet_model)
            st.write('### ResNet-50 Predictions:')
            st.write(f'**Age Group:** {resnet_age} (Confidence: {resnet_age_conf:.2f})')
            st.write(f'**Gender:** {resnet_gender} (Confidence: {resnet_gender_conf:.2f})')
            st.write('---')
            sleep(1)
            vgg_age, vgg_age_conf, vgg_gender, vgg_gender_conf = predict(image, vgg_model)
            st.write('### VGG-16 Predictions:')
            st.write(f'**Age Group:** {vgg_age} (Confidence: {vgg_age_conf:.2f})')
            st.write(f'**Gender:** {vgg_gender} (Confidence: {vgg_gender_conf:.2f})')

        st.success('Prediction Complete!')

    except UnidentifiedImageError:
        st.error('The uploaded file is not a valid image. Please upload a JPG or PNG image.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')


# In[ ]:
