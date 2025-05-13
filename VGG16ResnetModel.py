#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import streamlit as st
from densenet_model import predict 


# Define the custom model for Age and Gender prediction
class DenseNetGenderAgeModel(nn.Module):
    def __init__(self):
        super(DenseNetGenderAgeModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)  # Load pre-trained DenseNet
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 2)  # Gender classifier
        self.age_classifier = nn.Linear(self.densenet.classifier.in_features, 9)  # Age classifier

    def forward(self, x):
        features = self.densenet(x)
        gender_output = self.densenet.classifier(features)
        age_output = self.age_classifier(features)
        return gender_output, age_output

# Instantiate the model and load pre-trained weights
model = DenseNetGenderAgeModel()

# Load the model weights (assuming you have fine-tuned weights for the model)
# model.load_state_dict(torch.load('your_model_weights.pth'))
model.eval()

# Preprocessing function for input images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gender and age mappings
age_brackets = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
gender_labels = ['Male', 'Female']

def predict(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        gender_output, age_output = model(image)

    # Get the predicted gender and age group
    _, gender_pred = torch.max(gender_output, 1)
    _, age_pred = torch.max(age_output, 1)

    predicted_gender = gender_labels[gender_pred.item()]
    predicted_age_bracket = age_brackets[age_pred.item()]

    return predicted_gender, predicted_age_bracket

# Streamlit App UI
st.title("Age and Gender Prediction App")
st.write("Upload an image to predict age and gender.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict age and gender
    gender, age = predict(uploaded_file)

    # Display results
    st.subheader(f"Predicted Gender: {gender}")
    st.subheader(f"Predicted Age Group: {age}")

# In[ ]:
