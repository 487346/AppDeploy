#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

# Load the pre-trained DenseNet model
model = models.densenet121(pretrained=True)

# Modify the classifier for your specific task
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 10)  # Adjust the output layer (e.g., 10 classes: 8 age brackets + 2 genders)
)

# Load your trained weights if available
model.load_state_dict(torch.load('model.pth'))
model.eval()


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model (replace 'model.pth' with your model path)
model = torch.load('model.pth')
model.eval()

# Age bracket mapping (adjust according to your model's training data)
age_bracket_map = {
    '0-9': 0, '10-19': 1, '20-29': 2, '30-39': 3,
    '40-49': 4, '50-59': 5, '60-69': 6, '70-79': 7 , '80+': 8
}

reverse_age_bracket_map = {v: k for k, v in age_bracket_map.items()}
gender_map = {0: 'Male', 1: 'Female'}

# Streamlit App
st.title('Age and Gender Prediction App')
st.write('Upload an image to predict age and gender.')

# Image Upload
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        age_logits, gender_logits = model(image)  # Make sure your forward pass is returning two outputs
        age_pred = torch.argmax(age_logits, 1).item()
        gender_pred = torch.argmax(gender_logits, 1).item()
    
    # Display the predictions
    st.write(f'**Predicted Age Bracket:** {reverse_age_bracket_map[age_pred]}')
    st.write(f'**Predicted Gender:** {gender_map[gender_pred]}')



# In[ ]:
