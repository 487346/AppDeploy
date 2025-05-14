#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
import torch
import streamlit as st
from torchvision import models, transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained DenseNet model
model = models.densenet121(pretrained=True)

# Modify the classifier for your specific task
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 10)  # 8 age brackets + 2 genders
)

# Load your trained weights if available
try:
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    st.success('Model loaded successfully.')
except FileNotFoundError:
    st.error('model.pth not found. Please upload the model to the correct path.')

model.eval()

# Age and Gender Mapping
age_bracket_map = {
    0: '0-2', 1: '3-9', 2: '10-19', 3: '20-29',
    4: '30-39', 5: '40-49', 6: '50-59', 7: '60+'
}
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
        outputs = model(image)
        
        # Split the outputs for age and gender
        age_logits, gender_logits = outputs[:, :8], outputs[:, 8:]
        
        age_pred = torch.argmax(age_logits, 1).item()
        gender_pred = torch.argmax(gender_logits, 1).item()
    
    # Display the predictions
    st.write(f'**Predicted Age Bracket:** {age_bracket_map[age_pred]}')
    st.write(f'**Predicted Gender:** {gender_map[gender_pred]}')
# In[ ]:
