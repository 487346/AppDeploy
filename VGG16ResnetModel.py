#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
import torch
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

# Download the model from Dropbox
url = "https://www.dropbox.com/scl/fi/5m5f288mtry3nxac3e9ls/densenet121_dual_head.pth?rlkey=qjj51rrt76j63qg2ql5rehbnr&st=05r0cqns&dl=1"
model_path = "densenet121_dual_head.pth"

try:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        st.success('Model loaded successfully.')
    else:
        st.error(f"Failed to download model. HTTP Status Code: {response.status_code}")
except Exception as e:
    st.error(f"Error during model download: {e}")

model.eval()

# Age and Gender Mapping
age_bracket_map = {
    0: '0-9', 1: '10-19', 2: '20-29', 3: '30-39',
    4: '40-49', 5: '50-59', 6: '60-69', 7: '70-79', 8: '80+'
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
