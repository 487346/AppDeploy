#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import gender_guesser.detector as gender
from transformers import pipeline

# Create a gender detector instance
detector = gender.Detector()

# Function to predict gender using gender-guesser
def predict_gender(name):
    gender_result = detector.get_gender(name)
    if gender_result == 'male' or gender_result == 'mostly_male':
        return 'Male'
    elif gender_result == 'female' or gender_result == 'mostly_female':
        return 'Female'
    else:
        return 'Unknown'

# Function to predict age bracket based on name using a simple approach
def predict_age_bracket(age):
    age_brackets = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    if age < 10:
        return age_brackets[0]
    elif age < 20:
        return age_brackets[1]
    elif age < 30:
        return age_brackets[2]
    elif age < 40:
        return age_brackets[3]
    elif age < 50:
        return age_brackets[4]
    elif age < 60:
        return age_brackets[5]
    elif age < 70:
        return age_brackets[6]
    elif age < 80:
        return age_brackets[7]
    else:
        return age_brackets[8]

# Create a Streamlit app
# Load pre-trained MobileNetV2 model or any other model for age & gender prediction
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define a function to preprocess the image for MobileNetV2
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

# Function to predict age and gender
def predict_age_and_gender(image):
    # DeepFace can predict both age and gender
    analysis = DeepFace.analyze(image, actions=['age', 'gender'], enforce_detection=False)
    age = analysis[0]['age']
    gender = analysis[0]['dominant_gender']
    return age, gender

# Streamlit UI setup
st.title("Age and Gender Prediction App")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Show the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict age and gender
    st.write("Analyzing the image...")

    # Call the prediction function
    age, gender = predict_age_and_gender(image)

    # Display the results
    st.write(f"Predicted Age: {age} years")
    st.write(f"Predicted Gender: {gender}")

# In[ ]:
