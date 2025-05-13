#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Function to predict age and gender using DeepFace
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
