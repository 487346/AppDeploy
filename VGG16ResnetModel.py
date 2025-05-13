#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Function to predict gender using DeepFace
def predict_gender(image):
    # DeepFace can predict gender
    analysis = DeepFace.analyze(image, actions=['gender'], enforce_detection=False)
    gender = analysis[0]['dominant_gender']
    return gender

# Function to predict age using DeepFace
def predict_age(image):
    # DeepFace can predict age
    analysis = DeepFace.analyze(image, actions=['age'], enforce_detection=False)
    age = analysis[0]['age']
    return age

# Streamlit UI setup
st.title("Age and Gender Prediction App")

# Instruction for the user
st.write("Please upload an image to predict the age and gender.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Show the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict gender
    st.write("Analyzing the image for gender...")

    # Call the gender prediction function
    gender = predict_gender(image)

    # Display the gender result
    st.write(f"Predicted Gender: {gender}")

    # Predict age
    st.write("Analyzing the image for age...")

    # Call the age prediction function
    age = predict_age(image)

    # Display the age result
    st.write(f"Predicted Age: {age} years")

# In[ ]:
