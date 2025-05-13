#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
from deepface import DeepFace
import io

# Streamlit page configuration
st.set_page_config(page_title='Age and Gender Predictor', layout='centered')
st.title('Age and Gender Prediction App')

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Analyze the image using DeepFace
        st.write("Analyzing the image... This may take a few seconds.")
        
        # Convert the uploaded image to a byte stream and then to a file-like object
        img_path = io.BytesIO(uploaded_file.read())
        
        # DeepFace analyze to predict age, gender, and other attributes
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender'], enforce_detection=False)
        
        # Extract age and gender from result
        age = result[0]['age']
        gender = result[0]['gender']
        
        # Display the predictions
        st.write(f'**Predicted Age:** {age}')
        st.write(f'**Predicted Gender:** {gender}')
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# In[ ]:
