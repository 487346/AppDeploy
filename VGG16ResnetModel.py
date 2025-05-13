#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import streamlit as st
from PIL import Image
from deepface import DeepFace

# Streamlit page configuration
st.set_page_config(page_title='Age and Gender Predictor', layout='centered')
st.title('Age and Gender Prediction App')

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Analyze the image using DeepFace
        st.write("Analyzing the image... This may take a few seconds.")
        
        # DeepFace analyze to predict age, gender, and other attributes
        result = DeepFace.analyze(img_path=uploaded_file, actions=['age', 'gender'], enforce_detection=False)
        
        # Extract age and gender from result
        age = result[0]['age']
        gender = result[0]['gender']
        
        # Display the predictions
        st.write(f'**Predicted Age:** {age}')
        st.write(f'**Predicted Gender:** {gender}')
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")


# In[ ]:
