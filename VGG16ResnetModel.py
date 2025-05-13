#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

st.set_page_config(page_title='Age and Gender Predictor', layout='centered')
st.title('Age and Gender Prediction App')

# Load pre-trained models
try:
    age_model = tf.keras.models.load_model('age_model.h5')
    gender_model = tf.keras.models.load_model('gender_model.h5')
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    age_model = None
    gender_model = None

# Gender labels
gender_labels = ['Male', 'Female']

# Define age brackets (labels)
age_brackets = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file and age_model and gender_model:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (224, 224))
        img_expanded = np.expand_dims(img_resized, axis=0)
        img_normalized = img_expanded / 255.0

        # Predictions
        age_prediction = age_model.predict(img_normalized)
        gender_prediction = gender_model.predict(img_normalized)

        # Determine age bracket
        predicted_age = int(age_prediction[0][0])
        age_bracket = next((bracket for idx, bracket in enumerate(age_brackets) if idx * 10 <= predicted_age < (idx + 1) * 10), '80+')

        # Display results
        st.write(f'**Predicted Age Bracket:** {age_bracket}')
        st.write(f'**Predicted Gender:** {gender_labels[np.argmax(gender_prediction)]}')
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# In[ ]:
