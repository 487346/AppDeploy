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

# Gender labels
gender_labels = ['Male', 'Female']

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (224, 224))
        img_expanded = np.expand_dims(img_resized, axis=0)
        img_normalized = img_expanded / 255.0

        # Predictions
        age_prediction = age_model.predict(img_normalized)
        gender_prediction = gender_model.predict(img_normalized)

        # Display results
        age = int(age_prediction[0][0])
        gender = gender_labels[np.argmax(gender_prediction)]

        st.write(f'**Predicted Age:** {age} years')
        st.write(f'**Predicted Gender:** {gender}')
    except Exception as e:
        st.error(f"Error processing the image: {e}")
# In[ ]:
