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

st.set_page_config(page_title='Age and Gender Prediction App', layout='centered')
st.title('Age and Gender Prediction App')

def load_data():
    # Replace these paths with actual paths to your training and validation data
    train_dir = '/path/to/train/data'
    validation_dir = '/path/to/validation/data'
    
    # Use ImageDataGenerator for data loading and augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'  # Assuming gender labels are 'Male' and 'Female' in directories
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

# Build the model
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    age_output = Dense(1, activation='linear', name='age')(x)  # For age regression
    gender_output = Dense(2, activation='softmax', name='gender')(x)  # For gender classification

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    model.compile(optimizer='adam', loss={'age': 'mse', 'gender': 'categorical_crossentropy'}, metrics={'age': 'mae', 'gender': 'accuracy'})
    return model

# Train the model
def train_model():
    st.write("Training the model... This will take time.")
    train_generator, validation_generator = load_data()
    
    model = build_model()
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    
    # Save the model after training
    model.save('age_gender_model.h5')
    st.write("Model trained and saved successfully!")

# Load pre-trained models if already available
try:
    model = tf.keras.models.load_model('age_gender_model.h5')
    st.write("Pre-trained model loaded successfully!")
except Exception as e:
    st.write("Model not found. Training a new model...")
    train_model()

# Upload image for prediction
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image for prediction
    img_array = np.array(image)
    img_resized = tf.image.resize(img_array, (224, 224))
    img_expanded = np.expand_dims(img_resized, axis=0)
    img_normalized = img_expanded / 255.0

    # Make predictions
    age_prediction, gender_prediction = model.predict(img_normalized)

    # Age prediction (Regressed value)
    predicted_age = int(age_prediction[0][0])
    st.write(f"Predicted Age: {predicted_age}")

    # Gender prediction (Categorical output)
    gender_labels = ['Male', 'Female']
    predicted_gender = gender_labels[np.argmax(gender_prediction)]
    st.write(f"Predicted Gender: {predicted_gender}")


# In[ ]:
