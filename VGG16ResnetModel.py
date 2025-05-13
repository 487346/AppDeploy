#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# Load your pre-trained model here (e.g., gender and age prediction model)
model = torch.load('model.pth')
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    """Predict gender and age from an image."""
    image = Image.open(image)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        gender = 'Male' if outputs[0].item() > 0.5 else 'Female'
        age_range = '20-30'  # Placeholder: Replace with actual logic

    return gender, age_range

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    gender, age_range = predict_image(image)

    return jsonify({
        'gender': gender,
        'age_range': age_range
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)

# In[ ]:
