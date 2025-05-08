import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import streamlit as st
from time import sleep

# 🚀 Improved Age Brackets
age_brackets = [
    '0-2',   # Infant
    '3-5',   # Toddler
    '6-12',  # Child
    '13-19', # Teenager
    '20-29', # Young Adult
    '30-39', # Adult
    '40-49', # Middle-aged Adult
    '50-59', # Older Adult
    '60-69', # Senior Adult
    '70+'    # Elderly
]

# 🚀 Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🚀 Enhanced Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 ResNet-50 Model with Dropout Regularization
class ResNet50DualHead(nn.Module):
    def __init__(self, num_age_classes=10, num_gender_classes=2):
        super(ResNet50DualHead, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Identity()
        
        # Age Head with Dropout to prevent overfitting
        self.age_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_age_classes)
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_gender_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age_logits, gender_logits

# 🚀 Initialize the model
resnet_model = ResNet50DualHead().to(device)

# 🚀 Streamlit App Interface
st.title('Age and Gender Prediction App')
st.write('Upload an image to detect age group and gender using ResNet-50.')

uploaded_file = st.file_uploader('Upload an Image...', type=['jpg', 'png', 'jpeg'])

# 🚀 Prediction Logic
def predict(image, model):
    image_tensor = transform(image).unsqueeze(0).to(device)
    age_logits, gender_logits = model(image_tensor)

    # Apply Softmax to get probabilities
    age_confidences = torch.softmax(age_logits, dim=1)
    gender_confidences = torch.softmax(gender_logits, dim=1)

    # Get the predicted class and confidence
    age_index = torch.argmax(age_confidences, dim=1).item()
    gender_index = torch.argmax(gender_confidences, dim=1).item()

    # Map the index to labels
    predicted_age = age_brackets[age_index]
    age_confidence = age_confidences[0][age_index].item()
    predicted_gender = 'Male' if gender_index == 0 else 'Female'
    gender_confidence = gender_confidences[0][gender_index].item()

    return predicted_age, age_confidence, predicted_gender, gender_confidence

# 🚀 Streamlit Display Logic
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner('Processing the image...'):
            sleep(2)
            age, age_conf, gender, gender_conf = predict(image, resnet_model)
            st.write('### ResNet-50 Predictions:')
            st.write(f'**Age Group:** {age} (Confidence: {age_conf:.2f})')
            st.write(f'**Gender:** {gender} (Confidence: {gender_conf:.2f})')
            st.write('---')

        st.success('Prediction Complete!')

    except UnidentifiedImageError:
        st.error('The uploaded file is not a valid image. Please upload a JPG or PNG image.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
