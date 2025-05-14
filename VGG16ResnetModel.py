#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define the modified VGG-16 model with two heads (age and gender)
class VGG16DualHead(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2):
        super(VGG16DualHead, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier with a custom head
        self.base_model.classifier = nn.Identity()  # Remove the existing classifier
        
        # Age head (classifier)
        self.age_head = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, num_age_classes)
        )
        
        # Gender head (classifier)
        self.gender_head = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, num_gender_classes)
        )

    def forward(self, x):
        features = self.base_model.features(x)
        features = features.view(features.size(0), -1)

        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)

        age_logits = torch.softmax(age_logits, dim=1)  # Apply softmax for age ranges

        return age_logits, gender_logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = VGG16DualHead(num_age_classes=9, num_gender_classes=2).to(device)

# Define criterion and optimizer
age_criterion = nn.CrossEntropyLoss()
gender_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training function (same as provided)
model, history = train_and_validate(model, age_criterion, gender_criterion, optimizer, train_loader, test_loader, epochs=5)

# Save the state dict
torch.save(model.state_dict(), "vgg16_age_gender_state_dict.pth")
print("State dict saved as vgg16_age_gender_state_dict.pth")

# Define the checkpoint path
checkpoint_path = "vgg16_checkpoint.pth"

# Save the model, optimizer, and history
torch.save({
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, checkpoint_path)

print(f"Model saved successfully at {checkpoint_path}")

# Load the checkpoint
checkpoint = torch.load("vgg16_checkpoint.pth")

# Load model and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
history = checkpoint['history']
print(f"Model and optimizer loaded successfully from vgg16_checkpoint.pth")

# Make sure your model architecture is defined before loading
model = VGG16DualHead()  # Initialize your model class
model.load_state_dict(torch.load("vgg16_age_gender_state_dict.pth"))
model.eval()
print(type(history))
print(history)
# In[ ]:
