# Author: Xmind404 Franciszek Karbowniczek

# Include necessary imports
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os


# Define constants
MODEL_PATH = "models/emotion_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A simple CNN architecture for emotion recognition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


# Load the pre-trained model
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint['classes']
    num_classes = len(classes)
    model = SimpleCNN(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, classes


# Predict emotion from an image
def predict_emotion(image_path):
    model, classes = load_model()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)
    return classes[pred.item()]