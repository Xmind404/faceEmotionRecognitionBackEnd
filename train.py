# Author: Xmind404 Franciszek Karbowniczek

# include necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from teach import get_data_loaders
import os

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, classes = get_data_loaders(DATA_DIR, BATCH_SIZE)
NUM_CLASSES = len(classes)


# Define a simple CNN model
class SimpleCNN(nn.Module):

    # Constructor
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    # Forward pass
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
model_path = os.path.join(MODEL_DIR, 'emotion_cnn.pth')
torch.save({'model_state_dict': model.state_dict(),'classes': classes}, model_path)
print(f"Model saved to {model_path}")