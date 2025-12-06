# Author: Xmind404 Franciszek Karbowniczek

# include necessary imports
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function to create data loaders for training and validation datasets
def get_data_loaders(data_dir, batch_size=32):

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Grayscale(),                 # Convert images to grayscale
        transforms.ToTensor(),                  # Convert images to tensor
        transforms.Normalize((0.5,), (0.5,))    # Normalize for grayscale images
    ])


    # Define structure for training and validation datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes