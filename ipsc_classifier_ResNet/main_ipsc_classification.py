# ipsc_classification.py

# --- 0. Import necessary libraries ---
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. Image Loading and Label Extraction ---
def load_images(directory):
    """Loads image file paths and corresponding labels from the specified directory.
    Args:
        directory (str): Path to the directory containing images.
    Returns:
        Tuple[List[str], List[int]]: Lists of image paths and labels (0 for good, 1 for bad).
    """
    image_paths, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            label = 0 if 'good' in filename.lower() else 1
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels

# --- 3. Custom Dataset Class ---
class ColonyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            print(f"Error loading image: {self.image_paths[idx]}")
            # Create a blank image if loading fails
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# --- 4. Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced size for faster processing
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# --- 5. Model Definition ---
class iPSCClassifier(nn.Module):
    def __init__(self):
        super(iPSCClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # Binary classification

    def forward(self, x):
        x = self.resnet(x)
        return x

# --- 6. Training Function ---
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model

# --- 7. Model Evaluation ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_inputs, all_labels, all_predictions = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_inputs.extend(inputs.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predictions.cpu())
    accuracy = correct / total
    print(f'Test Loss: {test_loss / total:.4f}, Test Accuracy: {accuracy * 100:.2f}%')
    return all_inputs, all_labels, all_predictions

# --- 8. Display Images ---
def display_examples(inputs, labels, predictions=None, num_samples=20):
    """
    Displays images with their labels, and predictions if provided.
    """
    num_samples = min(num_samples, len(inputs))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Denormalize image
        image = inputs[i].clone()
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        image = image.clamp(0, 1)
        image = transforms.ToPILImage()(image)
        axes[i].imshow(image)
        label = 'Good' if labels[i].item() == 0 else 'Bad'
        if predictions is not None:
            prediction = 'Good' if predictions[i].item() == 0 else 'Bad'
            axes[i].set_title(f'Actual: {label}\nPredicted: {prediction}')
        else:
            axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration and random seed
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load images and labels
    image_directory = 'model_data/H9p36/'
    image_paths, labels = load_images(image_directory)
    print(f"Total images loaded: {len(image_paths)}")
    # Split data into training and testing sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=seed
    )
    print(f"Training samples: {len(train_paths)}")
    print(f"Testing samples: {len(test_paths)}")
    # Create datasets and dataloaders
    train_dataset = ColonyDataset(train_paths, train_labels, transform=data_transforms)
    test_dataset = ColonyDataset(test_paths, test_labels, transform=data_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # Display examples from the training set
    print("Displaying examples from the training set:")
    sample_inputs, sample_labels = next(iter(train_dataloader))
    display_examples(sample_inputs, sample_labels, num_samples=20)
    # Create the model, criterion, and optimizer
    model = iPSCClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Load or train the model
    model_weights_path = 'model_weights.pth'
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Loaded pre-trained model weights.")
    else:
        print("Training the model from scratch.")
        model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=3)
        torch.save(model.state_dict(), model_weights_path)
    # Evaluate the model on the test dataset
    print("Evaluating the model on the test dataset...")
    all_inputs, all_labels, all_predictions = evaluate_model(model, test_dataloader, criterion)
    # Display examples of model predictions
    print("Displaying examples of model predictions:")
    display_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=20)