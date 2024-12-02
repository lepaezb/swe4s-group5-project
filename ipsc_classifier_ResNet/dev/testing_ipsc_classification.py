# quick_test_colony_classification.py

# --- 0. Import necessary libraries ---
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 2. Image Loading and Label Extraction ---
def load_images(directory, max_images):
    """
    Loads a limited number of image file paths and corresponding labels from the specified directory.
    Args:
        directory (str): Path to the directory containing images.
        max_images (int): Maximum number of images to load.
    Returns:
        Tuple[List[str], List[int]]: Lists of image paths and labels (0 for good, 1 for bad).
    """
    image_paths, labels = [], []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            label = 0 if 'good' in filename.lower() else 1
            image_paths.append(img_path)
            labels.append(label)
            count += 1
            if count >= max_images:
                break
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
    transforms.Resize((64, 64)),  # Smaller size for faster processing
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # Simplified normalization
                         [0.5, 0.5, 0.5])
])

# --- 5. Simplified Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # Input channels, output channels, kernel size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Kernel size, stride
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 32 * 32, 1)  # Adjust input features based on image size
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 6. Training Function ---
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model

# --- 7. Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_inputs, all_labels, all_predictions = [], [], []  # For visualization
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid activation
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_inputs.extend(inputs.cpu())
            all_labels.extend(labels.cpu())
            all_predictions.extend(predictions.cpu())
    accuracy = correct / total
    print(f'Test Loss: {test_loss / total:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

# --- 8. Display Examples Function ---
def display_examples(inputs, labels, predictions=None, num_samples=10):
    """
    Displays images with their labels, and predictions if provided.
    """
    num_samples = min(num_samples, len(inputs))
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Denormalize image
        image = inputs[i].clone()
        image = image * 0.5 + 0.5  # Reverse normalization
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
    # Device configuration 
    seed = 42
    max_images = 20
    set_seed(seed)
    device = torch.device("cpu")  # Use CPU for simplicity
    print(f"Using device: {device}")

    # --- 9. Load images and labels ---
    image_directory = 'model_data/H9p36/' 
    image_paths, labels = load_images(image_directory, max_images)  
    print(f"Total images loaded: {len(image_paths)}")

    # --- 10. Split data into training and testing sets ---
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=seed
    )
    print(f"Training samples: {len(train_paths)}")
    print(f"Testing samples: {len(test_paths)}")

    # --- 11. Create datasets and dataloaders ---
    train_dataset = ColonyDataset(train_paths, train_labels, transform=data_transforms)
    test_dataset = ColonyDataset(test_paths, test_labels, transform=data_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    # --- 12. Display examples from the dataset ---
    print("Displaying examples from the training set:")
    sample_inputs, sample_labels = next(iter(train_dataloader))
    display_examples(sample_inputs, sample_labels, num_samples=10)

    # --- 13. Initialize the model, criterion, and optimizer ---
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 14. Train the model ---
    print("Training the model...")
    model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=3)

    # --- 15. Evaluate the model ---
    print("Evaluating the model on the test dataset...")
    all_inputs, all_labels, all_predictions = evaluate_model(model, test_dataloader)

    # --- 16. Display examples of model predictions ---
    print("Displaying examples of model predictions:")
    display_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=10)