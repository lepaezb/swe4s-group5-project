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

# --- 1. Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. Image Loading and Label Extraction ---
def load_images(directory, max_images=None):
    """
    Loads image file paths and corresponding labels from the specified directory.
    Args:
        directory (str): Path to the directory containing images.
        max_images (int, optional): Maximum number of images to load.
    Returns:
        Tuple[List[str], List[int]]: Lists of image paths and labels (0 for good, 1 for bad).
    """
    image_paths, labels = [], []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            if 'good' in filename.lower():
                label = 0
            elif 'bad' in filename.lower():
                label = 1
            else:
                print(f"Filename {filename} does not contain 'good' or 'bad'. Skipping.")
                continue
            image_paths.append(img_path)
            labels.append(label)
            count += 1
            if max_images is not None and count >= max_images:
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
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# --- 4. Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],  # Simplified normalization
                         [0.5, 0.5, 0.5])
])

# --- 5. Simplified Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Increased channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Added another conv layer
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        sample_input = torch.zeros(1, 3, 64, 64)
        sample_output = self.conv_layers(sample_input)
        num_features = sample_output.numel()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 1)
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
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1).to(device)
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
    all_inputs, all_labels, all_predictions = [], [], []  # For saving
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1).to(device)
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
    return all_inputs, all_labels, all_predictions  # Return these for saving images

# --- 8. Save Examples Function ---
def save_examples(inputs, labels, predictions=None, num_samples=10, save_dir='saved_images', prefix='example'):
    """
    Saves images with their labels, and predictions if provided, to the specified directory.
    """
    num_samples = min(num_samples, len(inputs))
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    for i in range(num_samples):
        # Denormalize image
        image = inputs[i].clone()
        image = image * 0.5 + 0.5  # Reverse normalization
        image = image.clamp(0, 1)
        image = transforms.ToPILImage()(image)
        label = 'Good' if labels[i].item() == 0 else 'Bad'
        if predictions is not None:
            prediction = 'Good' if predictions[i].item() == 0 else 'Bad'
            title = f'Actual_{label}_Predicted_{prediction}'
        else:
            title = f'Label_{label}'
        # Save image
        image_filename = f"{prefix}_{i}_{title}.png"
        image_path = os.path.join(save_dir, image_filename)
        image.save(image_path)
    print(f"Images saved to {save_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration
    seed = 42
    max_images = None  # Process all images
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 9. Load images and labels ---
    image_directory = '/scratch/Users/lupa9404/swe4s_project/images/'
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
    batch_size = 64  # Increase batch size
    num_workers = 8  # Adjust based on HPC node

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- 12. Save examples from the dataset ---
    print("Saving examples from the training set...")
    sample_inputs, sample_labels = next(iter(train_dataloader))
    save_examples(sample_inputs, sample_labels, num_samples=10, save_dir='training_examples', prefix='train')

    # --- 13. Initialize the model, criterion, and optimizer ---
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 14. Train the model ---
    num_epochs = 10  # Increase number of epochs
    print("Training the model...")
    model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs)

    # --- 15. Save the trained model ---
    model_save_path = '/scratch/Users/lupa9404/swe4s_project/trained_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- 16. Evaluate the model ---
    print("Evaluating the model on the test dataset...")
    all_inputs, all_labels, all_predictions = evaluate_model(model, test_dataloader, criterion)

    # --- 17. Save examples of model predictions ---
    print("Saving examples of model predictions...")
    save_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=10, save_dir='test_predictions', prefix='test')
