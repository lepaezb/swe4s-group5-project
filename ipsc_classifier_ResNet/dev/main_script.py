# Ignore CellPose warning due to future package updates
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 0. Import necessary libraries ---
import os
import cv2
import numpy as np
from cellpose import models as cp_models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --- 1. Image Loading and Label Extraction ---
def load_images(directory):
    """Loads image file paths and corresponding labels from the specified directory.
    Args:
        directory (str): Path to the directory containing images.
    Returns:
        Tuple[List[str], List[int]]: Lists of image paths and labels (0 for good, 1 for bad).
    """
    image_paths, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(directory, filename)
            label = 0 if 'good' in filename.lower() else 1
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels

# --- 2. Colony Segmentation with CellPose ---
class ColonySegmenter:
    """Segments colonies in images using the CellPose model.
    Args:
        use_gpu (bool): If True, uses GPU for CellPose (if available).
    """
    def __init__(self, use_gpu=False):
        self.model = cp_models.Cellpose(gpu=use_gpu, model_type='cyto')

    def segment_colonies(self, image, diameter=30):
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        masks, _, _, _ = self.model.eval(image, diameter=diameter, channels=[0, 0])
        return masks

# --- 3. Colony Extraction ---
def extract_colonies(image, masks):
    """Extracts individual colonies from the image based on segmentation masks.
    Args:
        image (np.array): Original image.
        masks (np.array): Segmentation masks for colonies.
    Returns:
        colonies (List[np.array]): List of colony images.
    """
    colonies = []
    for colony_id in np.unique(masks)[1:]:  # Skip background
        colony_mask = (masks == colony_id).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(colony_mask)
        colony_img = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=colony_mask[y:y+h, x:x+w])
        colonies.append(colony_img)
    return colonies

# --- 4. Custom Dataset Class ---
class ColonyDataset(Dataset):
    def __init__(self, image_paths, labels, segmenter, transform=None):
        self.colony_images, self.colony_labels = [], []
        self.transform = transform
        for img_path, label in zip(image_paths, labels):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error loading image: {img_path}")
                continue
            masks = segmenter.segment_colonies(image)
            colonies = extract_colonies(image, masks)
            self.colony_images.extend(colonies)
            self.colony_labels.extend([label] * len(colonies))

    def __len__(self):
        return len(self.colony_images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(self.colony_images[idx], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, self.colony_labels[idx]

# --- 5. Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced size for faster processing
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# --- 6. Model Definition ---
class iPSCClassifier(nn.Module):
    def __init__(self):
        super(iPSCClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use ResNet18 for lighter model
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet(x)
        return x
        
# --- 7. Training Function ---
def train_model(model, dataloader, criterion, optimizer, num_epochs=3):  # Fewer epochs for faster training
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

# --- 8. Model Evaluation ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_inputs, all_labels, all_predictions = [], [], [] # For visualization
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Try sigmoid activation for binary classification
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_inputs.extend(inputs.cpu()) # Save inputs for visualization
            all_labels.extend(labels.cpu()) # Save labels for visualization
            all_predictions.extend(predictions.cpu()) # Save predictions for visualization
    accuracy = correct / total
    print(f'Test Loss: {test_loss / total:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Display some examples of model predictions
    num_samples = 5
    print(f"Displaying {num_samples} examples of model predictions:")
    # Denormalize images
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                         std=[1/0.229, 1/0.224, 1/0.225]) # Inverse of normalization
    inputs_denorm = [inv_normalize(input) for input in all_inputs[:num_samples]]
    # Convert tensors to images
    images = [transforms.ToPILImage()(input) for input in inputs_denorm]
    # Plotting
    display_colonies(images, all_predictions[:num_samples])


# --- 9. Display segmented colonies ---
def display_colonies(dataset,labels, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        image = cv2.cvtColor(dataset[i], cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].set_title('Good' if labels[i] == 0 else 'Bad')
        axes[i].axis('off')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter = ColonySegmenter(use_gpu=torch.cuda.is_available())
    image_directory = 'model_data/H9p36/'
    image_paths, labels = load_images(image_directory)

    # Split data into training (80%) and testing (20%) sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create Datasets and show some examples of segmented colonies
    train_dataset = ColonyDataset(train_paths, train_labels, segmenter, transform=data_transforms)
    test_dataset = ColonyDataset(test_paths, test_labels, segmenter, transform=data_transforms)
    print(f"Number of training samples: {len(train_dataset)}. Here are some examples:")
    display_colonies(train_dataset.colony_images, train_dataset.colony_labels)
    print(f"Number of testing samples: {len(test_dataset)}. Here are some examples:")
    display_colonies(test_dataset.colony_images, test_dataset.colony_labels)
    
    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize the model, criterion, and optimizer
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
    evaluate_model(model, test_dataloader, criterion)