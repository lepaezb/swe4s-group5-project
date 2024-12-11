# --- 0. Import necessary libraries ---
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import copy

# --- 1. Set random seeds for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. Image Loading and Label Extraction ---
def load_images(directory, max_images=None):
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
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# --- 4. Data Transformations ---
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- 5. Enhanced Model Definition ---
def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif model_name == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
        return model
    elif model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
        return model
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Modify the final layer for ResNet architectures
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# --- 6. Training Function ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_precision_history = []
    val_precision_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                train_precision_history.append(epoch_precision)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                val_precision_history.append(epoch_precision)
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_precision_history, val_precision_history

# --- 7. Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_inputs, all_labels, all_predictions = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_inputs.extend(inputs.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    print(f'Test Loss: {test_loss / total:.4f}, Test Accuracy: {accuracy * 100:.2f}%, Test Precision: {precision * 100:.2f}%')
    return all_inputs, all_labels, all_predictions

# --- 8. Save Examples Function ---
def save_examples(inputs, labels, predictions=None, num_samples=10, save_dir='/scratch/Users/lupa9404/swe4s_project/results', prefix='example'):
    num_samples = min(num_samples, len(inputs))
    os.makedirs(save_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    for i in range(num_samples):
        image = inputs[i].cpu().clone()
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        image = image.clamp(0, 1)
        image = transforms.ToPILImage()(image)
        label = 'Good' if labels[i] == 0 else 'Bad'
        if predictions is not None:
            prediction = 'Good' if predictions[i] == 0 else 'Bad'
            title = f'Actual_{label}_Predicted_{prediction}'
        else:
            title = f'Label_{label}'
        image_filename = f"{prefix}_{i}_{title}.png"
        image_path = os.path.join(save_dir, image_filename)
        image.save(image_path)
    print(f"Images saved to {save_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    # Device configuration
    seed = 42
    max_images = None
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    # Load images and labels
    image_directory = '/scratch/Users/lupa9404/swe4s_project/training_images'
    image_paths, labels = load_images(image_directory, max_images)
    print(f"Total images loaded: {len(image_paths)}")

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=seed
    )
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # Create datasets and dataloaders
    train_dataset = ColonyDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = ColonyDataset(val_paths, val_labels, transform=test_transforms)
    batch_size = 64
    num_workers = 8

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Save examples from the training set
    print("Saving examples from the training set...")
    sample_inputs, sample_labels = next(iter(dataloaders['train']))
    save_examples(sample_inputs, sample_labels, num_samples=10, save_dir='/scratch/Users/lupa9404/swe4s_project/results', prefix='train')

    # Initialize models to train
    model_names = ['resnet18', 'resnet50', 'vgg16']
    num_epochs = 40

    # To store histories for each model
    model_histories = {}

    for model_name in model_names:
        print(f"\nTraining model: {model_name}")
        # Initialize the model, criterion, and optimizer
        model = get_model(model_name).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train the model
        print("Training the model...")
        model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_precision_history, val_precision_history = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

        # Save the trained model
        model_save_path = f'/scratch/Users/lupa9404/swe4s_project/results/trained_model_{model_name}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Store histories
        model_histories[model_name] = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_acc': train_acc_history,
            'val_acc': val_acc_history,
            'train_precision': train_precision_history,
            'val_precision': val_precision_history
        }

        # Evaluate the model
        print(f"Evaluating the model {model_name} on the validation dataset...")
        all_inputs, all_labels, all_predictions = evaluate_model(model, dataloaders['val'], criterion)

        # Save examples of model predictions
        print(f"Saving examples of model predictions for {model_name}...")
        save_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=10,
                      save_dir='/scratch/Users/lupa9404/swe4s_project/results', prefix=f'test_{model_name}')

    # Plot and save training history for all models
    plot_save_dir = '/scratch/Users/lupa9404/swe4s_project/results/'
    os.makedirs(plot_save_dir, exist_ok=True)

    # Plot Validation Loss for all models
    plt.figure()
    for model_name in model_names:
        plt.plot(range(1, num_epochs+1), model_histories[model_name]['val_loss'], label=f'{model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.title('Validation Loss for Different Models')
    plt.savefig(os.path.join(plot_save_dir, 'validation_loss_all_models.png'))
    plt.close()

    # Plot Validation Accuracy for all models
    plt.figure()
    for model_name in model_names:
        plt.plot(range(1, num_epochs+1), [acc * 100 for acc in model_histories[model_name]['val_acc']], label=f'{model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy for Different Models')
    plt.savefig(os.path.join(plot_save_dir, 'validation_accuracy_all_models.png'))
    plt.close()

    # Plot Validation Precision for all models
    plt.figure()
    for model_name in model_names:
        plt.plot(range(1, num_epochs+1), [prec * 100 for prec in model_histories[model_name]['val_precision']], label=f'{model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Precision (%)')
    plt.legend()
    plt.title('Validation Precision for Different Models')
    plt.savefig(os.path.join(plot_save_dir, 'validation_precision_all_models.png'))
    plt.close()