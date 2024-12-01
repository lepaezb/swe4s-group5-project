

# This Module will accept a path to a model file and a path to an image file as arguments.
# It will load the model, preprocess the image, and make a prediction using the model.
# The prediction will be printed to the console.
# USAGE: python classifier_single.py path_to_your_model.pth path_to_your_image.png


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.color import rgb2gray
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Net(torch.nn.Module):
    ''' VGG13 convolutional neural network'''
	
    def __init__(self, thickness=4):

        super(Net, self).__init__()
        self.conv1 = conv3x3(in_channels=1, out_channels=thickness)
        self.conv2 = conv3x3(in_channels=thickness, out_channels=thickness, pool=True)

        self.conv3 = conv3x3(in_channels=thickness, out_channels=thickness*2)
        self.conv4 = conv3x3(in_channels=thickness*2, out_channels=thickness*2, pool=True)

        self.conv5 = conv3x3(in_channels=thickness*2, out_channels=thickness*4)
        self.conv6 = conv3x3(in_channels=thickness*4, out_channels=thickness*4, pool=True)

        self.conv7 = conv3x3(in_channels=thickness*4, out_channels=thickness*8)
        self.conv8 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.conv9 = conv3x3(in_channels=thickness*8, out_channels=thickness*8)
        self.conv10 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.conv11 = conv3x3(in_channels=thickness*8, out_channels=thickness*8)
        self.conv12 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.fc1 = torch.nn.Linear(thickness * 8 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)    # 16 256 256
        x = self.conv2(x)          

        x = self.conv3(x)   # 32  128 128
        x = self.conv4(x)   

        x = self.conv5(x)   # 64  64 64
        x = self.conv6(x) 

        x = self.conv7(x)   # 128 32 32
        x = self.conv8(x)   

        x = self.conv9(x) # 128 16 16
        x = self.conv10(x)

        x = self.conv11(x) # 128 8 8
        x = self.conv12(x) # 128 4 4

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        return torch.sigmoid(x)
    
    # Define the model architecture
def conv3x3(in_channels, out_channels, pool=False, dropout=None):
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        torch.nn.PReLU(),
        torch.nn.BatchNorm2d(out_channels),
    ]

    if pool:
        layers.append(torch.nn.MaxPool2d((2, 2)))

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return torch.nn.Sequential(*layers)

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image)
    if image.shape[-1] == 4:
        image = rgba2rgb(image)
    image_gray = rgb2gray(image)  # Convert to grayscale
    img = img_as_ubyte(image_gray)
    image_eq = exposure.equalize_hist(img)
    image_eq = np.expand_dims(image_eq, axis=0)  # Shape: (1, H, W)
    image_tensor = torch.from_numpy(image_eq).type(torch.float32)
    return image_tensor

def classify_single_image(model, image_path, threshold=0.7):
    """
    Classify a single image using a trained model.
    Args:
      - model: Trained PyTorch model.
      - image_path: Path to the image to classify.
      - threshold: Confidence threshold for binary classification.
    """
    # Prepare for inference
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the image
    image_tensor = preprocess_image(image_path).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0)).squeeze()
        prediction = torch.sigmoid(output).item()

    # Determine label
    label = 'healthy' if prediction >= threshold else 'unhealthy'
    print(f"This colony is {label} with confidence {prediction:.4f}")

# Example usage
model_path = './models/Simple_model_best_model_0.92.pt'  # Update with your model path
image_path = './model_data/H9p36/20Bx40_bad.png'  # Update with your image path

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(thickness=4)  # Use the same thickness as in the training script
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Classify the image
classify_single_image(model, image_path)