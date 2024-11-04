
# This Module will accept a path to a model file, a path to a directory with images(pngs) as arguments and an output path.
# It will load the model, preprocess the images, and make a prediction using the model.
# The prediction will be appended to a csv file at the output path.
# USAGE: python classifier_multi.py path_to_your_model.pth path_to_your_image.jpg output_path.csv

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import argparse

class Net(nn.Module):
    ''' VGG13 convolutional neural network'''
    
    def __init__(self, thickness=8):
        super(Net, self).__init__()
        self.conv1 = conv3x3(in_channels=1, out_channels=thickness)
        self.conv2 = conv3x3_maxpool4(in_channels=thickness, out_channels=thickness, pool=True)

        self.conv3 = conv3x3(in_channels=thickness, out_channels=thickness*2)
        self.conv4 = conv3x3(in_channels=thickness*2, out_channels=thickness*2, pool=True)

        self.conv5 = conv3x3(in_channels=thickness*2, out_channels=thickness*4)
        self.conv6 = conv3x3(in_channels=thickness*4, out_channels=thickness*4, pool=True)

        self.conv7 = conv3x3(in_channels=thickness*4, out_channels=thickness*8)
        self.conv8 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.conv9 = conv3x3(in_channels=thickness*8, out_channels=thickness*8)
        self.conv10 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)
      
        self.fc1 = nn.Linear(thickness * 8 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)   # 16 256 256
        x = self.conv2(x)   # 16 64 64 
        
        x = self.conv3(x)   # 32 64 64
        x = self.conv4(x)   # 32 32 32 

        x = self.conv5(x)   # 64 32 32
        x = self.conv6(x)   # 64 16 16

        x = self.conv7(x)   # 128 16 16
        x = self.conv8(x)   # 128 8 8

        x = self.conv9(x)   # 128 8 8
        x = self.conv10(x)  # 128 4 4
        
        x = x.view(x.shape[0], -1)
       
        x = self.fc1(x)
        return torch.sigmoid(x)

def conv3x3(in_channels, out_channels, pool=False):
    '''3x3 convolution with padding'''
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if pool:
        layers.append(nn.MaxPool2d((2, 2)))
    return nn.Sequential(*layers)

def conv3x3_maxpool4(in_channels, out_channels, pool=False, dropout=None):
    '''3x3 convolution with padding, PReLU activation, and batch normalization'''
    layers = [
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
    ]

    if pool:
        layers.append(nn.MaxPool2d((4, 4)))

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

def load_model(model_path):
    '''Load a CNN model from a file'''
    model = Net()  # Initialize your model
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)  # Ignore unexpected keys
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_image(img_path, input_size):
    '''Preprocess images for the model'''
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust mean and std for grayscale
    ])
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_image(model, img_tensor):
    '''Make a prediction using the pre-trained model'''
    with torch.no_grad():
        outputs = model(img_tensor)
        binary_output = (outputs >= 0.5).float()
    return outputs, binary_output

def process_directory(model_path, image_dir, output_csv):
    '''Classify images as 0 or 1 (unhealthy or healthy) in a directory using a pre-trained model'''
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist.")
        sys.exit(1)

    input_size = (256, 256)  # Adjust input size as needed
    model = load_model(model_path)

    results = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if os.path.isfile(img_path):
            img_tensor = preprocess_image(img_path, input_size)
            outputs, binary_output = predict_image(model, img_tensor)
            results.append({
                'Image': img_name,
                'Confidence Score': outputs.item(),
                'Binary Prediction': binary_output.item()
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify images in a directory using a pre-trained model.')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    parser.add_argument('image_dir', type=str, help='Directory containing images to classify')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    process_directory(args.model_path, args.image_dir, args.output_csv)