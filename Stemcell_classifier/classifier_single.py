

# This Module will accept a path to a model file and a path to an image file as arguments.
# It will load the model, preprocess the image, and make a prediction using the model.
# The prediction will be printed to the console.
# USAGE: python classifier_single.py path_to_your_model.pth path_to_your_image.jpg

import sys
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

# Define your model architecture here
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
    '''3x3 convolution with padding and optional max pooling'''
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
    '''Load a PyTorch model from a file'''
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
    '''Preprocess an image for model inference'''
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
    '''Make a prediction with the model'''
    with torch.no_grad():
        outputs = model(img_tensor)
        binary_output = (outputs >= 0.5).float()
    return outputs

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        sys.exit(1)

    input_size = (256, 256)  # Adjust input size as needed
    model = load_model(model_path)
    img_tensor = preprocess_image(image_path, input_size)  # Adjust input_size as needed
    outputs = predict_image(model, img_tensor)
    binary_output = (outputs >= 0.5).float()

print(f"likeleyhood of healthy colony: {outputs.item()}")
print(f"Binary Prediction: {binary_output.item()}. (0 = unhealthy, 1 = healthy)")

if binary_output.item() == 1:
    print("The model predicts the cells are healthy.")
else:
    print("The model predicts the cells are unhealthy.")