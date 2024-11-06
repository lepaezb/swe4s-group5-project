import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys

class ImageClassifier:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust for single-channel input
        ])

    def load_model(self, model_path):
        model = Net()  # Initialize your model
        try:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict, strict=False)  # Ignore unexpected keys
            model.eval()  # Set the model to evaluation mode
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def classify_image(self, image_path):
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(image)
            prediction = torch.sigmoid(output).item()
        
        return prediction

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
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if pool:
        layers.append(nn.MaxPool2d((2, 2)))
    return nn.Sequential(*layers)

def conv3x3_maxpool4(in_channels, out_channels, pool=False, dropout=None):
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

if __name__ == "__main__":
    model_path = 'Experiments/VGG13_histeq_tff/Simple_model_best_model_0.91.pt'
    image_path = 'H9p36/14Cx40_bad.png'

    classifier = ImageClassifier(model_path)
    prediction = classifier.classify_image(image_path)
    print(f'Prediction: {prediction}')