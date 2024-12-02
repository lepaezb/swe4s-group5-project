import argparse
import numpy as np
import torch
from PIL import Image
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.color import rgb2gray


def conv3x3(in_channels, out_channels, pool=False):
    """
    Creates a convolutional block with optional pooling.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool (bool): If True, adds a max-pooling layer.
    
    Returns:
        torch.nn.Sequential: A convolutional block.
    """
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.PReLU(),
        torch.nn.BatchNorm2d(out_channels),
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(kernel_size=2))
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    """
    A convolutional neural network for binary classification.
    
    Args:
        thickness (int): Base channel multiplier for convolutional layers.
    """
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


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesses an image for model inference.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing.
    
    Returns:
        torch.Tensor: Preprocessed image as a PyTorch tensor.
    """
    image = Image.open(image_path).resize(target_size, Image.LANCZOS)
    image = np.array(image)
    if image.ndim == 3 and image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image_gray = rgb2gray(image)
    image_gray = img_as_ubyte(image_gray)
    image_eq = exposure.equalize_hist(image_gray)
    image_eq = np.expand_dims(image_eq, axis=0)  # Add channel dimension
    return torch.tensor(image_eq, dtype=torch.float32)


def classify_single_image(model, image_path, threshold=0.7):
    """
    Classifies a single image using a trained model.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        image_path (str): Path to the image file.
        threshold (float): Threshold for binary classification.
    
    Prints:
        The classification label and confidence.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_tensor = preprocess_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor).item()

    label = "healthy" if output >= threshold else "unhealthy"
    print(f"This colony is {label} with confidence {output:.4f}")


def main():
    """
    Main function to load the model, preprocess an image, and make a prediction.
    """
    parser = argparse.ArgumentParser(description="Classify a single image using a trained model.")
    parser.add_argument("--model-path", type=str, help="Path to the trained model file (.pth).")
    parser.add_argument("--image-path", type=str, help="Path to the image file to classify.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for classification.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(thickness=4)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    classify_single_image(model, args.image_path, args.threshold)


if __name__ == "__main__":
    main()