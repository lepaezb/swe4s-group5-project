import pytest
import torch
import numpy as np
from torchvision import transforms
from Stemcell_classifier.classify_image import Net, preprocess_image, conv3x3

# Unit Test for conv3x3 function
def test_conv3x3():
    """Test conv3x3 layer creation."""
    conv_layer = conv3x3(1, 16, pool=True)
    assert isinstance(conv_layer, torch.nn.Sequential)
    assert len(conv_layer) == 4  # Conv2d, PReLU, BatchNorm2d, MaxPool2d

def test_net_initialization():
    """Test the initialization of the Net model."""
    model = Net(thickness=4)
    assert isinstance(model, Net)
    assert model.conv1[0].in_channels == 1  # First conv layer should take 1 channel
    assert model.fc1.in_features == 512  # Flattened input size should match

def test_preprocess_image():
    """Test the image preprocessing."""
    image_path = './model_data/H9p36/3x40_good.png'  # Provide a valid test image path
    image_tensor = preprocess_image(image_path)
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.ndimension() == 3  # Shape should be (1, H, W)
    assert image_tensor.dtype == torch.float32

# Run the tests
if __name__ == '__main__':
    pytest.main()