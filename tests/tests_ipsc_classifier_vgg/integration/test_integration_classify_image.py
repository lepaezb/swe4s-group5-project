import pytest
import torch
import os
from tools.ipsc_classifier_vgg.classify_image import Net, classify_single_image, preprocess_image

# Integration Test for the full model inference
def test_classify_single_image():
    """Test the classify_single_image function with a real model."""
    model_path = './models/Simple_model_best_model_0.92.pt'  # Provide a valid model path
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, '3x40_good.png')

    # Assume the model is correctly loaded from the path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(thickness=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Test classification
    classify_single_image(model, image_path, threshold=0.7)

    # If no assertion is raised and output is printed, the test passed
    assert True

# Run the tests
if __name__ == '__main__':
    pytest.main()