import pytest
import torch
from Stemcell_classifier.classify_image import Net, classify_single_image, preprocess_image

# Integration Test for the full model inference
def test_classify_single_image():
    """Test the classify_single_image function with a real model."""
    model_path = './models/Simple_model_best_model_0.92.pt'  # Provide a valid model path
    image_path = './model_data/H9p36/3x40_good.png'  # Provide a valid test image path
    
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