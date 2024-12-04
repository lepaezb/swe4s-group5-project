"""
Test the segmentation_cellpose.py module: determine that the directory/file reading functions work on mock files, and that the 
cellpose segmentation returns the expected output statement.
"""
import pytest
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from segmentation_cellpose.py import read_all_tiff_files, run_cellpose, main  
import tifffile as tiff
import matplotlib.pyplot as plt

@pytest.fixture
def mock_directory(tmp_path):
    """Create a temporary directory with sample TIFF files."""
    files = ["image1.tif", "image2.tif"]
    for file in files:
        tiff_path = tmp_path / file
        tiff_path.write_bytes(b"Fake TIFF data")
    return tmp_path

def test_read_all_tiff_files(mock_directory):
    """Test reading TIFF files from a directory."""
    result = read_all_tiff_files(str(mock_directory), channel_number=2)
    assert len(result) == 2
    assert all(isinstance(img, np.ndarray) for img in result.values())

@patch("segmentation_cellpose.model.eval")
def test_run_cellpose(mock_model_eval):
    """Test running CellPose segmentation."""
    mock_model_eval.return_value = (np.zeros((10, 10)), None, None, None)
    img = np.zeros((256, 256), dtype=np.float32)
    result = run_cellpose(img, channel_number=2, diam=100.0)
    assert result == "Segmentation complete."


# Assuming run_cellpose is in the module 'segmentation_cellpose'
# from segmentation_cellpose import run_cellpose
# Mocking the model.eval() method
class MockModel:
    def eval(self, img, channels, diameter, flow_threshold):
        # Return mock masks, flows, styles, and diams
        masks = np.zeros_like(img[0], dtype=int)  # Mock mask (zeros)
        flows = None  # Mock flow data
        styles = None  # Mock style data
        diams = None  # Mock diameter data
        return masks, flows, styles, diams

# Assuming run_cellpose function is in the module "cellpose_module"
from cellpose_module import run_cellpose  # Adjust import based on actual location

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_image():
    # Create a mock image (1x256x256 grayscale image)
    return np.random.rand(1, 256, 256)

def test_run_cellpose_output_file(mock_model, mock_image, tmpdir):
    # Given parameters
    direct = tmpdir  # Temporary directory
    num = 1
    channel_number = 0
    diam = 30
    
    # Replace the model with the mock model
    global model
    model = mock_model
    
    # Call the run_cellpose function
    masks = run_cellpose(direct, num, mock_image, channel_number, diam)
    
    # Check that masks is returned and is a numpy array
    assert isinstance(masks, np.ndarray), "Masks should be a numpy array."
    assert masks.shape == mock_image[0].shape, "Masks shape doesn't match the input image."

    # Check that the output file is created
    output_file = direct.join(f"output{num}.png")
    assert os.path.exists(output_file), f"Output file {output_file} was not created."

    # Optionally, you can check if the output file is non-empty (i.e., the image was saved)
    # Open the saved image and check if it is not empty
    img_output = plt.imread(output_file)
    assert img_output.size > 0, "Saved image file is empty."






