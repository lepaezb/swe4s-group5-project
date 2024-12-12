"""
Test the segmentation_cellpose.py module: determine that the directory/file reading functions work on mock files, and that the 
cellpose segmentation returns the expected output statement.
"""
import sys
sys.path.append("./../") 
import pytest
import os
import numpy as np
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock, call
from segmentation_cellpose import read_all_tiff_files, run_cellpose, is_fiji_open, subdir_segmentation
import tifffile 
import matplotlib.pyplot as plt
import psutil
import pandas as pd
from tifffile import TiffFile
# pip install pytest-mock
# Assuming run_cellpose function is in the module "cellpose_module"
# from cellpose_module import run_cellpose  # Adjust import based on actual location



# Create fake directories with different combinations of TIFF files
@pytest.fixture
def mock_directory(tmp_path):
    """Create a temporary directory with sample TIFF files."""
    files = ["image1.tif", "image2.tif"]
    for file in files:
        tiff_path = tmp_path / file
        for i in range(1, 3):
            img = np.zeros((10, 10), dtype=np.uint8)  # A simple 10x10 black image
            tiff_path = tmp_path / f"image{i}.tif"
            tifffile.imwrite(tiff_path, img)
    return tmp_path

@pytest.fixture
def mock_directory_none(tmp_path):
    """Create a temporary directory with no TIFF files."""
    files = ["image1.png", "image2.png"]
    for file in files:
        non_path = tmp_path / file
        non_path.write_bytes(b"Fake data") 
    return tmp_path

@pytest.fixture
def mock_directory_mixed(tmp_path):
    """Create a temporary directory with TIFF files and other filetypes."""
    files = ["image1.tif", "image2.png"]
    for file in files:
        mixed_path = tmp_path / file
        for file in files:
            mixed_path = tmp_path / file
            if file.endswith(".tif"):
                img = np.zeros((10, 10), dtype=np.uint8)  # Simple black 10x10 image
                tifffile.imwrite(mixed_path, img)  # Write a real TIFF file
            else:
                mixed_path.write_bytes(b"Fake non-TIFF data") 
    return tmp_path

# Test read_all_tiff_files function
class TestReadAllTiffFiles: 
    def test_file_Fxn(self, mock_directory):
        """Test reading TIFF files from a directory."""
        # Run read_all_tiff_files function
        result = read_all_tiff_files(str(mock_directory))
        assert len(result) == 2 # Validate length of returned dictionary
        assert all(isinstance(img, np.ndarray) for img in result.values()) # Check that all instances are an array (representative of an image)

    """Test reading files from directory with no TIFFs"""
    def test_no_tiffs(self, mock_directory_none):
        with pytest.raises(ValueError):  # Replace ValueError with the actual error you expect
            read_all_tiff_files(str(mock_directory_none))

    def test_mixed_tiffs(self, mock_directory_mixed):
        """Test reading TIFF files from a mixed directory"""
        result_mixed = read_all_tiff_files(str(mock_directory_mixed))
        assert len(result_mixed) == 1
        assert all(isinstance(img, np.ndarray) for img in result_mixed.values())





# Test is_fiji_open function
class TestIsFijiOpen:
    # Scenario 1: ImageJ process exists
    def test_fiji_open(self, mocker):
        mock_process_iter = mocker.patch('psutil.process_iter')
        mock_process_iter.return_value = [
            MagicMock(info={'name': 'ImageJ'}),
            MagicMock(info={'name': 'OtherProcess'}),
        ]
        assert is_fiji_open() == True, "Expected True when ImageJ is in the process list."

        # Scenario 2: No ImageJ process
    def test_fiji_closed(self, mocker):
        mock_process_iter = mocker.patch('psutil.process_iter')
        mock_process_iter.return_value = [
            MagicMock(info={'name': 'OtherProcess'}),
            MagicMock(info={'name': 'AnotherProcess'}),
        ]
        assert is_fiji_open() == False, "Expected False when ImageJ is not in the process list."

        # Scenario 3: Empty process list
    def test_no_processes(self, mocker):
        mock_process_iter = mocker.patch('psutil.process_iter')
        mock_process_iter.return_value = []
        assert is_fiji_open() == False, "Expected False when there are no processes."
   


'''
@pytest.fixture
def mock_directory_subdir(tmp_path):
    """Create a temporary directory with sub directories containing sample TIFF files."""
    dirs = ["image1_MASKED/", "image2_MASKED/"]
    for dir in dirs:
        files = ["image1.tif", "image2.tif", "image3.tif"]
        for file in files:
        tiff_path = tmp_path/dir/file
        for i in range(1, 4):
            img = np.zeros((10, 10), dtype=np.uint8)  # A simple 10x10 black image
            tiff_path = tmp_path / f"image{i}.tif"
            tifffile.imwrite(tiff_path, img)
    return tmp_path

def mock_directory_no_subdir(tmp_path):
    """Create a temporary directory with sub directories containing sample TIFF files."""
    dirs = ["image1_MASKED/", "image2/"]
    for dir in dirs:
        files = ["image1.tif", "image2.tif", "image3.tif"]
        for file in files:
        tiff_path = tmp_path/dir/file
        for i in range(1, 4):
            img = np.zeros((10, 10), dtype=np.uint8)  # A simple 10x10 black image
            tiff_path = tmp_path / f"image{i}.tif"
            tifffile.imwrite(tiff_path, img)
    return tmp_path
'''


# Test run_cellpose function
class TestSubdirSegmentation(unittest.TestCase):
    @patch('segmentation_cellpose.run_cellpose')
    @patch('segmentation_cellpose.read_all_tiff_files')
    @patch('pathlib.Path.iterdir')
    @patch('pathlib.Path.is_dir')
    @patch('pandas.DataFrame.to_excel')  # Mock Excel saving
    @patch('matplotlib.pyplot.savefig')  # Mock savefig to avoid creating actual plots
    @patch('matplotlib.pyplot.imshow')   # Mock imshow to avoid actual image display
    @patch('cellpose.models.Cellpose')  # Mock the Cellpose model
    @patch('sys.stdout', new_callable=MagicMock)
    def test_subdir_segmentation(self, mock_stdout, mock_Cellpose, mock_imshow, mock_savefig, mock_to_excel, mock_is_dir, mock_iterdir, mock_read_all_tiff_files, mock_run_cellpose):
        # Mock input data
        parent_directory = Path("/mock/parent_directory")
        channel_number = 1

        # Mock subdirectories (one with "MASKED" and one without)
        subdir_1 = parent_directory / "MASKED_subdir_1"
        subdir_2 = parent_directory / "MASKED_subdir_2"
        unrelated_subdir = parent_directory / "unrelated_subdir"

        # Mock directory structure (only the MASKED ones should be processed)
        mock_iterdir.return_value = [(subdir_1), (subdir_2), (unrelated_subdir)]
        mock_is_dir.return_value = [(subdir_1), (subdir_2)]
       

        # Mock TIFF images in each subdirectory
        mock_read_all_tiff_files.side_effect = lambda subdir: {
            "frame_1.tif": np.ones((10, 10), dtype=np.uint16),
            "frame_2.tif": np.ones((10, 10), dtype=np.uint16) * 2,
        } if "MASKED_subdir_1" in str(subdir) else {
            "frame_1.tif": np.ones((10, 10), dtype=np.uint16) * 3,
            "frame_2.tif": np.ones((10, 10), dtype=np.uint16) * 4,
        }

        # Mock the behavior of the Cellpose model
        mock_model = mock_Cellpose.return_value  # Mock instance of the model
        mock_model.eval.return_value = (np.array([[1, 0], [0, 1]]), None, None, None)  # Mock the eval output (masks)

        # Mock CellPose output (random segmentation mask)
        mock_run_cellpose.side_effect = lambda parent_dir, subdir, idx, img, channel: np.array([[1, 0], [0, 1]])  # Return a simple mask

        # Run the function
        subdir_segmentation(parent_directory, channel_number)

        # Assertions for directory iteration
        mock_iterdir.assert_called_once()

        # Assertions for TIFF reading (verify that the function reads files from the correct subdirectories)
        mock_read_all_tiff_files.assert_has_calls([
            call(str(subdir_1)),
            call(str(subdir_2)),
        ])

        # Assertions for Excel output (check that the DataFrame was saved for each subdirectory)
        self.assertEqual(mock_to_excel.call_count, 2)  # Two subdirectories should generate Excel files
        output_paths = [call_args[0][0] for call_args in mock_to_excel.call_args_list]
        self.assertIn(str(subdir_1 / "MASKED.xlsx"), output_paths)
        self.assertIn(str(subdir_2 / "MASKED.xlsx"), output_paths)



# Assuming run_cellpose is in your_module
# from your_module import run_cellpose

class TestRunCellpose(unittest.TestCase):
    @patch('matplotlib.pyplot.savefig')  # Mock savefig to avoid actual file writing
    @patch('matplotlib.pyplot.imshow')   # Mock imshow to avoid displaying images
    @patch('matplotlib.pyplot.draw')     # Mock draw to prevent figure drawing
    @patch('cellpose.models.Cellpose')   # Mock the CellPose model
    def test_run_cellpose(self, mock_Cellpose, mock_draw, mock_imshow, mock_savefig):
        # Mock the Cellpose model's eval method
        mock_model = mock_Cellpose.return_value
        mock_model.eval.return_value = (np.array([[1, 0], [0, 1]]), None, None, None)  # Fake segmentation mask

        # Mock input parameters
        direct = Path("/mock/parent_directory")
        subdir = Path("/mock/parent_directory/MASKED_subdir_1")
        num = 1
        img = np.ones((10, 10), dtype=np.uint16)  # Dummy image for testing
        channel_number = 1

        # Call the function
        masks = run_cellpose(direct, subdir, num, img, channel_number)

        # Assertions
        mock_Cellpose.assert_called_once_with(model_type='nuclei')  # Check if model was initialized
        mock_model.eval.assert_called_once_with(img, channels=[channel_number, 0], diameter=100.0, flow_threshold=0.5)  # Check eval method call
        
        # Ensure imshow, draw, and savefig were called
        mock_imshow.assert_called()
        mock_draw.assert_called()
        mock_savefig.assert_called_once_with(str(subdir) + "/cellpose_mask1")

        # Check that the returned masks match the expected output
        expected_masks = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_equal(masks, expected_masks)  # Ensure masks match the expected value









