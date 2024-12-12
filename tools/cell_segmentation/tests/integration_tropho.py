"""
This document tests integration between the segmentation_cellpose file, dapi_actin_merge_ARGS file, and the run.sh 
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys
sys.path.append("./../")
import shutil
from segmentation_cellpose import main, run_fiji_macro, subdir_segmentation

class TestImageProcessingPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.fiji_path = "C:/Users/laure/OneDrive/Desktop/Fiji.app/ImageJ-win64.exe"  # Update this with the actual path
        
        # Create mock TIFF files in a subdirectory
        self.mock_subdir = os.path.join(self.test_dir, "MASKED")
        os.makedirs(self.mock_subdir, exist_ok=True)

        for i in range(3):
            mock_tiff_path = os.path.join(self.mock_subdir, f"image_{i+1}.tif")
            with open(mock_tiff_path, "wb") as f:
                f.write(b"Mock TIFF data")  # Writing mock binary data

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    @patch('segmentation_cellpose.run_fiji_macro')
    @patch('segmentation_cellpose.subdir_segmentation')
    def test_pipeline(self, mock_subdir_segmentation, mock_run_fiji_macro):
        """Integration test for the image processing pipeline."""
        # Mock the behavior of run_fiji_macro
        mock_run_fiji_macro.return_value = "Fiji executed successfully."

        # Mock the behavior of subdir_segmentation
        mock_subdir_segmentation.return_value = None

        try:
            # Run the main function with test parameters
            result = main(
                fiji_path=self.fiji_path,
                parent_directory=Path(self.test_dir),
                thresh_min=5500,
                thresh_max=10500,
                channel_number=1
            )

            # Verify the output
            self.assertEqual(result, "Segmentation complete.", "Main function did not return expected message.")

            # Assert that the mocks were called with expected arguments
            mock_run_fiji_macro.assert_called_once_with(self.fiji_path, Path(self.test_dir), 5500, 10500)
            mock_subdir_segmentation.assert_called_once_with(Path(self.test_dir), 1)

        except Exception as e:
            self.fail(f"Test failed with error: {e}")


