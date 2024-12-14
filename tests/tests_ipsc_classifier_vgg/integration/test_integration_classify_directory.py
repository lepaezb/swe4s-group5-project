import pytest
import numpy as np
import torch
import os
from PIL import Image
from tools.ipsc_classifier_vgg.classify_directory import classify_images_with_model, Net, main  # Replace with your module's name

class TestIntegration:
    """Integration tests for combined components."""

    @pytest.fixture
    def dummy_data(self, tmp_path):
        """Creates dummy image data for testing."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        # Create dummy images
        num_images = 5
        for i in range(num_images):
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(image).save(image_dir / f"image_{i}.png")

        labels = [1 if i % 2 == 0 else 0 for i in range(num_images)]  # Binary labels
        return image_dir, labels

    @pytest.fixture
    def dummy_model(self):
        """Creates a dummy model for testing."""
        return Net(thickness=4)

    def test_classify_images_with_model(self, dummy_data, dummy_model, tmp_path):
        """Tests the classify_images_with_model function."""
        image_dir, _ = dummy_data
        model = dummy_model
        output_file = tmp_path / "output.csv"

        classify_images_with_model(
            model,
            image_dir,
            output_file,
            threshold=0.5,
            batch_size=2,
            num_workers=0,
        )

        assert output_file.exists(), "Output file was not created."
        with open(output_file, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "Output file does not contain results."
            assert "Filename,Label,Confidence" in lines[0], "Header is incorrect."

    def test_main_function(self, monkeypatch, tmp_path, dummy_data, dummy_model):
        """Tests the main function with mocked arguments."""
        image_dir, _ = dummy_data
        model_path = tmp_path / "dummy_model.pt"
        output_file = tmp_path / "output.csv"

        torch.save(dummy_model.state_dict(), model_path)

        monkeypatch.setattr(
            "sys.argv",
            [
                "script_name",
                "--model-path",
                str(model_path),
                "--input-directory",
                str(image_dir),
                "--output-file",
                str(output_file),
                "--threshold",
                "0.5",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
            ],
        )

        main()

        assert output_file.exists(), "Output file was not created by main function."
        with open(output_file, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "Output file does not contain results."
            assert "Filename,Label,Confidence" in lines[0], "Header is incorrect."