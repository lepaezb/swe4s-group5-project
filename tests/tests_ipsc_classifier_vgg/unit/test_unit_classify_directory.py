import pytest
import torch
import numpy as np
from PIL import Image
from tools.ipsc_classifier_vgg.classify_directory import ClassificationDataSet, Net  # Replace with your module's name

class TestUnit:
    """Unit tests for individual components."""

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

    def test_classification_dataset(self, dummy_data):
        """Tests the ClassificationDataSet class."""
        image_dir, labels = dummy_data
        images = [f"image_{i}.png" for i in range(len(labels))]

        dataset = ClassificationDataSet(images, labels, image_dir)

        assert len(dataset) == len(labels), "Dataset length mismatch."
        for i in range(len(dataset)):
            image, label = dataset[i]
            assert isinstance(image, torch.Tensor), "Image is not a tensor."
            assert image.shape[0] == 1, "Image is not grayscale."
            assert isinstance(label, torch.Tensor), "Label is not a tensor."
            assert label.item() in [0, 1], "Label is not binary."

    def test_equalization_in_dataset(self, dummy_data):
        """Tests histogram equalization in the dataset."""
        image_dir, labels = dummy_data
        images = [f"image_{i}.png" for i in range(len(labels))]

        dataset = ClassificationDataSet(images, labels, image_dir)

        for i in range(len(dataset)):
            image, _ = dataset[i]
            image_np = image.numpy().squeeze()
            assert np.max(image_np) <= 1.0, "Image values are not normalized to [0, 1]."

    def test_model_forward_pass(self):
        """Tests the forward pass of the model."""
        model = Net(thickness=4)

        batch_size, channels, height, width = 4, 1, 256, 256
        dummy_input = torch.rand((batch_size, channels, height, width))

        output = model(dummy_input)
        assert output.shape == (batch_size, 1), "Model output shape is incorrect."