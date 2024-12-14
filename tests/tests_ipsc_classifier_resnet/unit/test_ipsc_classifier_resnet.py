import os
import pytest
import random
import numpy as np
import torch
from unittest.mock import patch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from tools.ipsc_classifier_resnet.ipsc_classifier_resnet50 import (
    set_seed,
    load_images,
    ColonyDataset,
    iPSCClassifier,
    train_model,
    evaluate_model,
    save_examples,
    test_transforms,
    train_transforms
)


@pytest.fixture
def temporary_image_dir(tmp_path):
    # Arrange: Create a temporary directory with some test images
    good_img = np.zeros((224, 224, 3), dtype=np.uint8)  # Black image for "good"
    bad_img = np.ones((224, 224, 3), dtype=np.uint8)*255  # White image for "bad"
    unknown_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    good_path = tmp_path / "test_good_image.png"
    bad_path = tmp_path / "test_bad_image.jpg"
    unknown_path = tmp_path / "test_unknown.png"

    Image.fromarray(good_img).save(good_path)
    Image.fromarray(bad_img).save(bad_path)
    Image.fromarray(unknown_img).save(unknown_path)

    return tmp_path


class TestSetSeed:
    def test_set_seed_deterministic(self):
        # Arrange
        seed = 1234
        set_seed(seed)
        
        # Act
        r1 = random.random()
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()

        # Reset seeds and generate again
        set_seed(seed)
        r2 = random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        # Assert
        assert r1 == r2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2


class TestLoadImages:
    def test_load_images_with_good_and_bad(self, temporary_image_dir):
        # Act
        image_paths, labels = load_images(str(temporary_image_dir))

        # Assert: 'test_good_image.png' -> label 0, 'test_bad_image.jpg' -> label 1
        # 'test_unknown.png' is skipped
        assert len(image_paths) == 2
        assert len(labels) == 2
        assert 0 in labels and 1 in labels

    def test_load_images_with_max_images(self, temporary_image_dir):
        # Act
        image_paths, labels = load_images(str(temporary_image_dir), max_images=1)

        # Assert
        assert len(image_paths) == 1
        assert len(labels) == 1


class TestColonyDataset:
    def test_colony_dataset_length(self, temporary_image_dir):
        # Arrange
        image_paths, labels = load_images(str(temporary_image_dir))
        dataset = ColonyDataset(image_paths, labels, transform=None)

        # Act
        length = len(dataset)

        # Assert
        assert length == len(image_paths)

    def test_colony_dataset_getitem(self, temporary_image_dir):
        # Arrange
        image_paths, labels = load_images(str(temporary_image_dir))
        dataset = ColonyDataset(image_paths, labels, transform=transforms.ToTensor())

        # Act
        image, label = dataset[0]

        # Assert
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3
        assert label in [0, 1]


class TestiPSCClassifier:
    def test_ipsc_classifier_output(self):
        # Arrange
        model = iPSCClassifier()
        input_data = torch.randn(2, 3, 224, 224)  # Batch of size 2

        # Act
        output = model(input_data)

        # Assert
        # Should output a tensor of shape (2,1) for binary classification
        assert output.shape == (2, 1)


class TestTrainModel:
    def test_train_model_single_epoch(self):
        # Arrange
        model = iPSCClassifier()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create a small synthetic dataset
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        dataloaders = {
            'train': DataLoader(dataset, batch_size=2, shuffle=True),
            'val': DataLoader(dataset, batch_size=2, shuffle=False)
        }

        # Act
        trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, _, _ = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=1
        )

        # Assert
        assert len(train_loss_history) == 1
        assert len(val_loss_history) == 1
        assert len(train_acc_history) == 1
        assert len(val_acc_history) == 1
        assert isinstance(trained_model, torch.nn.Module)


class TestEvaluateModel:
    def test_evaluate_model_runs(self):
        # Arrange
        model = iPSCClassifier()
        criterion = torch.nn.BCEWithLogitsLoss()

        # Small dataset
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        # Act
        all_inputs, all_labels, all_predictions = evaluate_model(model, dataloader, criterion)

        # Assert
        assert len(all_inputs) == 4
        assert len(all_labels) == 4
        assert len(all_predictions) == 4


class TestSaveExamples:
    def test_save_examples(self, tmp_path):
        # Arrange
        dummy_images = torch.randn(5, 3, 224, 224)
        dummy_labels = np.array([0,1,0,1,0])
        save_dir = tmp_path / 'results'

        # Act
        save_examples(dummy_images, dummy_labels, predictions=None, num_samples=3, save_dir=str(save_dir), prefix='test_example')

        # Assert
        files = os.listdir(save_dir)
        assert len(files) == 3
        for f in files:
            assert f.startswith('test_example')
            assert f.endswith('.png')