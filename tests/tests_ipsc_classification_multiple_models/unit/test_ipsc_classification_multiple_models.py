import os
import shutil
import random
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from tools.ipsc_classification_multiple_models.ipsc_classification_multiple_models import (
    set_seed,
    load_images,
    ColonyDataset,
    get_model,
    train_model,
    evaluate_model,
    save_examples
)

@pytest.fixture
def temporary_image_dir(tmp_path):
    # Arrange: Set up a temporary directory with test images
    good_img = np.zeros((224, 224, 3), dtype=np.uint8)  # dummy black image
    bad_img = np.ones((224, 224, 3), dtype=np.uint8)*255  # dummy white image

    good_filename = "test_good_image.png"
    bad_filename = "test_bad_image.jpg"
    unknown_filename = "test_unknown_image.png"

    good_path = os.path.join(tmp_path, good_filename)
    bad_path = os.path.join(tmp_path, bad_filename)
    unknown_path = os.path.join(tmp_path, unknown_filename)

    Image.fromarray(good_img).save(good_path)
    Image.fromarray(bad_img).save(bad_path)
    Image.fromarray(good_img).save(unknown_path) # doesn't contain 'good' or 'bad' in filename

    return tmp_path


class TestSetSeed:
    def test_set_seed_consistency(self):
        # Arrange
        seed = 42

        # Act
        set_seed(seed)
        random_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Reset seeds and generate again to check consistency
        set_seed(seed)
        random_val2 = random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        # Assert
        assert random_val == random_val2
        assert np_val == np_val2
        assert torch_val == torch_val2


class TestLoadImages:
    def test_load_images_with_known_files(self, temporary_image_dir):
        # Arrange: temporary_image_dir has a good, a bad, and an unknown image
        # Act
        image_paths, labels = load_images(str(temporary_image_dir))

        # Assert
        # Expect: 'test_good_image.png' -> label 0, 'test_bad_image.jpg' -> label 1
        # 'test_unknown_image.png' should be skipped
        assert len(image_paths) == 2
        assert len(labels) == 2
        assert all([os.path.exists(p) for p in image_paths])
        assert set(labels) == {0, 1}

    def test_load_images_with_no_images(self, tmp_path):
        # Arrange: empty directory
        # Act
        image_paths, labels = load_images(str(tmp_path))

        # Assert
        assert len(image_paths) == 0
        assert len(labels) == 0

    def test_load_images_with_max_images(self, temporary_image_dir):
        # Arrange/Act: limit max_images to 1
        image_paths, labels = load_images(str(temporary_image_dir), max_images=1)

        # Assert: only one image should be loaded
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
        transform = transforms.ToTensor()
        dataset = ColonyDataset(image_paths, labels, transform=transform)

        # Act
        image, label = dataset[0]

        # Assert
        # The returned image should be a tensor of shape (3, H, W)
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3
        assert label in [0, 1]


class TestGetModel:
    @pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    def test_get_model_resnet_variants(self, model_name):
        # Arrange/Act
        model = get_model(model_name)

        # Assert: Model should be a resnet variant and final fc layer should have out_features=1
        assert hasattr(model, 'fc')
        assert model.fc.out_features == 1

    @pytest.mark.parametrize("model_name", ["vgg16", "alexnet"])
    def test_get_model_vgg_alexnet(self, model_name):
        # Arrange/Act
        model = get_model(model_name)

        # Assert: Model should have classifier[6] as a Linear layer with out_features=1
        assert hasattr(model, 'classifier')
        assert model.classifier[6].out_features == 1

    def test_get_model_unknown(self):
        # Arrange/Act/Assert
        with pytest.raises(ValueError):
            get_model("unknown_model")


class TestTrainModel:
    def test_train_model_runs(self):
        # Arrange
        # Create a small dataset and a simple model for sanity check
        set_seed(42)
        dummy_images = torch.randn(10, 3, 224, 224)
        dummy_labels = torch.tensor([0,1]*5)
        dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        dataloaders = {
            'train': DataLoader(dataset, batch_size=2, shuffle=True),
            'val': DataLoader(dataset, batch_size=2, shuffle=False)
        }

        model = get_model('resnet18')
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Act
        trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, _, _ = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=1
        )

        # Assert
        # We just check that the training loop runs and produces some results
        assert len(train_loss_history) == 1
        assert len(val_loss_history) == 1
        assert len(train_acc_history) == 1
        assert len(val_acc_history) == 1
        assert isinstance(trained_model, torch.nn.Module)


class TestEvaluateModel:
    def test_evaluate_model_runs(self):
        # Arrange
        set_seed(42)
        dummy_images = torch.randn(6, 3, 224, 224)
        dummy_labels = torch.tensor([0,1,0,1,0,1])
        dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=2)
        model = get_model('resnet18')
        criterion = torch.nn.BCEWithLogitsLoss()

        # Act
        all_inputs, all_labels, all_predictions = evaluate_model(model, dataloader, criterion)

        # Assert
        assert len(all_inputs) == len(dataset)
        assert len(all_labels) == len(dataset)
        assert len(all_predictions) == len(dataset)


class TestSaveExamples:
    def test_save_examples(self, tmp_path):
        # Arrange
        dummy_images = torch.randn(5, 3, 224, 224)
        dummy_labels = np.array([0, 1, 0, 1, 0])
        save_dir = os.path.join(tmp_path, 'results')

        # Act
        save_examples(dummy_images, dummy_labels, predictions=None, num_samples=3, save_dir=save_dir, prefix='example')

        # Assert
        saved_files = os.listdir(save_dir)
        assert len(saved_files) == 3
        for f in saved_files:
            assert f.startswith('example')
            assert f.endswith('.png')