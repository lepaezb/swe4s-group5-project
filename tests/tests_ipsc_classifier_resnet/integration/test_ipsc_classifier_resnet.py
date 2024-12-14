import os
import pytest
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tools.ipsc_classifier_resnet.ipsc_classifier_resnet50 import (
    set_seed,
    load_images,
    ColonyDataset,
    iPSCClassifier,
    train_transforms,
    test_transforms,
    train_model,
    evaluate_model,
    save_examples
)


@pytest.fixture
def integration_image_dir(tmp_path):
    # Arrange: Create a few synthetic 'good' and 'bad' images to simulate the dataset
    # We'll create a small set of images for a minimal integration test

    good_img = np.zeros((224, 224, 3), dtype=np.uint8)  # All black for 'good'
    bad_img = np.ones((224, 224, 3), dtype=np.uint8)*255  # All white for 'bad'

    for i in range(2):
        Image.fromarray(good_img).save(os.path.join(tmp_path, f"test_good_{i}.png"))
        Image.fromarray(bad_img).save(os.path.join(tmp_path, f"test_bad_{i}.jpg"))

    return tmp_path


class TestIntegration:
    def test_full_pipeline(self, integration_image_dir):
        # --- Arrange ---
        seed = 42
        set_seed(seed)
        device = torch.device("cpu")  # Use CPU for testing

        # Load images
        image_paths, labels = load_images(str(integration_image_dir))
        assert len(image_paths) == 4
        assert len(labels) == 4

        # Split into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.5, random_state=seed
        )

        # Create datasets
        train_dataset = ColonyDataset(train_paths, train_labels, transform=train_transforms)
        val_dataset = ColonyDataset(val_paths, val_labels, transform=test_transforms)

        # Create dataloaders
        batch_size = 2
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        }

        # Initialize model, criterion, optimizer
        model = iPSCClassifier().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # --- Act ---
        # Train model for a small number of epochs to ensure pipeline runs
        num_epochs = 2
        model, train_loss, val_loss, train_acc, val_acc, train_prec, val_prec = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=num_epochs
        )

        # Evaluate the model
        all_inputs, all_labels, all_predictions = evaluate_model(model, dataloaders['val'], criterion)

        # Save examples (we will just do it in a temporary directory)
        results_dir = os.path.join(str(integration_image_dir), "results")
        save_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=2, save_dir=results_dir, prefix='integration_test')

        # --- Assert ---
        # Check training histories are recorded
        assert len(train_loss) == num_epochs
        assert len(val_loss) == num_epochs
        assert len(train_acc) == num_epochs
        assert len(val_acc) == num_epochs
        assert len(train_prec) == num_epochs
        assert len(val_prec) == num_epochs

        # Check we got predictions
        assert len(all_inputs) == len(val_dataset)
        assert len(all_labels) == len(val_dataset)
        assert len(all_predictions) == len(val_dataset)

        # Check that images were saved
        saved_files = os.listdir(results_dir)
        assert len(saved_files) == 2
        for f in saved_files:
            assert f.startswith('integration_test')
            assert f.endswith('.png')