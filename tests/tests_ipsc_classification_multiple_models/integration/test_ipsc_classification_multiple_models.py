import os
import torch
import pytest
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tools.ipsc_classification_multiple_models.ipsc_classification_multiple_models import (
    set_seed,
    load_images,
    ColonyDataset,
    get_model,
    train_model,
    evaluate_model,
    save_examples,
    test_transforms,
    train_transforms
)

@pytest.fixture
def temporary_image_dir_for_integration(tmp_path):
    # Arrange: Create synthetic 'good' and 'bad' images in a temporary directory.
    # We'll create a small set of images to simulate loading and training.
    
    good_img = np.zeros((224, 224, 3), dtype=np.uint8)  # Black image for 'good'
    bad_img = np.ones((224, 224, 3), dtype=np.uint8)*255  # White image for 'bad'

    # Create a few good and bad images
    for i in range(3):
        Image.fromarray(good_img).save(os.path.join(tmp_path, f"train_good_{i}.png"))
        Image.fromarray(bad_img).save(os.path.join(tmp_path, f"train_bad_{i}.jpg"))

    return tmp_path

class TestIntegration:
    def test_full_pipeline(self, temporary_image_dir_for_integration):
        # --- Arrange ---
        seed = 42
        set_seed(seed)
        device = torch.device("cpu")  # For test simplicity, use CPU
        image_dir = str(temporary_image_dir_for_integration)

        # Load images and labels
        image_paths, labels = load_images(image_dir)

        # Split data into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.5, random_state=seed
        )

        # Create datasets and dataloaders
        train_dataset = ColonyDataset(train_paths, train_labels, transform=train_transforms)
        val_dataset = ColonyDataset(val_paths, val_labels, transform=test_transforms)

        batch_size = 2
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        }

        # Initialize a small model (e.g., resnet18) for quick training
        model_name = 'resnet18'
        model = get_model(model_name).to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # --- Act ---
        # Train the model for a few epochs to ensure pipeline runs
        num_epochs = 2  # very small, just for integration testing
        trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, _, _ = train_model(
            trained_model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

        # Evaluate the model on the validation set
        all_inputs, all_labels, all_predictions = evaluate_model(trained_model, dataloaders['val'], criterion)

        # Save examples
        results_dir = os.path.join(image_dir, "results")
        save_examples(all_inputs, all_labels, predictions=all_predictions, num_samples=2, save_dir=results_dir, prefix='test_integration')

        # --- Assert ---
        # Check that we have some training histories recorded
        assert len(train_loss_history) == num_epochs
        assert len(val_loss_history) == num_epochs
        assert len(train_acc_history) == num_epochs
        assert len(val_acc_history) == num_epochs

        # Check that model predictions were generated
        assert len(all_inputs) == len(all_labels) == len(all_predictions)
        assert len(all_inputs) > 0

        # Check that the images were saved
        saved_files = os.listdir(results_dir)
        assert len(saved_files) > 0
        for f in saved_files:
            assert f.startswith('test_integration')
            assert f.endswith('.png')

        # If we reach here, the entire integration pipeline worked without errors.