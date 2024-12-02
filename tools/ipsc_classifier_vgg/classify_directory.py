import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.color import rgb2gray
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


def resize_images(file_list, directory, target_size=(256, 256)):
    """
    Resize images to the specified size.
    
    Args:
        file_list (list): List of image filenames.
        directory (str): Directory containing the images.
        target_size (tuple): Target size for resizing (width, height).
    
    Returns:
        list: List of resized images as numpy arrays.
    """
    resized_images = []
    for file in file_list:
        image_path = os.path.join(directory, file)
        image = Image.open(image_path).resize(target_size, Image.LANCZOS)
        image_array = np.array(image)
        resized_images.append(image_array)
    return resized_images


def equalize_images(images):
    """
    Apply histogram equalization to a list of images.
    
    Args:
        images (list): List of images as numpy arrays.
    
    Returns:
        list: List of histogram-equalized images.
    """
    return [exposure.equalize_hist(img_as_ubyte(img)) for img in images]


def train_val_split(dataset, labels, test_size=0.2, random_state=1):
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset (list): List of data samples.
        labels (list): Corresponding labels for the dataset.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducibility.
    
    Returns:
        tuple: Dictionaries of training and validation datasets and labels.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        dataset, labels, test_size=test_size, random_state=random_state
    )
    return {"train": np.array(X_train), "val": np.array(X_val)}, {"train": np.array(y_train), "val": np.array(y_val)}


class ClassificationDataSet(Dataset):
    """
    PyTorch Dataset for classification tasks with optional augmentations.
    
    Args:
        inputs (list): List of image filenames.
        labels (list): List of corresponding labels.
        directory (str): Path to the directory containing the images.
        transform (albumentations.Compose, optional): Augmentation pipeline.
        phase (str): Phase of the dataset ('train' or 'val').
    
    Returns:
        Dataset: A PyTorch Dataset object.
    """
    def __init__(self, inputs, labels, directory, transform=None, phase='val'):
        self.inputs = inputs
        self.labels = labels
        self.directory = directory
        self.transform = transform
        self.phase = phase
        self.default_transform = A.Compose([A.CenterCrop(width=256, height=256)])
        if phase == 'train' and not transform:
            self.transform = A.Compose([
                A.RandomCrop(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                ToTensorV2()
            ])
        elif phase == 'val' and not transform:
            self.transform = self.default_transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.inputs[idx])
        image = Image.open(image_path).resize((512, 512), Image.LANCZOS)
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']
        image = rgb2gray(image)
        image = exposure.equalize_hist(img_as_ubyte(image))
        
        image = np.expand_dims(image, axis=2).astype(float)

        label = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        return torch.tensor(image).permute(2, 0, 1).float(), label


def conv3x3(in_channels, out_channels, pool=False, dropout=None):
    """
    Create a 3x3 convolutional block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool (bool): Whether to add a pooling layer.
        dropout (float, optional): Dropout probability.
    
    Returns:
        torch.nn.Sequential: A convolutional block.
    """
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.PReLU(),
        torch.nn.BatchNorm2d(out_channels),
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(kernel_size=2))
    if dropout:
        layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    """
    VGG-like convolutional neural network for binary classification.
    
    Args:
        thickness (int): Base number of filters in the convolutional layers.
    """
    def __init__(self, thickness=4):

        super(Net, self).__init__()
        self.conv1 = conv3x3(in_channels=1, out_channels=thickness)
        self.conv2 = conv3x3(in_channels=thickness, out_channels=thickness, pool=True)

        self.conv3 = conv3x3(in_channels=thickness, out_channels=thickness*2)
        self.conv4 = conv3x3(in_channels=thickness*2, out_channels=thickness*2, pool=True)

        self.conv5 = conv3x3(in_channels=thickness*2, out_channels=thickness*4)
        self.conv6 = conv3x3(in_channels=thickness*4, out_channels=thickness*4, pool=True)

        self.conv7 = conv3x3(in_channels=thickness*4, out_channels=thickness*8)
        self.conv8 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.conv9 = conv3x3(in_channels=thickness*8, out_channels=thickness*8)
        self.conv10 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.conv11 = conv3x3(in_channels=thickness*8, out_channels=thickness*8)
        self.conv12 = conv3x3(in_channels=thickness*8, out_channels=thickness*8, pool=True)

        self.fc1 = torch.nn.Linear(thickness * 8 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)    # 16 256 256
        x = self.conv2(x)          

        x = self.conv3(x)   # 32  128 128
        x = self.conv4(x)   

        x = self.conv5(x)   # 64  64 64
        x = self.conv6(x) 

        x = self.conv7(x)   # 128 32 32
        x = self.conv8(x)   

        x = self.conv9(x) # 128 16 16
        x = self.conv10(x)

        x = self.conv11(x) # 128 8 8
        x = self.conv12(x) # 128 4 4

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        return torch.sigmoid(x)


def classify_images_with_model(model, directory, output_file, threshold=0.5, batch_size=64, num_workers=4):
    """
    Classify images using a trained model and save results.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        directory (str): Directory containing images.
        output_file (str): Path to save classification results.
        threshold (float): Threshold for binary classification.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of DataLoader worker processes.
    """
    images = [file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset = ClassificationDataSet(images, [0] * len(images), directory, phase='val')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            predictions = outputs.cpu().numpy()
            results.extend([
                (images[i], 'good' if pred >= threshold else 'bad', pred)
                for i, pred in enumerate(predictions)
            ])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Filename,Label,Confidence\n")
        for filename, label, confidence in results:
            f.write(f"{filename},{label},{confidence:.4f}\n")
    print(f"Classification results saved to {output_file}")



def main():
    """
    Main entry point for the image classification script.

    This function uses `argparse` to parse command-line arguments for model inference. 
    It loads a pre-trained PyTorch model, processes a directory of images using a custom 
    dataset class, and saves the classification results to a specified CSV file.

    Command-line arguments:
        --model-path (str): Path to the trained PyTorch model file.
        --input-directory (str): Directory containing the images to classify.
        --output-file (str): File path to save the classification results (in CSV format).
        --threshold (float): Confidence threshold for binary classification. Default is 0.5.
        --batch-size (int): Number of images processed per batch. Default is 64.
        --num-workers (int): Number of worker threads for DataLoader. Default is 0.

    Usage Example:
        Run the script with the following command:
        ```
        python classify_directory.py --model-path ./models/Simple_model_best_model_0.92.pt \
                           --input-directory ./model_data/H9p36 \
                           --output-file ./classification_results.csv \
                           --threshold 0.625 \
                           --batch-size 64 \
                           --num-workers 4
        ```

    Raises:
        FileNotFoundError: If the model file specified in `--model-path` does not exist.
        FileNotFoundError: If the input directory specified in `--input-directory` does not exist.
        ValueError: If invalid arguments are passed (handled implicitly by argparse).

    Side Effects:
        - Saves classification results in the specified output CSV file.
        - Prints a confirmation message with the output file path upon completion.
        """
    parser = argparse.ArgumentParser(description="Image Classification Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--input-directory", type=str, required=True, help="Directory containing images to classify.")
    parser.add_argument("--output-file", type=str, required=True, help="File path to save classification results.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for classification (default: 0.5).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader (default: 64).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader (default: 0).")
    
    args = parser.parse_args()

    # Load the trained model
    model = Net(thickness=4)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Run the classification
    classify_images_with_model(
        model=model,
        directory=args.input_directory,
        output_file=args.output_file,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()