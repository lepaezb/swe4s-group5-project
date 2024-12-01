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

photos_x10, labels_x10 = [], []
photos_x40, labels_x40 = [], []

directory = './model_data/H9p36'

files = os.listdir(directory)
for file in files:
  filename = file.split('.')
  if filename[1] == 'png':
    picture = filename[0].split('_')
    if len(picture) != 1:
      picture_scale = picture[0].split('x')
      if len(picture_scale) == 1:
        photos_x10.append(picture_scale[0])
        labels_x10.append(picture[1])
      else:
        photos_x40.append(file)
        labels_x40.append(picture[1])

def resize(dataset):
  resize_images = []
  for data in dataset:
    image = Image.open(os.path.join(directory, data))
    image = image.resize((256, 256), Image.LANCZOS)
    image = np.array(image)

    if image.shape != (256, 256, 3):
      image = rgba2rgb(image)

    resize_images.append(image)
  return resize_images

resize_photos_x40 = resize(photos_x40)

from skimage.util import img_as_ubyte
from skimage import exposure

equalize_photos_x40 = []

for i in resize_photos_x40:
  img = img_as_ubyte(i)
  img_rescale = exposure.equalize_hist(img)
  equalize_photos_x40.append(img_rescale)

from sklearn.model_selection import train_test_split
def train_val_split(dataset, labels):
  X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=1)

  X = {"train":np.array(X_train), "val":np.array(X_val)}
  Labels = {"train":np.array(y_train), "val":np.array(y_val)}
  
  return X, Labels

X, Labels = train_val_split(photos_x40, labels_x40)

# Classification Dataset
class ClassificationDataSet(Dataset):
    def __init__(self, inputs: list, labels, transform=None, phase='val'):
        transform1 = A.Compose([
          A.RandomCrop(width=256, height=256),
          A.HorizontalFlip(p=0.5),
          A.Transpose(p=0.5),
          A.VerticalFlip(p=0.5),
        ])
        transform2 = A.Compose([
           A.CenterCrop(width=256, height=256)                     
        ])
        self.inputs = inputs
        self.labels = labels
        self.transform = None
        self.inputs_dtype = torch.float
        self.targets_dtype = torch.float
        self.phase = phase
        if self.phase == 'train':
          self.transform =  transform1
        if self.phase == 'val':
          self.transform =  transform2

    def augmentation(self, x):
      return self.transform(image = x)["image"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        input_ID = self.inputs[index]
        x = Image.open(os.path.join(directory, input_ID))   

        if self.transform:
          x = x.resize((512, 512), Image.LANCZOS)
          #x = x.resize((256, 256), Image.ANTIALIAS)
          x = np.array(x)

          x  = self.augmentation(x)
        else:
          x = x.resize((256, 256), Image.LANCZOS)
          x = np.array(x)


        img = img_as_ubyte(x)
        x = exposure.equalize_hist(img)
        x = rgb2gray(x)
        x = np.expand_dims(x, axis=2)
        x = x.astype(float)

        y = self.labels[index]
        y = torch.from_numpy(np.array(y).astype(float))
        y = y.type(torch.float)

        return torch.from_numpy(x).type(torch.float).permute(2,0,1), y

batch_size = 64
num_workers = 0

for key in Labels:
  Labels[key][Labels[key]=='good'] = 1
  Labels[key][Labels[key]=='bad'] = 0

trainset = ClassificationDataSet(X['train'], Labels['train'], transform=True, phase='train')
valset = ClassificationDataSet(X['val'], Labels['val'], transform=True, phase='val')


class Net(torch.nn.Module):
    ''' VGG13 convolutional neural network'''
	
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
    
    # Define the model architecture
def conv3x3(in_channels, out_channels, pool=False, dropout=None):
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        torch.nn.PReLU(),
        torch.nn.BatchNorm2d(out_channels),
    ]

    if pool:
        layers.append(torch.nn.MaxPool2d((2, 2)))

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return torch.nn.Sequential(*layers)


# Classification Function
def classify_images_with_model(model, directory, output_file, threshold=0.625, batch_size=64, num_workers=0):
    """
    Classify images in a directory using a trained model.
    Args:
      - model: Trained PyTorch model.
      - directory: Directory containing images to classify.
      - output_file: Path to save classification results.
      - threshold: Confidence threshold for binary classification.
      - batch_size: Number of images per batch.
      - num_workers: Number of worker threads for DataLoader.
    """
    # Collect all image files
    images = [file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    labels = [0] * len(images)  # Dummy labels (not used for inference)

    # Prepare dataset and DataLoader
    dataset = ClassificationDataSet(images, labels, directory, phase='val')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Prepare for inference
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images).squeeze()
            predictions = torch.sigmoid(outputs).cpu().numpy()

            for i, pred in enumerate(predictions):
                label = 'good' if pred >= threshold else 'bad'
                results.append((images[i], label, pred))

    # Save results to a file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Filename,Label,Confidence\n")
        for filename, label, confidence in results:
            f.write(f"{filename},{label},{confidence:.4f}\n")
    print(f"Results saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    # Load your trained model
    model_path = './models/Simple_model_best_model_0.92.pt'
    model = Net(thickness=4)  # Replace with your model class
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Directory of images to classify
    input_directory = './model_data/H9p36'
    output_file = './classification_results3.csv'

    classify_images_with_model(model, input_directory, output_file)