import pandas
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from tqdm import tqdm
import os

photos_x10, labels_x10 = [], []
photos_x40, labels_x40 = [], []

# directory = os.path.join('/content/drive/MyDrive' ,'H9p36/H9p36')
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

len(photos_x40)

from PIL import Image
from skimage.color import rgba2rgb

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
plt.imshow(resize_photos_x40[-1])


# ## Histogram equalization

from skimage.util import img_as_ubyte
from skimage import exposure

equalize_photos_x40 = []

for i in resize_photos_x40:
  img = img_as_ubyte(i)
  img_rescale = exposure.equalize_hist(img)
  equalize_photos_x40.append(img_rescale)


# ## Split data into training and validation sets

from sklearn.model_selection import train_test_split

def train_val_split(dataset, labels):
  X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=1)

  X = {"train":np.array(X_train), "val":np.array(X_val)}
  Labels = {"train":np.array(y_train), "val":np.array(y_val)}
  
  return X, Labels

X, Labels = train_val_split(photos_x40, labels_x40)
#X, Labels = train_val_split(equalize_photos_x40, labels_x40)


# ## Wrap data by DataLoader

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import exposure


from skimage.io import imread

import albumentations as A

class ClassificationDataSet(Dataset):
    def __init__(self, inputs: list, labels, transform=None, phase='train'):
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

dataloaders = {
    'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val': torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
# Define the model directory
model_directory = "./models"

# ## CNN VGG13 Implementation

def conv3x3(in_channels, out_channels, pool=False, dropout=None):
    layers = [
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
    ]

    if pool:
        layers.append(nn.MaxPool2d((2, 2)))

    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class Net(nn.Module):
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

        self.fc1 = nn.Linear(thickness * 8 * 4 * 4, 1)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda
net = Net(4)
net = net.to(device=device)
print(net)


from torch.optim.lr_scheduler import ReduceLROnPlateau
'''
optim.SGD → implemets stochastic gradient descent
net.parameters() → gets CNN parameters
lr → learning rate of gradient descent
momentum → speedup gradient vectors in needed direction
'''
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min')

model_directory = './models'

def write_metrics_in_file(epochs_train, epochs_valid, epoch_valid_metrics):
  print(epochs_train)
  print()
  print(epochs_valid)
  print()
  print(epoch_valid_metrics)
  
  with open(os.path.join(model_directory, "epochs_valid_metrics.txt"), 'a') as file:
    file.write(f"{epoch_valid_metrics['Accuracy'][-1]} {epoch_valid_metrics['Precision'][-1]} {epoch_valid_metrics['Recall'][-1]} {epoch_valid_metrics['F1'][-1]} \n")
  
  with open(os.path.join(model_directory, "epochs_valid.txt"), 'a') as f:
    f.write(f"{epochs_valid[-1]} \n")
  with open(os.path.join(model_directory, "epochs_train.txt"), 'a') as f:
    f.write(f"{epochs_train[-1]} \n")
  
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from torch.nn.functional import normalize

epochs = 80
border = 0.7
best_F_score = 0
phases = ['train', 'val']

epochs_valid = []
epoch_valid_metrics = defaultdict(list)
epochs_train = []
for epoch in range(epochs):  # multiple walk through dataset
    loss_train = []
    loss_valid = []
    for phase in phases:
      running_loss = 0.0
      prediction = []
      label = [] 

      for i, data in tqdm(enumerate(dataloaders[phase]), total = len(dataloaders[phase])):
          # get input data as list [inputs, labels]
          inputs, labels = data
          print(inputs.shape)
          # raise Exception
          #img = inputs[0].permute(1,2,0).numpy()
          #plt.imshow(img.astype(int))
          #plt.show()
          #inputs = normalize(inputs, p=2.0, dim = 0)
          #img = 255 * inputs[0].permute(1,2,0).numpy()
          #plt.imshow(img.astype(int))
          #plt.show()
          #raise Exception

          inputs, labels = inputs.to(device), labels.to(device)

          # zero gradient parameters
          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == "train"):
            outputs = net(inputs.float())
            loss = criterion(outputs.flatten(), labels)
            if phase == 'train':
              loss.backward()
              optimizer.step()
              loss_train.append(loss.item())
            if phase == 'val':
              loss_valid.append(loss.item())
              res = outputs.detach().cpu().numpy() 
              res = res > border
              res  = res.astype(int)
              res = res.squeeze()
              prediction.extend(res)
              labels = labels.detach().cpu().numpy()
              label.extend(labels)
            if i % 5 == 0:
              tqdm.write(f"Epochs[{epoch+1}/{epochs}], phase:{phase}")
            running_loss = 0.0

      if phase == 'train':
        epochs_train.append(np.mean(loss_train))


      if phase == "val":
          epochs_valid.append(np.mean(loss_valid))


          conf_mtrx = confusion_matrix(label, prediction)
          accuracy = (conf_mtrx[0, 0] + conf_mtrx[1, 1]) / (conf_mtrx[0, 0] + conf_mtrx[1, 1] + conf_mtrx[0, 1] + conf_mtrx[1, 0])
          precision = (conf_mtrx[0, 0]) / (conf_mtrx[0, 0] + conf_mtrx[0, 1])
          recall = (conf_mtrx[0, 0]) / (conf_mtrx[0, 0] + conf_mtrx[1, 0])
          F_1 = 2 * (precision * recall) / (precision + recall)
          epoch_valid_metrics["Accuracy"].append(accuracy)
          epoch_valid_metrics["Precision"].append(precision)
          epoch_valid_metrics["Recall"].append(recall)
          epoch_valid_metrics["F1"].append(F_1)
          tqdm.write(f"Epochs[{epoch+1}/{epochs}], Accuracy {accuracy},  Precision {precision}, Recall {recall}, F1 {F_1} ")

          loss_valid = []
          if best_F_score < epoch_valid_metrics["F1"][-1]:
            best_conf = np.copy(confusion_matrix)
            best_epoch = epoch + 1
            best_F_score = epoch_valid_metrics["F1"][-1]
            torch.save(net.state_dict(), os.path.join(model_directory, f"{'Simple_model'}_best_model_{round(best_F_score, 2)}.pt"))

    torch.save(net.state_dict(), os.path.join(model_directory, f"{'Simple_model'}_last_model.pt"))
    write_metrics_in_file(epochs_train, epochs_valid, epoch_valid_metrics)
