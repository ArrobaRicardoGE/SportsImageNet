import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

from model import LeNet

# from torchmetrics import Accuracy


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # img = torchvision.io.read_image(f"dataset/train-resized/{self.paths[idx].replace('.jpg', '.png')}")
        # img = cv2.imread(f"dataset/train-resized/{self.paths[idx].replace('.jpg', '.png')}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(f"dataset/train-resized/{self.paths[idx].replace('jpg', 'png')}")
        return self.transform(img), self.labels[idx]


# Data loading and transformations
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

df = pd.read_csv("dataset/train.csv")
print(df.label.values)
lb = LabelBinarizer()
labels = lb.fit_transform(df.label.values).astype('f')
print(labels)
print(lb.classes_)
trainset = ImgDataset(df.image_ID.values[:6000], labels[:6000], transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = ImgDataset(df.image_ID.values[6000:], labels[6000:], transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Creating a PyTorch Lightning model and trainer
model = LeNet()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, trainloader, testloader)
