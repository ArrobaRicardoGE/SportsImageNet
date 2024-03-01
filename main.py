import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image

from model import SportsNet


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(f"dataset/train/{self.paths[idx]}").convert("RGB")
        return self.transform(img), self.labels[idx]


# Data loading and transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

df = pd.read_csv("dataset/train.csv")
print(df.label.values)
lb = LabelBinarizer()
labels = lb.fit_transform(df.label.values).astype("f")
print(labels)
print(lb.classes_)

X_train, X_val, y_train, y_val = train_test_split(
    df.image_ID.values, labels, test_size=0.25, random_state=62, stratify=labels
)

trainset = ImgDataset(X_train, y_train, transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = ImgDataset(X_val, y_val, transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Creating a PyTorch Lightning model and trainer
model = SportsNet()
trainer = pl.Trainer(max_epochs=250)
trainer.fit(
    model,
    trainloader,
    testloader,
)
