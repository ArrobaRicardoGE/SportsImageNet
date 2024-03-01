from model import SportsNet
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")
lb = LabelBinarizer()
labels = lb.fit_transform(df.label.values).astype("f")
print(lb.classes_)

model = SportsNet.load_from_checkpoint(
    "lightning_logs/version_30/checkpoints/epoch=219-step=20680.ckpt"
)
model = model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

labels = torch.tensor(labels[6000:]).to(model.device)

m = 10
for name, label in zip(df.image_ID.values[6000:], labels):
    path = f'dataset/train/{name}'
    img = Image.open(path).convert('RGB')
    img = transform(img)
    pred = model(img.to(model.device))

    ypred = torch.argmax(pred, 1).item()
    y = torch.argmax(label).item()
    if y != ypred:
        print(f'Mismatch in {name}, predicted {lb.classes_[ypred]}, should be {lb.classes_[y]}')
        m -= 1
    if not m:
        break