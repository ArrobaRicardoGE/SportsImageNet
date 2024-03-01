from model import SportsNet
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchmetrics import ConfusionMatrix
import torchmetrics
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")
lb = LabelBinarizer()
labels = lb.fit_transform(df.label.values).astype("f")
print(lb.classes_)

model = SportsNet.load_from_checkpoint(
    "lightning_logs/version_33/checkpoints/epoch=250-step=24347.ckpt"
)
model = model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

_, X_val, _, y_val = train_test_split(
    df.image_ID.values, labels, test_size=0.25, random_state=62, stratify=labels
)

y_val = torch.tensor(y_val).to(model.device)

imgs = []
for name in X_val:
    path = f"dataset/train/{name}"
    img = Image.open(path).convert("RGB")
    img = transform(img)
    imgs += [img.unsqueeze(0)]

imgs = torch.cat(imgs, dim=0)
preds = model(imgs.to(model.device))

preds_int = torch.argmax(preds, dim=1)
labels_int = torch.argmax(y_val, dim=1)

print(
    "Accuracy:",
    torchmetrics.functional.accuracy(
        preds_int, labels_int, "multiclass", num_classes=7
    ),
)
print(
    "F1:",
    torchmetrics.functional.f1_score(
        preds_int, labels_int, "multiclass", num_classes=7
    ),
)
print(
    "Precision:",
    torchmetrics.functional.precision(
        preds_int, labels_int, "multiclass", num_classes=7
    ),
)

cm = ConfusionMatrix(task="multiclass", num_classes=7).to(model.device)
cm.update(preds_int, labels_int)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

cm.plot(labels=list(lb.classes_), ax=ax)
# plt.show()

fig.savefig('conf.png')