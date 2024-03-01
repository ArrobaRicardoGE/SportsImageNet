from model import SportsNet
from PIL import Image
import torchvision.transforms as transforms
import torch
import argparse

classes = [
    "Badminton",
    "Cricket",
    "Karate",
    "Soccer",
    "Swimming",
    "Tennis",
    "Wrestling",
]

model = SportsNet.load_from_checkpoint(
    "lightning_logs/version_33/checkpoints/epoch=250-step=24347.ckpt"
)
model = model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img")
    args = parser.parse_args()
    img = Image.open(args.img).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    img = transform(img)
    out = model(img.to(model.device))

    print(out)
    print(torch.argmax(out, 1).item())
    print(classes[torch.argmax(out, 1).item()])
