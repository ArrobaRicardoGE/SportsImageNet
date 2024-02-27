from model import LeNet
from PIL import Image
import torchvision.transforms as transforms
import torch

model = LeNet.load_from_checkpoint(
    "/Users/arroba/Documents/UP/code/ProyectoFinal/lightning_logs/version_36/checkpoints/epoch=25-step=2444.ckpt"
)

model = model.eval()

img = Image.open(f"dataset/test-resized/c4fbe789fc.png")

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
print(torch.max(out, 1))
