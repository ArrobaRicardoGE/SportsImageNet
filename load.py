from model import LeNet
from PIL import Image
import torchvision.transforms as transforms
import torch

model = LeNet.load_from_checkpoint(
    "/Users/arroba/Documents/UP/code/ProyectoFinal/lightning_logs/version_15/checkpoints/epoch=99-step=9400.ckpt"
)

model = model.eval()

img = Image.open(
    f"/Users/arroba/Documents/UP/code/ProyectoFinal/dataset/test/00b1df8c4f.jpg"
)

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
