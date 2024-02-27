# Resize a 256x256
import cv2
import os

cnt = 0
for file in os.listdir("dataset/train"):
    cnt += 1
    if cnt % 10 == 0:
        print(cnt)
    path = f"dataset/train/{file}"
    img = cv2.imread(path)
    resized = cv2.resize(img, (256, 256))
    status = cv2.imwrite(f"dataset/train-resized/{file[:-4]}.png", resized)

for file in os.listdir("dataset/test"):
    cnt += 1
    if cnt % 10 == 0:
        print(cnt)
    path = f"dataset/test/{file}"
    img = cv2.imread(path)
    resized = cv2.resize(img, (256, 256))
    cv2.imwrite(f"dataset/test-resized/{file[:-4]}.png", resized)
