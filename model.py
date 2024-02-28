import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics


class LeNet(pl.LightningModule):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 7)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=7
        )

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 256 * 4 * 4)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = F.softmax(self.fc3(x), dim=1)
        # print(x.shape)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        # Log images to TensorBoard
        if batch_idx == 0:  # Log one batch of images per epoch
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            grid = torchvision.utils.make_grid(inputs)
            self.logger.experiment.add_image("input_images", grid, self.current_epoch)
            self.log(
                "train_acc",
                self.accuracy(outputs, torch.argmax(labels, dim=1)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.accuracy(outputs, torch.argmax(labels, dim=1)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


class MiniAlexNet(pl.LightningModule):
    def __init__(self):
        super(MiniAlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(384 * 2 * 2, 384 * 2),
            nn.ReLU(),
            nn.Linear(384 * 2, 384),
            nn.ReLU(),
            nn.Linear(384, 7),
            nn.Softmax(),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=7
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 384 * 2 * 2)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        # Log images to TensorBoard
        if batch_idx == 0:  # Log one batch of images per epoch
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_acc",
                self.accuracy(outputs, torch.argmax(labels, dim=1)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.accuracy(outputs, torch.argmax(labels, dim=1)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
