import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str | Path, batch_size: int = 64, *, seed: int = 420):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.seed = seed

    def prepare_data(self):
        # Download or prepare data here
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Set up datasets for train, val, test here
        if stage in ("fit", "validate"):
            mnist_total = MNIST(self.data_dir, train=True, transform=self.transforms)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_total,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(self.seed),
            )
        elif stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transforms
            )
        elif stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        # Return training dataloader
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # Return validation dataloader
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        # Return test dataloader
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)


class SimplePredictor(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.mlp(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
