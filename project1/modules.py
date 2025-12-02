from collections.abc import Iterable
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST


class MnistDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        *,
        num_workers: int = 0,
        seed: int = 420,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())]
        )
        self.num_workers = num_workers
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
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Return validation dataloader
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # Return test dataloader
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MnistClassifier(L.LightningModule):
    def __init__(self, hidden_layer_sizes: Iterable[int], dropout: float = 0.2):
        super().__init__()

        self.save_hyperparameters(
            {"hidden_layer_sizes": hidden_layer_sizes, "dropout": dropout}
        )

        hidden_layers = []
        input_size = 28 * 28
        for h in hidden_layer_sizes:
            hidden_layers.append(nn.Linear(input_size, h))
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            input_size = h
        self.mlp = nn.Sequential(*hidden_layers, nn.Linear(input_size, 10))

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.mlp(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.mlp(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.mlp(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
