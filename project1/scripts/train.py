import lightning as L
from lightning.pytorch.loggers import WandbLogger

from project1.modules import MnistClassifier, MnistDataModule

SEED = 420

if __name__ == "__main__":
    mnist_datamodule = MnistDataModule(
        "./data", batch_size=64, num_workers=0, seed=SEED
    )
    wandb_logger = WandbLogger(
        log_model="all", project="uni-mlops", save_dir="./wandb_logs"
    )

    model = MnistClassifier(hidden_layer_sizes=[128, 64], dropout=0.2)

    wandb_logger.watch(model)

    trainer = L.Trainer(logger=wandb_logger, max_epochs=5)
    trainer.fit(model, datamodule=mnist_datamodule)
