import lightning as L
import optuna
import wandb
from lightning.pytorch.loggers import WandbLogger
from optuna import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from project1.config import Settings
from project1.modules import MnistClassifier, MnistDataModule

SEED = 42
SETTINGS = Settings()


def objective(trial: Trial):
    dropout = trial.suggest_float("dropout", 0.0, 1.0)

    hidden_layer_sizes_options = {
        str(sizes): sizes for sizes in [(128, 64), (128, 64, 32), (64, 128, 32)]
    }
    hidden_layer_sizes_key = trial.suggest_categorical(
        "hidden_layer_sizes", list(hidden_layer_sizes_options.keys())
    )
    hidden_layer_sizes = hidden_layer_sizes_options[hidden_layer_sizes_key]

    mnist_datamodule = MnistDataModule(
        SETTINGS.MNIST_DATA_DIR,
        batch_size=64,
        num_workers=0,
        seed=SEED,
    )
    wandb_logger = WandbLogger(
        name=f"trial_{trial.number}",
        log_model=True,
        project=SETTINGS.WANDB_PROJECT,
        save_dir=SETTINGS.WANDB_SAVE_DIR,
        reinit=True,
    )

    model = MnistClassifier(hidden_layer_sizes=hidden_layer_sizes, dropout=dropout)
    pruning_callback = PyTorchLightningPruningCallback(trial=trial, monitor="val/loss")

    trainer = L.Trainer(
        logger=wandb_logger, max_epochs=10, callbacks=[pruning_callback]
    )
    trainer.fit(model, datamodule=mnist_datamodule)

    wandb.finish()

    return trainer.callback_metrics["val/accuracy"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
