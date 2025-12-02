from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MNIST_DATA_DIR: str = "./data"
    WANDB_PROJECT: str = "uni-mlops"
    WANDB_SAVE_DIR: str = "./wandb_logs"
