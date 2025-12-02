import wandb
from project1.config import Settings
from pathlib import Path

if __name__ == "__main__":
    SETTINGS = Settings()

    api = wandb.Api()
    artifact = api.artifact(
        "nikita-makarevich-poznan-university-of-technology/uni-mlops/model-hz1r3swr:v0"
    )
    artifact_dir = Path(artifact.download())
    ckpt_path = artifact_dir / "model.ckpt"

    print(f"{ckpt_path = }")
