from project1.modules import MnistDataModule
from project1.config import Settings
from torchvision.transforms.functional import to_pil_image
from pathlib import Path

if __name__ == "__main__":
    SETTINGS = Settings()
    mnist_datamodule = MnistDataModule(
        data_dir=SETTINGS.MNIST_DATA_DIR, batch_size=5, num_workers=0
    )
    mnist_datamodule.setup(stage="predict")

    imgs_dir = Path(".") / "predict_images"
    imgs_dir.mkdir(exist_ok=True)
    for batch in mnist_datamodule.predict_dataloader():
        images, _ = batch
        for i, img in enumerate(images):
            img = img.resize(28, 28)
            img_pil = to_pil_image(img)
            img_path = imgs_dir / f"image_{i}.png"
            img_pil.save(img_path)
            print(f"Saved image to {img_path}")
        break
