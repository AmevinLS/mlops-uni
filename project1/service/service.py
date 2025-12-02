import bentoml
from PIL.Image import Image
from project1.modules import MnistClassifier
import torch
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer


@bentoml.service()
class MnistService:
    def __init__(self) -> None:
        self.classifier = MnistClassifier.load_from_checkpoint(
            checkpoint_path=R"d:/Documents/University/MSc/Sem2/MLOps/mlops-uni/artifacts/model-hz1r3swr-v0/model.ckpt"
        )
        self.classifier.eval()
        self.trainer = Trainer()

    @bentoml.api()
    def predict_mnist_class(self, img: Image) -> int:
        img_t = pil_to_tensor(img).float()

        predict_dataset = TensorDataset(img_t.flatten().unsqueeze(0))
        predict_dataloader = DataLoader(predict_dataset, batch_size=64, shuffle=False)

        predictions = self.trainer.predict(self.classifier, predict_dataloader)
        return torch.argmax(predictions[0], dim=1).item()
