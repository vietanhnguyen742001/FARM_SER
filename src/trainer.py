import torch
from torch import Tensor
from typing import Dict
from configs.base import Config
from models.networks import RAFM_SER
from utils.torch.trainer import TorchTrainer
from models.losses import BYOLLoss


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: RAFM_SER,
        criterion: BYOLLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad()

        input_text, input_audio, label = batch
        input_text = input_text.to(self.device)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)

        output = self.network(input_text, input_audio)

        total_loss, logits, ce_loss, contrastive_loss = self.criterion(output, label)

        total_loss.backward()
        self.optimizer.step()

        _, preds = torch.max(logits, 1)
        accuracy = torch.mean((preds == label).float())

        return {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "acc": accuracy.item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.network.eval()
        input_text, input_audio, label = batch
        input_text = input_text.to(self.device)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            output = self.network(input_text, input_audio)
            total_loss, logits, ce_loss, contrastive_loss = self.criterion(output, label)
            _, preds = torch.max(logits, 1)
            accuracy = torch.mean((preds == label).float())

        return {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "acc": accuracy.item(),
        }

