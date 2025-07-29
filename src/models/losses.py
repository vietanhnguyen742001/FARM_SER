import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CELoss
from typing import Tuple, List
import logging 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_emb, audio_emb, labels):
        batch_size = labels.size(0)
        if batch_size == 1:
            logging.warning("ContrastiveLoss (InfoNCE): Kích thước batch là 1. Loss tương phản sẽ là 0 vì không có mẫu âm tính trong batch.")
            return torch.tensor(0.0, device=text_emb.device)

        text_emb = F.normalize(text_emb, dim=-1)
        audio_emb = F.normalize(audio_emb, dim=-1)

        # Ma trận tương đồng cosine
        logits = torch.matmul(text_emb, audio_emb.T) / self.temperature

        labels_reshaped = labels.view(-1, 1)
        mask = torch.eq(labels_reshaped, labels_reshaped.T).float()
        logits_mask = 1 - torch.eye(batch_size, device=labels.device)
        mask = mask * logits_mask

        log_probs = F.log_softmax(logits, dim=1)
        loss_per_sample = -(mask * log_probs).sum(1) / (mask.sum(1) + 1e-8)
        return loss_per_sample.mean()
    
class BYOLLoss(nn.Module):
    def __init__(self, ce_weight=None, contrastive_weight=1.0, temperature=0.07):
        super(BYOLLoss, self).__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(self, outputs, labels):
        """
        outputs: (logits, text_emb_pooled, audio_emb_pooled)
        labels: ground truth labels
        """
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("Expected outputs to be (logits, text_emb_pooled, audio_emb_pooled)")

        logits, text_feat, audio_feat = outputs

        # CE loss
        ce_loss = self.ce_loss_fn(logits, labels)

        # Normalize for cosine similarity
        text_feat = F.normalize(text_feat, dim=-1)
        audio_feat = F.normalize(audio_feat, dim=-1)

  
        sim = F.cosine_similarity(text_feat, audio_feat, dim=-1) 
        contrastive_loss = (2 - 2 * sim).mean()

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        return total_loss, logits, ce_loss, contrastive_loss
