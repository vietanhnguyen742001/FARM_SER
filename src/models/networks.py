import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.base import Config
from .modules import build_audio_encoder, build_text_encoder

class AttentionGuidedPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionGuidedPooling, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = self.attn(x)  
        weights = F.softmax(weights, dim=1)  
        pooled = torch.sum(weights * x, dim=1) 
        return pooled

class RAFM_SER(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(RAFM_SER, self).__init__()
        self.cfg = cfg
        self.device = device

        self.text_encoder = build_text_encoder(cfg.text_encoder_type).to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.audio_encoder = build_audio_encoder(cfg).to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        # Projectors
        self.text_proj = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.audio_proj = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)

        self.residual_attn = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.fusion_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.agp = AttentionGuidedPooling(cfg.fusion_dim)

        # MLP Classifier
        self.linear_layer_output = cfg.linear_layer_output
        prev_dim = cfg.fusion_dim
        for i, dim in enumerate(cfg.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.classifier = nn.Linear(prev_dim, cfg.num_classes)

    def forward(self, input_text, input_audio, output_attentions=False):
        text_emb_seq = self.text_encoder(input_text).last_hidden_state
        text_proj_seq = self.text_proj(text_emb_seq)

        if len(input_audio.size()) != 2:
            B, N = input_audio.size(0), input_audio.size(1)
            audio_emb_seq = self.audio_encoder(input_audio.view(-1, *input_audio.shape[2:])).last_hidden_state
            audio_emb_seq = audio_emb_seq.mean(1).view(B, N, -1)
        else:
            audio_emb_seq = self.audio_encoder(input_audio)
        audio_proj_seq = self.audio_proj(audio_emb_seq)

        attn_out, attn_weights = self.residual_attn(
            text_proj_seq, audio_proj_seq, audio_proj_seq, average_attn_weights=False
        )
        fused = self.norm(text_proj_seq + attn_out)
        fused = self.dropout(fused)

        pooled = self.agp(fused)

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = F.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifier(x)

        return out, self.agp(text_proj_seq), self.agp(audio_proj_seq)

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state

