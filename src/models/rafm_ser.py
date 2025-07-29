import torch
import torch.nn as nn
from configs.base import Config
from .modules import build_audio_encoder, build_text_encoder

class RAFM_SER(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(RAFM_SER, self).__init__()
        self.cfg = cfg
        self.device = device

        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

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

        self.linear_layer_output = cfg.linear_layer_output
        prev_dim = cfg.fusion_dim
        for i, dim in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.classifier = nn.Linear(prev_dim, cfg.num_classes)

    def forward(self, input_text, input_audio, output_attentions=False):
        text_emb = self.text_encoder(input_text).last_hidden_state
        text_proj = self.text_proj(text_emb)

        if len(input_audio.size()) != 2:
            B, N = input_audio.size(0), input_audio.size(1)
            audio_emb = self.audio_encoder(input_audio.view(-1, *input_audio.shape[2:])).last_hidden_state
            audio_emb = audio_emb.mean(1).view(B, N, -1)
        else:
            audio_emb = self.audio_encoder(input_audio)
        audio_proj = self.audio_proj(audio_emb)

        attn_out, attn_weights = self.residual_attn(text_proj, audio_proj, audio_proj, average_attn_weights=False)
        fused = self.norm(text_proj + attn_out)
        fused = self.dropout(fused)

        pooled = fused[:, 0, :] if self.cfg.fusion_head_output_type == "cls" else \
                 fused.mean(1) if self.cfg.fusion_head_output_type == "mean" else \
                 fused.max(1)[0] if self.cfg.fusion_head_output_type == "max" else \
                 fused.min(1)[0]

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        out = self.classifier(self.dropout(x))

        if output_attentions:
            return [out, pooled], [attn_weights]
        return out, pooled, text_proj, audio_proj
