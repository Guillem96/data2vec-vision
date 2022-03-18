from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len: int, dim: int) -> None:
        super().__init__()
        pos_weights = torch.empty(1, seq_len, dim)
        nn.init.normal_(pos_weights, std=0.02)
        self.pos_embedding = nn.Parameter(pos_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class Data2Vec(nn.Module):

    def __init__(self,
                 in_features: int,
                 seq_len: int,
                 dim: int = 768,
                 nhead: int = 8,
                 dropout: float = 0.2,
                 n_encoders: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.dim = dim
        self.dropout = dropout
        self.nhead = nhead
        self.n_encoders = n_encoders

        el = nn.TransformerEncoderLayer(dim,
                                        nhead=nhead,
                                        dim_feedforward=dim * 4,
                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(el, n_encoders)
        self.patch_linear = nn.Linear(self.in_features, dim)
        self.pos_embedding = PositionalEmbedding1D(seq_len, dim)

        self.regressor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.hs: List[torch.Tensor] = []
        for l in self.encoder.layers:
            l.dropout2.register_forward_hook(self._register_hs_hook)

    def _register_hs_hook(self, layer: nn.Module, inputs: torch.Tensor,
                          outputs: torch.Tensor) -> None:
        self.hs.append(outputs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hs.clear()
        x = self.patch_linear(x)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        return self.regressor(x), torch.stack([h.clone() for h in self.hs])

    def save(self, fpath: Union[Path, str]) -> None:
        checkpoint = {
            "config": {
                "in_features": self.in_features,
                "seq_len": self.seq_len,
                "dim": self.dim,
                "nhead": self.nhead,
                "n_encoders": self.n_encoders,
            },
            "weights": self.state_dict(),
        }
        torch.save(checkpoint, fpath)

    @classmethod
    def from_pretrained(
            cls,
            fpath: Union[Path, str],
            map_location: Optional[torch.device] = None) -> "Data2Vec":
        checkpoint = torch.load(fpath, map_location=map_location)
        model = cls(**checkpoint.pop("config"))
        model.load_state_dict(checkpoint.pop("weights"))
        model.eval()
        return model
