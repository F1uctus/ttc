"""Shared multilingual encoder with cue, pair and scorer heads."""

import torch
from torch import nn
from transformers import AutoModel


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
    )


class AttributionModel(nn.Module):
    def __init__(self, checkpoint: str, dim: int | None, hidden: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.dim = dim or self.encoder.config.hidden_size
        self.cue_head = nn.Linear(self.dim, 5)  # BIO labels
        self.pair_head = mlp(3 * self.dim, hidden, 1)
        self.scorer_head = mlp(2 * self.dim + 2, hidden, 1)  # candidate and ranker

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    @staticmethod
    def pool(hidden: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        return torch.stack(
            [hidden[s:e].mean(0) if e > s else hidden.mean(0) * 0 for s, e in spans]
        )

    def cue_loss(self, hidden: torch.Tensor, bio: torch.Tensor) -> torch.Tensor:
        logits = self.cue_head(hidden)
        return nn.functional.cross_entropy(logits.flatten(0, 1), bio.flatten())

    def pair_loss(self, hidden, a_spans, b_spans, labels) -> torch.Tensor:
        a, b = self.pool(hidden, a_spans), self.pool(hidden, b_spans)
        x = torch.cat([a, b, a * b], dim=-1)
        return nn.functional.binary_cross_entropy_with_logits(
            self.pair_head(x)[:, 0], labels.float()
        )

    def _scores(self, hidden, replica_range, cand_ranges, extra) -> torch.Tensor:
        r = self.pool(hidden, [replica_range]).expand(len(cand_ranges), -1)
        m = self.pool(hidden, cand_ranges)
        x = torch.cat([r, m, extra], dim=-1)
        return self.scorer_head(x)[:, 0]

    def scorer_loss(self, hidden, replica_range, cand_ranges, extra, labels):
        return nn.functional.binary_cross_entropy_with_logits(
            self._scores(hidden, replica_range, cand_ranges, extra), labels.float()
        )

    def ranker_loss(self, hidden, replica_range, cand_ranges, extra, gold: int):
        scores = self._scores(hidden, replica_range, cand_ranges, extra)
        return nn.functional.cross_entropy(
            scores[None], torch.tensor([gold], device=scores.device)
        )
