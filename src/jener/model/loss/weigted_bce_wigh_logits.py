import torch
from torch import nn


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, active_weight=True, weight=None, pos_weight=None):
        super().__init__()
        self.active_weight = active_weight
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        logits = logits.view(-1, labels.size(-1))
        labels = labels.view(-1, labels.size(-1))

        is_active = labels.sum(dim=-1) > 0
        labels = labels[is_active]
        logits = logits[is_active]

        if not self.active_weight:
            loss_fct = nn.BCEWithLogitsLoss()
            return loss_fct(logits, labels.float())

        weight = (
            None
            if self.weight is None
            else torch.tensor(self.weight, device=logits.device)
        )
        pos_weight = (
            None
            if self.pos_weight is None
            else torch.tensor(self.pos_weight, device=logits.device)
        )
        loss_fct = nn.BCEWithLogitsLoss(
            weight=weight,
            pos_weight=None,
        )
        return loss_fct(logits, labels.float())
