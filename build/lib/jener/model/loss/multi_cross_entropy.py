import torch
from torch import nn

INF = 1e4


class MultiLabelCELoss(nn.Module):
    def forward(self, outputs, labels):
        labels = labels.view(-1, labels.size(-1))
        outputs = outputs.view(*labels.size())

        is_active = labels.sum(dim=-1) > 0
        labels = labels[is_active]
        outputs = outputs[is_active]

        mask = (1 - labels.float()) * -INF

        loss = -torch.logsumexp(outputs + mask, -1) + torch.logsumexp(outputs, -1)
        return loss.mean()
