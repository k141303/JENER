from omegaconf import OmegaConf

import torch
from torch import nn
from torch.cuda.amp import autocast

from liat_ml_roberta import RoBERTaConfig
from transformers import RobertaModel

from ..utils.data import save_json

from ..model.loss import (
    MultiLabelCELoss,
    WeightedBCEWithLogitsLoss,
    AsymmetricLossOptimized,
)
from ..model.base import BaseModel
from ..model.utils import PatialEERCRF

class NERModel(BaseModel):
    def __init__(self, cfg, num_labels=3, ene_tags=[], ene_count=None, **kwargs):
        super().__init__(cfg, num_labels=num_labels, ene_tags=ene_tags, **kwargs)

        self.cfg = cfg
        self.num_labels = num_labels
        self.ene_tags = ene_tags
        self.num_tags = len(ene_tags)

        self.config = RoBERTaConfig.from_pretrained(self.cfg.model.bert.name)
        self.bert = RobertaModel(self.config)

        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.iob2_layer = nn.Linear(self.config.hidden_size // 2, self.config.hidden_size)
        self.iob2_output_layer = nn.Linear(self.config.hidden_size, self.num_labels)

        self.ene_layer = nn.Linear(self.config.hidden_size // 2, self.config.hidden_size)
        self.ene_output_layer = nn.Linear(self.config.hidden_size, self.num_tags)

        self.ene_weight = None
        self.ene_pos_weight = None
        if ene_count is not None:
            self.ene_weight = []
            self.ene_pos_weight = []
            max_ene = max(ene_count[ene] for ene in self.ene_tags)
            for ene in self.ene_tags:
                self.ene_weight.append(max_ene / ene_count.get(ene, max_ene))
                self.ene_pos_weight.append(
                    ene_count["total"] / ene_count.get(ene, ene_count["total"])
                )

    def transformer(self, input_ids, attention_mask):
        hidden_state = self.bert(
            input_ids, attention_mask=attention_mask.long()
        ).last_hidden_state
        hidden_state = self.tanh(hidden_state)
        hidden_state = self.dropout(hidden_state)

        iob2_hidden_state = self.dropout(
            self.tanh(self.iob2_layer(hidden_state[..., : self.config.hidden_size // 2]))
        )
        iob2_logits = self.iob2_output_layer(iob2_hidden_state)

        ene_hidden_state = self.dropout(
            self.tanh(self.ene_layer(hidden_state[..., self.config.hidden_size // 2 :]))
        )
        ene_logits = self.ene_output_layer(ene_hidden_state)

        return iob2_logits, ene_logits

    def forward(self, input_ids, attention_mask, labels=None):
        losses, outputs = {}, {}

        iob2_logits, ene_logits = self.transformer(input_ids, attention_mask)

        outputs["iob2"] = iob2_logits.argmax(dim=-1)

        outputs["ene"] = ene_logits.sigmoid()

        if labels is not None:
            iob2_logits = iob2_logits.view(-1, 3)
            iob2_labels = labels["iob2"].view(-1)

            iob2_logits = iob2_logits[iob2_labels >= 0]
            iob2_labels = iob2_labels[iob2_labels >= 0]

            loss_fct = nn.CrossEntropyLoss()
            losses["iob2"] = loss_fct(iob2_logits, iob2_labels)

            loss_fct = MultiLabelCELoss()
            losses["ene"] = loss_fct(ene_logits, labels["ene"])

            loss_fct = WeightedBCEWithLogitsLoss()
            losses["ene"] += loss_fct(ene_logits, labels["ene"])
            losses["ene"] /= 2

            return outputs, losses

        return outputs
