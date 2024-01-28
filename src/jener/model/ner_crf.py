import torch
from torch.cuda.amp import autocast

from ..model.loss import (
    MultiLabelCELoss,
    WeightedBCEWithLogitsLoss,
    AsymmetricLossOptimized,
    TwoWayLoss,
)
from ..model.ner import NERModel
from ..model.utils import PatialEERCRF


class NERCRFModel(NERModel):
    def __init__(self, cfg, num_labels=3, num_tags=3, set_labels=None, eer={}, **kwargs):
        super().__init__(
            cfg, num_labels=num_labels, num_tags=num_tags, set_labels=set_labels, **kwargs
        )

        self.set_labels = {int(k): v for k,v in set_labels.items()}
        self.label2idx = {v: k for k, v in set_labels.items()}
        self.crf = eval(self.cfg.type.model.crf.cls)(cfg, set_labels=set_labels, eer=eer)

    def convert_logit_to_bin(self, outputs):
        flags = (outputs > 0.5).any(dim=-1).unsqueeze(-1)
        argmax = outputs.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(argmax, num_classes=self.num_tags)

        ene_outputs = torch.where(flags, outputs >= 0.5, one_hot)
        return ene_outputs

    def refine(self, labels, outputs):
        ne_flags = labels["iob2"][..., self.label2idx["O"] + 1 :].any(dim=-1)
        s_flags = labels["iob2"][..., self.label2idx["B-NE"]] == 1
        if "U-NE" in self.label2idx:
            s_flags |= labels["iob2"][..., self.label2idx["U-NE"]] == 1

        mask_iob2_outputs = outputs["iob2"].clone()
        mask_iob2_labels = labels["iob2"].argmax(dim=-1).clone()

        mask_iob2_outputs[~ne_flags] = -100
        mask_iob2_labels[~ne_flags] = -100

        iob2_cover_flags = (mask_iob2_outputs == mask_iob2_labels).all(dim=-1)

        ene_outputs = self.convert_logit_to_bin(outputs["ene"].clone())

        mask_ene_outputs = ene_outputs.clone()
        mask_ene_labels = labels["ene"].clone()

        mask_ene_outputs[~s_flags] = -100
        mask_ene_labels[~s_flags] = -100

        ene_cover_flags = (mask_ene_outputs == mask_ene_labels).all(dim=-1).all(dim=-1)
        cover_flags = iob2_cover_flags & ene_cover_flags

        one_hot_iob2_outputs = torch.nn.functional.one_hot(
            outputs["iob2"], num_classes=self.num_labels
        )
        outputs_ne_flags = one_hot_iob2_outputs[..., self.label2idx["O"] + 1 :].any(
            dim=-1
        )
        ene_outputs[~outputs_ne_flags] = 0

        assert (ene_outputs >= 0).any().item()
        assert (one_hot_iob2_outputs >= 0).any().item()

        return {
            "outputs": {
                "ene": ene_outputs.bool(),
                "iob2": one_hot_iob2_outputs.bool(),
            },
            "flags": cover_flags,
        }

    def forward(self, input_ids, attention_mask, labels={}, do_refine=False):
        losses, outputs = {}, {}

        bsz, seq_len = input_ids.size()

        iob2_logits, ene_logits = self.transformer(input_ids, attention_mask)

        with autocast(enabled=False):
            outputs, losses = self.crf(
                iob2_logits.float(),
                attention_mask,
                labels=labels.get("iob2"),
                outputs=outputs,
                losses=losses,
            )

        if "iob2" in outputs:
            outputs["iob2"] = outputs["iob2"].to(device=iob2_logits.device)

        outputs["ene"] = ene_logits.sigmoid()

        if do_refine:
            return self.refine(labels, outputs)

        if len(labels) > 0:
            loss_fct = WeightedBCEWithLogitsLoss(
                active_weight=self.cfg.loss.class_weight.active,
                weight=self.ene_weight,
                pos_weight=self.ene_pos_weight,
            )
            losses["ene_bce"] = loss_fct(ene_logits, labels["ene"])

            loss_fct = MultiLabelCELoss()
            losses["ene_ce"] = loss_fct(ene_logits, labels["ene"])

            return outputs, losses

        return outputs
