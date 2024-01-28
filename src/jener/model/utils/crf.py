import torch
import torch.nn as nn
from torch_struct import LinearChainCRF

from typing import *

from ...utils.array import padding

INF = 1e6


class PatialEERCRF(nn.Module):
    def __init__(
        self,
        cfg,
        set_labels={},
        eer={},
        use_transitions=True,
    ):
        super().__init__()
        self.num_labels = len(set_labels)
        self.eer = eer
        self.cfg = cfg

        self.use_transitions = use_transitions
        self.set_labels = {int(k): v for k, v in set_labels.items()}
        self.label2idx = {v: k for k, v in self.set_labels.items()}

        constraints = allowed_transitions(self.cfg.data.encoding, self.set_labels)
        self.transition_constraints = nn.Parameter(
            constraints.float(), requires_grad=False
        )
        self.transition_params = nn.Parameter(0.001 * torch.randn_like(constraints))

    def forward(self, tag_scores, attention_mask, labels=None, outputs={}, losses={}):
        B, N, C = tag_scores.shape

        lengths = attention_mask.long().sum(-1)
        max_length = lengths.max()
        tag_scores = tag_scores[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        if labels is not None:
            labels = labels[:, :max_length]
        lengths += 1  # torch_struct expects n+1 as the size

        local_potentials = tag_scores

        # Convert to batch size, i, c_{i+1}, c_i for torch_struct
        log_phis = self._expand_potentials(tag_scores)

        # print(f"lens: {lengths}")
        crf = LinearChainCRF(log_phis, lengths)

        if not self.training:
            """
            outputs["iob2"] = LinearChainCRF.struct.from_parts(crf.argmax[:, :-1])[
                0
            ]  # need to chop of last dummy node
            """
            outputs["iob2"] = self.decode(tag_scores, attention_mask)
            # for i in range(attention_mask.size(0)):
            #     print(outputs["iob2"][i][attention_mask[i] == 1])

        if labels is not None:
            losses.update(self.loss(labels, local_potentials, crf))

        return outputs, losses

    def decode(self, emissions, mask):
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1).bool()

        pad = torch.zeros_like(mask[0]).unsqueeze(0)
        mask = torch.cat((mask[1:], pad), dim=0)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_labels
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        transitions = (
            self.transition_params.detach().clone()
            + (1 - self.transition_constraints) * -INF
        )

        score = transitions[self.label2idx["START"]].clone() + emissions[1]
        history = []

        for i in range(2, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_emission = emissions[i].unsqueeze(1)

            next_score = broadcast_score + transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += transitions[:, self.label2idx["END"]].clone()
        _, best_last_tags = score.max(dim=1)

        # seq_len, bsz, num_tags
        history = torch.stack(history, dim=0).flip(dims=[0])
        sub_mask = mask[2:seq_length].flip(dims=[0])

        history = torch.where(
            sub_mask.unsqueeze(-1), history, best_last_tags.view(1, -1, 1)
        )

        best_tags_list = [best_last_tags]
        for hist in history:
            best_tags = hist[torch.arange(batch_size), best_tags_list[-1]]
            best_tags_list.append(best_tags)
        best_tags_list.append(torch.zeros_like(best_last_tags))
        best_tags_list = torch.stack(best_tags_list, dim=0).flip(dims=[0])
        return best_tags_list.transpose(0, 1)

    def loss(
        self,
        labels,
        local_potentials,
        pred_crf,
    ):
        outputs = {}

        constrained_pred_potentials = self._expand_potentials(
            self._constrain_potentials(labels, local_potentials),
        )

        constrained_pred_crf = LinearChainCRF(
            constrained_pred_potentials, lengths=pred_crf.lengths
        )
        outputs["constrained_pred_crf"] = constrained_pred_crf

        losses = {}
        losses["ner"] = self._supervised_loss(constrained_pred_crf, pred_crf) / 10
        if self.eer.get("weight", -1.0) > 0:
            losses["eer"] = self._prior_margin_loss(pred_crf)

        return losses

    def _supervised_loss(self, constrained_pred_crf, pred_crf):
        loss = (pred_crf.partition - constrained_pred_crf.partition).mean()
        return loss

    def _prior_margin_loss(self, crf: LinearChainCRF, **kwargs) -> Dict[str, Any]:
        """Compute a margin-based marginal entity tag ratio loss on tagging posterior."""
        B, N, C, _ = crf.log_potentials.shape
        # Note: the sums over full seq lens automatically incorporate length info since crf puts zero mass on pads
        tag_marginals = crf.marginals.sum(2)  # shape: B, N, C  (sum out c_{i+1})
        if self.cfg.type.model.crf.add_se_tag:
            E_entity_counts = tag_marginals[
                :, :, 3:
            ].sum()  # =sum[tag \neq O, START, END]
        else:
            E_entity_counts = tag_marginals[:, :, 1:].sum()  # =sum[tag \neq O]
        EER = E_entity_counts / tag_marginals.sum()  # / B*N

        dist_from_center = (EER - self.eer.get("ratio")).abs()
        margin_loss = self.eer.get("weight") * (
            dist_from_center - self.eer.get("margin")
        ).clamp(min=0)
        return margin_loss

    def _expand_potentials(self, local_potentials):
        B, N, C = local_potentials.shape
        potentials = local_potentials.unsqueeze(2).repeat(1, 1, C, 1)

        transitions = self.transition_params.t()  # flip to c_{i+1}, c_i
        transitions = transitions.reshape(1, 1, C, C).repeat(B, N, 1, 1)
        potentials = potentials + transitions
        return potentials

    def _constrain_potentials(self, labels, local_potentials):
        mask = torch.logical_not(labels).float() * -INF
        return local_potentials + mask

    def _constrain_transitions(self, log_phis, weight=INF):
        B, N, C, _ = log_phis.shape
        ok_transitions = (
            self.transition_constraints.t().reshape(1, 1, C, C).repeat(B, N, 1, 1)
        )
        log_phis = log_phis + -weight * (1 - ok_transitions)
        return log_phis


def allowed_transitions(
    constraint_type: str,
    labels: Dict[int, str],
    include_start_stop: bool = False,
    as_tensor: bool = True,
) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items())
    if include_start_stop:
        labels_with_boundaries.extend([(start_tag, "START"), (end_tag, "END")])

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(
                constraint_type, from_tag, from_entity, to_tag, to_entity
            ):
                allowed.append((from_label_index, to_label_index))

    if as_tensor:
        allowed_tensor = torch.zeros(num_labels, num_labels)
        for i, j in allowed:
            allowed_tensor[i, j] = 1.0
        allowed = allowed_tensor

    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    from_entity: ``str``, required
        The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    to_entity: ``str``, required
        The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("B", "I", "O", "L", "U")
        if to_tag == "END":
            return from_tag in ("B", "I", "O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I")
                and to_tag in ("I", "L")
                and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag == "B" and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise TypeError(f"Unknown constraint type: {constraint_type}")
