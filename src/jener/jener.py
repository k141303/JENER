import re
import math
import argparse
import warnings
import pkg_resources

from collections import defaultdict

import torch

from omegaconf import OmegaConf

from liat_ml_roberta import RoBERTaTokenizer

from .model import load_finetuned_model
from .utils.array import slice, padding
from .utils.data import load_ene2name

warnings.simplefilter("ignore")

class JENER(object):
    def __init__(self, model_dir:str=None, ene_def_path:str=None, seq_len:int=512, dup_len:int=32, deactive_cuda:bool=False) -> None:
        self.seq_len = seq_len
        self.dup_len = dup_len
        self.deactive_cuda = deactive_cuda
        if model_dir is None:
            model_dir = pkg_resources.resource_filename("jener", "data/model/jener_v1")

        self.cfg, self.model = load_finetuned_model(model_dir)
        self.model.eval()

        self.tokenizer = RoBERTaTokenizer.from_pretrained(self.cfg.model.bert.name)

        if ene_def_path is None:
            ene_def_path = pkg_resources.resource_filename("jener", "data/ENE_Definition_v9.0.0-with-attributes-20220714.jsonl")
        self.ene2name = load_ene2name(ene_def_path)

    def __call__(self, text: str) -> list:
        tokens, offsets = self.tokenize(text)

        spans = slice(len(tokens), self.seq_len-2, self.dup_len)

        outputs = {"iob2": torch.LongTensor(), "ene": torch.Tensor()}
        for i, (s, e) in enumerate(spans):
            input_tokens = [self.tokenizer.cls_token] + tokens[s:e] + [self.tokenizer.sep_token]
            attention_mask = [1] * len(input_tokens)

            input_tokens = padding(input_tokens, self.seq_len, self.tokenizer.pad_token)
            attention_mask = padding(attention_mask, self.seq_len, 0)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            input_ids = torch.LongTensor([input_ids])
            attention_mask = torch.BoolTensor([attention_mask])

            if not self.deactive_cuda and torch.cuda.is_available():
                model = model.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            with torch.no_grad():
                outputs_ = self.model(input_ids, attention_mask)

            for key, value in outputs_.items():
                value = value.squeeze(0)
                value = value[:e-s+2][1:-1]
                if i != 0:
                    value = value[:, math.ceil(self.dup_len):]
                if i != len(spans)-1:
                    value = value[:, :-math.floor(self.dup_len)]

                outputs[key] = torch.cat([outputs[key], value.cpu()], dim=0)

        pred_offsets = self.iob2offset(outputs["iob2"].tolist())

        predicts = []
        for s, e in sorted(pred_offsets, key=lambda x: x[0]):
            s_, e_ = offsets[s][0], offsets[e-1][1]

            ene_ids = self.convert_vec_to_ene_ids(outputs["ene"][s])
            enes = [self.ene2name[ene_id] for ene_id in ene_ids]
            d = {"surf": text[s_:e_], "span": (s_, e_), "ENEs": enes}
            predicts.append(d)

        return predicts

    def tokenize(self, text: str) -> (list, list):
        space_shift = {}
        for m in re.finditer("\s+", text):
            space_shift[m.span(0)[0]] = m.span(0)[1] - m.span(0)[0]

        tokens = self.tokenizer.tokenize(text)

        rm_head = lambda token: re.sub("@@$", "", token)
        orig_tokens = list(map(rm_head, tokens))

        offsets = []
        for token in orig_tokens:
            if len(offsets) == 0:
                s = 0
            else:
                s = offsets[-1][-1]
            s += space_shift.get(s, 0)
            offsets.append([s, s + len(token)])

        return tokens, offsets

    def convert_vec_to_ene_ids(self, vector: torch.Tensor) -> set:
        if (vector >= 0.5).any():
            vector = vector >= 0.5
            ene_ids = set(
                self.model.ene_tags[idx] for idx, flag in enumerate(vector.tolist()) if flag
            )
        else:
            ene_idx = vector.argmax(dim=-1).item()
            ene_ids = {self.model.ene_tags[ene_idx]}
        return ene_ids

    def iob2offset(self, iob2_outputs: list) -> set:
        offsets = set()
        start = None
        iob2_tags = [self.model.set_labels[tag] for tag in iob2_outputs]
        if self.cfg.data.encoding in ["BIO"]:
            for i, tag in enumerate(iob2_tags):
                if start is not None and tag != "I-NE":
                    offsets.add((start, i))
                    start = None
                if tag == "B-NE":
                    start = i
            if start is not None:
                offsets.add((start, len(iob2_outputs)))
        elif self.cfg.data.encoding in ["BIOUL"]:
            for i, tag in enumerate(iob2_tags):
                if tag == "I-NE":
                    continue
                if start is not None:
                    if tag == "L-NE":
                        offsets.add((start, i + 1))
                    start = None
                if tag == "U-NE":
                    offsets.add((i, i + 1))
                if tag == "B-NE":
                    start = i
        else:
            raise NotImplementedError()

        return offsets