from omegaconf import OmegaConf

import torch
from torch import nn

from ..utils.data import save_json


class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.kwargs = dict(kwargs)

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, "pytorch_model.bin")
        OmegaConf.save(config=self.cfg, f="config.yaml")
        save_json("config.json", self.kwargs)

    def load(self):
        state_dict = torch.load("pytorch_model.bin")
        self.load_state_dict(state_dict)
