import os
from omegaconf import OmegaConf

import torch

from ..utils.data import load_json

from ..model.ner import NERModel
from ..model.ner_crf import NERCRFModel


def load_finetuned_model(file_dir):
    cfg = OmegaConf.load(os.path.join(file_dir, "config.yaml"))

    config = load_json(os.path.join(file_dir, "config.json"))
    model = eval(cfg.type.model.cls)(cfg, **config)

    if torch.cuda.is_available():
        state_dict = torch.load(os.path.join(file_dir, "pytorch_model.bin"))
    else:
        state_dict = torch.load(os.path.join(file_dir, "pytorch_model.bin"), torch.device('cpu')) 
    model.load_state_dict(state_dict)
    return cfg, model
