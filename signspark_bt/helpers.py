# coding: utf-8
"""
Collection of helper functions
"""
from __future__ import annotations

import copy
import glob
import os
import os.path
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Any, Callable, Optional, Union
import numpy as np
from pathlib import Path

import torch
from torch import nn, Tensor

from .version import __version__
import yaml
from .data.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    model_dir = os.path.abspath(model_dir)
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                f"Model directory already exists: {model_dir}\n"
                f"Either point training.model_dir somewhere else, remove that "
                f"directory, or set training.overwrite: true in the config "
                f"(which deletes it and starts from scratch)."
            )
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("signspark-bt %s", __version__)
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of the data and vocabularies.
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    # From the index, not train_data[0], which would decode poses from LMDB.
    first = train_data.examples[0]
    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(first.gls), " ".join(first.txt)
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def expand_env_vars(value):
    """Expand ${VAR} and ~ in every string in a nested config."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file, expanding ${VAR} and ~.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return expand_env_vars(cfg)


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu", weights_only=False)
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def init_wandb(cfg: dict, logger: logging.Logger):
    """Start or resume a W&B run. Auth is left to wandb (WANDB_API_KEY).

    Returns the run (None if disabled) and the config, which gains a wandb_id
    so an interrupted run can resume.
    """
    if not cfg.get("wandb", False):
        return None, cfg

    import wandb

    project = cfg.get("wandb_project", "back_translation")
    entity = cfg.get("wandb_entity", os.environ.get("WANDB_ENTITY"))

    wb_id = cfg.get("wandb_id", None)
    if wb_id is not None:
        logger.info("Resuming wandb run %s", wb_id)
        return wandb.init(project=project, entity=entity, id=wb_id, resume="allow"), cfg

    wb_id = wandb.util.generate_id()
    logger.info("Starting wandb run %s (entity: %s)", wb_id, entity)
    wb_writer = wandb.init(
        project=project,
        entity=entity,
        id=wb_id,
        config=cfg,
        group=cfg.get("wandb_group", None),
        name=cfg.get("name", ""),
    )
    cfg["wandb_id"] = wb_id
    return wb_writer, cfg


def change_config(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides, e.g. training.batch_size.

    Each override is appended to the run name so sweeps stay distinguishable.
    """
    name = []
    for k, v in overrides.items():
        keys = k.split('.')
        target = cfg
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = v
        name.append(f'{k}-{v}')
    name = '_'.join(name)
    if name != '':
        cfg['name'] = cfg['name'] + '_' + name
    return cfg


def save_config(config: dict, dir: Union[str, Path]):
    local_config = copy.deepcopy(config)
    # Remove non valid datatypes
    def remove_non_valid_types(d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = remove_non_valid_types(v)
            if not isinstance(v, (float, int, str, list, dict, tuple)) and v is not None:
                d.update({k: str(v)})
        return d
    local_config = remove_non_valid_types(local_config)
    sdir = os.path.join(dir, "config.yaml")
    with open(sdir, "w", encoding="utf-8") as ymlfile:
        yaml.safe_dump(
            local_config,
            ymlfile,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    return str(sdir)
