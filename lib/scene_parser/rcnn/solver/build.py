# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR # 0.005
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = cfg.SOLVER.WEIGHT_DECAY # 0.0005
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR # 0.01 = 0.005 * 2
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS # 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM) # 0.9
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS, # 70000, 90000
        cfg.SOLVER.GAMMA, # 0.1
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR, # 0.33333
        warmup_iters=cfg.SOLVER.WARMUP_ITERS, # 500
        warmup_method=cfg.SOLVER.WARMUP_METHOD, # linear
    )
