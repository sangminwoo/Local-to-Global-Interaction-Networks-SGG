# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN # 800
        max_size = cfg.INPUT.MAX_SIZE_TRAIN # 1024
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN # 0.0
        brightness = cfg.INPUT.BRIGHTNESS # 0.0
        contrast = cfg.INPUT.CONTRAST # 0.0
        saturation = cfg.INPUT.SATURATION # 0.0
        hue = cfg.INPUT.HUE # 0.0
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST # 800
        max_size = cfg.INPUT.MAX_SIZE_TEST # 1024
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255 # true
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255 # (102.9801, 115.9465, 122.7717), (1.0, 1.0, 1.0)
    )
    color_jitter = T.ColorJitter(
        brightness=brightness, # 0.0
        contrast=contrast, # 0.0
        saturation=saturation, # 0.0
        hue=hue, # 0.0
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size), # (800, 1024)
            # T.RandomHorizontalFlip(flip_horizontal_prob), # NOTE: mute this since spatial repations is snesible to this
            # T.RandomVerticalFlip(flip_vertical_prob), # NOTE: mute this since spatial repations is snesible to this
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
