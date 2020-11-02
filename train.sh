#!/usr/bin/env bash

max_iter=50000

CUDA_VISIBLE_DEVICES=0 \
python tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR CSIPredictor \
SOLVER.IMS_PER_BATCH 4 \
TEST.IMS_PER_BATCH 2 \
DTYPE "float16" \
SOLVER.MAX_ITER 50000 \
SOLVER.VAL_PERIOD 2000 \
SOLVER.CHECKPOINT_PERIOD 2000

# CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR CSIPredictor SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000