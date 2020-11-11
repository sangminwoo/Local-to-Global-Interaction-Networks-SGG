#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --master_port 10001 \
--nproc_per_node=4 tools/detector_pretrain_net.py \
--config-file "configs/e2e_relation_detector_VGG16_1x.yaml" \
SOLVER.IMS_PER_BATCH 8 \
TEST.IMS_PER_BATCH 4 \
DTYPE "float16" \
SOLVER.MAX_ITER 200000 \
SOLVER.STEPS "(30000, 45000)" \
SOLVER.VAL_PERIOD 5000 \
SOLVER.CHECKPOINT_PERIOD 5000 \
MODEL.RELATION_ON False \
OUTPUT_DIR /home/t2_u1/repo/csi-net/checkpoints/pretrained_faster_rcnn/vgg16_backbone/ \
SOLVER.PRE_VAL False