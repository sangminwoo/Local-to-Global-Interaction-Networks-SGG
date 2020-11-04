#!/bin/bash

read -p "enter gpu: " gpu
read -p "enter mode (sggen=0; sgcls=1; predcls=2; brd=3): " mode

if [ "${mode}" == "sggen" ] || [ "${mode}" == 0 ] ; then
	use_gt_box=False
	use_gt_obj_label=False
	brd=False
elif [ "${mode}" == "sgcls" ] || [ "${mode}" == 1 ] ; then
	use_gt_box=True
	use_gt_obj_label=False
	brd=False
elif [ "${mode}" == "predcls" ] || [ "${mode}" == 2 ] ; then
	use_gt_box=True
	use_gt_obj_label=True
	brd=False
elif [ "${mode}" == "brd" ] || [ "${mode}" == 3 ] ; then
	use_gt_box=True
	use_gt_obj_label=True
	brd=True
fi

# training settings
run="tools/relation_train_net.py"
config="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
predictor="CSIPredictor"
resolution=7
train_img_per_batch=4
test_img_per_batch=2
dtype="float16"
max_iter=50000
val_period=2000
checkpoint_period=2000

# cut
use_cut=False #True
relevance_dim=256
num_pair_proposals=64
# split
use_mask_conv=True
use_coord_conv=False
att_type='self_att' # cbam, self_att
# interact
edge2edge=True
graph_interact_module='gat' # gcn, gat

if [ ${#gpu} > 2 ] ; then
	CUDA_VISIBLE_DEVICES=${gpu} \
	python -m torch.distributed.launch --nproc_per_node=${#gpu} ${run} \
	--config-file ${config} \
	MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${use_gt_box} \
	MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${use_gt_obj_label} \
	DATASETS.BI_REL_DET ${brd} \
	MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ${resolution} \
	MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_CUT ${use_cut} \
	MODEL.ROI_RELATION_HEAD.CSINET.RELEVANCE_DIM  ${relevance_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_PAIR_PROPOSALS  ${num_pair_proposals} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_MASK_CONV ${use_mask_conv} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_COORD_CONV  ${use_coord_conv} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE  ${att_type} \
	MODEL.ROI_RELATION_HEAD.CSINET.EDGE2EDGE  ${edge2edge} \
	MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE  ${graph_interact_module} \
	SOLVER.IMS_PER_BATCH ${train_img_per_batch} \
	TEST.IMS_PER_BATCH ${test_img_per_batch} \
	DTYPE ${dtype} \
	SOLVER.MAX_ITER ${max_iter} \
	SOLVER.VAL_PERIOD ${val_period} \
	SOLVER.CHECKPOINT_PERIOD ${checkpoint_period}
else
	CUDA_VISIBLE_DEVICES=${gpu} \
	python ${run} \
	--config-file ${config} \
	MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${use_gt_box} \
	MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${use_gt_obj_label} \
	DATASETS.BI_REL_DET ${brd} \
	MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ${resolution} \
	MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_CUT ${use_cut} \
	MODEL.ROI_RELATION_HEAD.CSINET.RELEVANCE_DIM  ${relevance_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_PAIR_PROPOSALS  ${num_pair_proposals} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_MASK_CONV ${use_mask_conv} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_COORD_CONV  ${use_coord_conv} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE  ${att_type} \
	MODEL.ROI_RELATION_HEAD.CSINET.EDGE2EDGE  ${edge2edge} \
	MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE  ${graph_interact_module} \
	SOLVER.IMS_PER_BATCH ${train_img_per_batch} \
	TEST.IMS_PER_BATCH ${test_img_per_batch} \
	DTYPE ${dtype} \
	SOLVER.MAX_ITER ${max_iter} \
	SOLVER.VAL_PERIOD ${val_period} \
	SOLVER.CHECKPOINT_PERIOD ${checkpoint_period}
fi