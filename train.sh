#!/bin/bash

read -p "enter gpu: " gpu
read -p "enter mode (sggen=0; sgcls=1; predcls=2; brd=3): " mode
num_gpu=${gpu//[^0-9]}

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
train_img_per_batch=1
test_img_per_batch=2
dtype="float16"
max_iter=50000
val_period=2000
checkpoint_period=50000
random_seed=0

# preset
use_bias=False # True, False
pool_sbj_obj=True # True, False
use_masking=False # True, False
# cut
use_cut=False #True, False
relevance_dim=256
num_pair_proposals=256
# split
reduce_dim=False # True, False
use_att=True # True, False
att_all=True # True, False
att_type='non_local' # awa, cbam, self_att, non_local
flatten=True # True, False
# interact
use_gin=True # True, False
gin_layers=1 # 1, 2, 4
edge2edge=False # True, False
graph_interact_module='gat' # gcn, gat, again

if [ ${#num_gpu} > 1 ] ; then # multi-gpu training
	CUDA_VISIBLE_DEVICES=${gpu} \
	python -m torch.distributed.launch --nproc_per_node=${#num_gpu} ${run} \
	--config-file ${config} \
	MODEL.RANDOM_SEED ${random_seed} \
	MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${use_gt_box} \
	MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${use_gt_obj_label} \
	DATASETS.BI_REL_DET ${brd} \
	MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ${resolution} \
	MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
	MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS ${use_bias} \
	MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ ${pool_sbj_obj} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_CUT ${use_cut} \
	MODEL.ROI_RELATION_HEAD.CSINET.RELEVANCE_DIM  ${relevance_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_PAIR_PROPOSALS  ${num_pair_proposals} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING ${use_masking} \
	MODEL.ROI_RELATION_HEAD.CSINET.REDUCE_DIM ${reduce_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_ATT  ${use_att} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE ${att_all} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE  ${att_type} \
	MODEL.ROI_RELATION_HEAD.CSINET.FLATTEN ${flatten} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_GIN ${use_gin} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_GIN_LAYERS ${gin_layers} \
	MODEL.ROI_RELATION_HEAD.CSINET.EDGE2EDGE  ${edge2edge} \
	MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE  ${graph_interact_module} \
	SOLVER.IMS_PER_BATCH ${train_img_per_batch} \
	TEST.IMS_PER_BATCH ${test_img_per_batch} \
	DTYPE ${dtype} \
	SOLVER.MAX_ITER ${max_iter} \
	SOLVER.VAL_PERIOD ${val_period} \
	SOLVER.CHECKPOINT_PERIOD ${checkpoint_period}
else # single-gpu training
	CUDA_VISIBLE_DEVICES=${gpu} \
	python ${run} \
	--config-file ${config} \
	MODEL.RANDOM_SEED ${random_seed} \
	MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${use_gt_box} \
	MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${use_gt_obj_label} \
	DATASETS.BI_REL_DET ${brd} \
	MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ${resolution} \
	MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
	MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS ${use_bias} \
	MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ ${pool_sbj_obj} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_CUT ${use_cut} \
	MODEL.ROI_RELATION_HEAD.CSINET.RELEVANCE_DIM  ${relevance_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_PAIR_PROPOSALS  ${num_pair_proposals} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING ${use_masking} \
	MODEL.ROI_RELATION_HEAD.CSINET.REDUCE_DIM ${reduce_dim} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_ATT  ${use_att} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE ${att_all} \
	MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE  ${att_type} \
	MODEL.ROI_RELATION_HEAD.CSINET.FLATTEN ${flatten} \
	MODEL.ROI_RELATION_HEAD.CSINET.USE_GIN ${use_gin} \
	MODEL.ROI_RELATION_HEAD.CSINET.NUM_GIN_LAYERS ${gin_layers} \
	MODEL.ROI_RELATION_HEAD.CSINET.EDGE2EDGE  ${edge2edge} \
	MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE  ${graph_interact_module} \
	SOLVER.IMS_PER_BATCH ${train_img_per_batch} \
	TEST.IMS_PER_BATCH ${test_img_per_batch} \
	DTYPE ${dtype} \
	SOLVER.MAX_ITER ${max_iter} \
	SOLVER.VAL_PERIOD ${val_period} \
	SOLVER.CHECKPOINT_PERIOD ${checkpoint_period}
fi