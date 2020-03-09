import os
import pprint
import argparse
import numpy as np
import torch
import datetime

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger
from lib.config import cfg
from lib.data.build import build_data_loader
from lib.data.evaluation import evaluate, evaluate_sg
from lib.data.evaluation.sg.sg_eval import do_sg_evaluation

parser = argparse.ArgumentParser(description="Graph Reasoning Machine for Visual Question Answering")
parser.add_argument("--config-file", default="configs/sgg_res101_step.yaml") # baseline_res101.yaml, faster_rcnn_res101.yaml, sgg_res101_joint.yaml, sgg_res101_step.yaml
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--session", type=int, default=0)
parser.add_argument("--resume", type=int, default=0)
parser.add_argument("--batchsize", type=int, default=0)
parser.add_argument("--inference", action='store_true')
parser.add_argument("--instance", type=int, default=-1)
parser.add_argument("--use_freq_prior", action='store_true')
parser.add_argument("--visualize", action='store_true')
parser.add_argument("--algorithm", type=str, default='sg_grcnn') # sg_baseline, sg_imp, sg_msdn, sg_grcnn, sg_reldn
args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

cfg.merge_from_file(args.config_file)
cfg.resume = args.resume
cfg.instance = args.instance
cfg.inference = args.inference
cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
cfg.MODEL.ALGORITHM = args.algorithm
if args.batchsize > 0:
    cfg.DATASET.TRAIN_BATCH_SIZE = args.batchsize
if args.session > 0:
    cfg.MODEL.SESSION = str(args.session)   
# cfg.freeze()

if not os.path.exists("logs") and get_rank() == 0:
    os.mkdir("logs")
logger = setup_logger("scene_graph_generation", "logs", get_rank(),
    filename="{}_{}.txt".format(args.algorithm, get_timestamp()))
logger.info(args)
logger.info("Loaded configuration file {}".format(args.config_file))
output_config_path = os.path.join("logs", 'config.yml')
logger.info("Saving config into: {}".format(output_config_path))
save_config(cfg, output_config_path)

# if not args.inference:
#     model = train(cfg, args)
# else:
#     test(cfg, args)

output_folder = "results"
data_loader_test = build_data_loader(cfg, split="test", is_distributed=False)
predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
predictions_pred = torch.load(os.path.join(output_folder, "predictions_pred.pth"))
extra_args = dict(
    box_only=False,
    iou_types=("bbox",),
    expected_results=[],
    expected_results_sigma_tol=4,
)
multiple_preds = True # False
evaluate_sg(dataset=data_loader_test.dataset,
            predictions=predictions,
            predictions_pred=predictions_pred,
            output_folder=output_folder,
            multiple_preds=multiple_preds,
            **extra_args)