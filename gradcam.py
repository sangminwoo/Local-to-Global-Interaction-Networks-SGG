import os
import argparse
import cv2
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger
from lib.scene_parser.parser import build_scene_parser, build_scene_parser_optimizer

class GradCAM:
	def __init__(self, model_dict, verbose=False):
		self.model = model_dict['model']
		layer_name = model_dict['layer_name']

		self.graidents = dict()
		self.activations = dict()

		def backward_hook(model, grad_input, grad_output):
			self.gradients['value'] = grad_output[0]
			return None

		def forward_hook(model, input, output):
			self.activations['value'] = output
			return None

		for module in self.model.modules():
			if module.__class__.__name__ == layer_name:
				target_layer = module

		# target_layer = model.module.linear1 # DataParallel

		# forward hook will be called every time after forward() has computed an output.
		target_layer.register_forward_hook(forward_hook)
		# backward hook will be called every time the gradients w.r.t. module inputs are computed.
		target_layer.register_backward_hook(backward_hook)

		if verbose:
		 	try:
		 		input_size = model_dict['input_size']
		 	except KeyError:
		 		print('please specify size of input image in model_dict. e.g. {input_size:(224,224)}')
		 		pass
		 	else:
		 		device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
		 		self.model(torch.zeros(1, 3, *(input_size), device=device))
		 		print('saliency_map size :', self.activations['value'][0].shape)

	def forward(self, input, class_idx=None, retain_graph=False):
		N, C, H, W = input.shape

		box_list, box_pair_list= self.model(input)
		logit = box_list[0].get_field('logits')

		if class_idx is None:
			idxs = logit.max(dim=1)[1]
			score = logit[:, idxs].squeeze()
		else:
			score = logit[:, class_idx].squeeze()

		self.model.zero_grad()
		score.backward(retain_graph=retain_graph)
		gradients = self.gradients['value']
		activations = self.activations['value']
		N, K, U, V = graidents.size()

		alpha = gradients.view(b, k , -1).mean(2)
		weights = alpha.view(N, K, U, V)

		saliency_map = (weights*activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(H, W), mode='bilinear', align_corners=False)
		saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

		return saliency_map, logit

	def __call__(self, input, class_idx=None, retain_graph=False):
		return self.forward(input, class_idx, retain_graph)

class Normalize:
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		return self.do(tensor)

	def normalize(self, tensor, mean, std):
		if not tensor.ndimension() == 4:
			raise TypeError('tensor should be 4D')

		mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
		std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

		return tensor.sub(std).div(mean)

	def denormalize(self, tensor, mean, std):
		if not tensor.ndimension() == 4:
			raise TypeError('tensor should be 4D')

		mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
		std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

		return tensor.mul(std).add(mean)

	def do(self, tensor):
		return self.normalize(tensor, self.mean, self.std)

	def undo(self, tensor):
		return self.denormalize(tensor, self.mean, self.std)

	def __repr__(self):
		return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def visualize_cam(mask, img):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
	heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
	b, g, r = heatmap.split(1)
	heatmap = torch.cat([r, g, b])

	result = heatmap + img.cpu()
	result = result.div(result.max()).squeeze()

	return heatmap, result

def run_gradcam(img_dir, save_dir, model, layer_name, input_size=(None, None)):
	pil_img = PIL.Image.open(img_dir)
	normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
	torch_img = F.interpolate(torch_img, size=input_size, mode='bilinear', align_corners=False)
	normed_torch_img = normalizer(torch_img)

	model.eval()
	model_dict = dict(model=model, layer_name=layer_name, input_size=input_size)
	gradcam = GradCAM(model_dict, True)

	mask, _ = gradcam(normed_torch_img)
	heatmap, result = visualize_cam(mask, torch_img)

	images = torch.stack([torch_img.squeeze().cpu(), heatmap, result], 0)
	images = make_grid(images, nrow=1)

	if not os.path.exists(save_dir):
		os.path.makedirs(save_dir)

	save_image(images, os.path.join(save_dir, 'visualize.jpg'))
	PIL.Image.open(save_dir)

if __name__=='__main__':
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

	# scene_parser = build_scene_parser(cfg); scene_parser.to('cuda')
	# sp_optimizer, sp_scheduler, sp_checkpointer, extra_checkpoint_data = \
 #            build_scene_parser_optimizer(cfg, scene_parser, local_rank=args.local_rank, distributed=args.distributed)
	arguments = {}
	arguments["iteration"] = 0
	model = build_model(cfg, arguments, args.local_rank, args.distributed)
	scene_parser = model.scene_parser

	run_gradcam(
		img_dir='../scene-graph-TF-release/data_tools/VG/images/1.jpg',
		save_dir='gradcam',
		model=scene_parser,
		layer_name='InstanceConvolution',
		input_size=(800, 600))