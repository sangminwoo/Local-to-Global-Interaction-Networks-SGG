import os
import numpy as np
import torch

def load_glove_model(glove_file):
	print('Loading pre-trained GloVe word embeddings...')
	f = open(glove_file, 'r')
	model = {}
	for line in f:
		splitline = line.split()
		word = splitline[0]
		embedding = np.array([float(val) for val in splitline[1:]])
		model[word] = embedding
	print(f'{len(model)} words loaded!')
	return model

def get_glove_embeddings(data_dir='glove', obj_file='vg_obj.txt', dim=50):
	assert dim in [50, 100, 200, 300]

	if dim == 50:
		glove =  load_glove_model(os.path.join(data_dir, 'glove.6B.50d.txt'))
	elif dim == 100:
		glove = load_glove_model(os.path.join(data_dir, 'glove.6B.100d.txt'))
	elif dim == 200:
		glove = load_glove_model(os.path.join(data_dir, 'glove.6B.200d.txt'))
	elif dim == 300:
		glove = load_glove_model(os.path.join(data_dir, 'glove.6B.300d.txt'))

	f = open(os.path.join(data_dir, obj_file), 'r')
	obj_labels = []
	for obj_label in f:
		obj_labels.append(obj_label[:-1])

	obj_embeddings = {}; obj_embeddings_idx = {}
	for obj_label in obj_labels:
		obj_embeddings[obj_label] = glove[obj_label]
		obj_embeddings_idx[label_to_idx[obj_label]] = glove[obj_label]

	background = np.array([0.] * dim)
	obj_embeddings['background'] = background
	obj_embeddings_idx[0] = background

	sorted_idx = np.sort([k for k, v in obj_embeddings_idx.items()])
	obj_embeddings_list = []
	for idx in sorted_idx:
		obj_embeddings_list.append(torch.FloatTensor(obj_embeddings_idx[idx]))

	obj_embeddings_tensor = torch.stack(obj_embeddings_list, dim=0)

	torch.save(obj_embeddings_tensor, os.path.join(data_dir, f'glove_embeddings_{dim}.pth'))
	return obj_embeddings, obj_embeddings_idx, obj_embeddings_tensor

def load_embeddings(data_dir='glove', dim=50):
	if os.path.exists(os.path.join(data_dir, f'glove_embeddings_{dim}.pth')):
		glove_emb = torch.load(os.path.join(data_dir, f'glove_embeddings_{dim}.pth'))
		print('glove embeddings loaded successfully!')
	else:
		_, _, glove_emb = get_glove_embeddings(dim=dim)

	return glove_emb

label_to_idx = {
"background": 0,
"kite": 69,
"pant": 87,
"bowl": 18,
"laptop": 72,
"paper": 88,
"motorcycle": 80,
"railing": 103,
"chair": 28,
"windshield": 146,
"tire": 130,
"cup": 34,
"bench": 10,
"tail": 127,
"bike": 11,
"board": 13,
"orange": 86,
"hat": 60,
"finger": 46,
"plate": 97,
"woman": 149,
"handle": 59,
"branch": 21,
"food": 49,
"bear": 8,
"vase": 140,
"vegetable": 141,
"giraffe": 52,
"desk": 36,
"lady": 70,
"towel": 132,
"glove": 55,
"bag": 4,
"nose": 84,
"rock": 104,
"guy": 56,
"shoe": 112,
"sneaker": 120,
"fence": 45,
"people": 90,
"house": 65,
"seat": 108,
"hair": 57,
"street": 124,
"roof": 105,
"racket": 102,
"logo": 77,
"girl": 53,
"arm": 3,
"flower": 48,
"leaf": 73,
"clock": 30,
"hill": 63,
"bird": 12,
"umbrella": 139,
"leg": 74,
"screen": 107,
"men": 79,
"sink": 116,
"trunk": 138,
"post": 100,
"sidewalk": 114,
"box": 19,
"boy": 20,
"cow": 33,
"skateboard": 117,
"plane": 95,
"stand": 123,
"pillow": 93,
"ski": 118,
"wire": 148,
"toilet": 131,
"pot": 101,
"sign": 115,
"number": 85,
"pole": 99,
"table": 126,
"boat": 14,
"sheep": 109,
"horse": 64,
"eye": 43,
"sock": 122,
"window": 145,
"vehicle": 142,
"curtain": 35,
"kid": 68,
"banana": 5,
"engine": 42,
"head": 61,
"door": 38,
"bus": 23,
"cabinet": 24,
"glass": 54,
"flag": 47,
"train": 135,
"child": 29,
"ear": 40,
"surfboard": 125,
"room": 106,
"player": 98,
"car": 26,
"cap": 25,
"tree": 136,
"bed": 9,
"cat": 27,
"coat": 31,
"skier": 119,
"zebra": 150,
"fork": 50,
"drawer": 39,
"airplane": 1,
"helmet": 62,
"shirt": 111,
"paw": 89,
"boot": 16,
"snow": 121,
"lamp": 71,
"book": 15,
"animal": 2,
"elephant": 41,
"tile": 129,
"tie": 128,
"beach": 7,
"pizza": 94,
"wheel": 144,
"plant": 96,
"tower": 133,
"mountain": 81,
"track": 134,
"hand": 58,
"fruit": 51,
"mouth": 82,
"letter": 75,
"shelf": 110,
"wave": 143,
"man": 78,
"building": 22,
"short": 113,
"neck": 83,
"phone": 92,
"light": 76,
"counter": 32,
"dog": 37,
"face": 44,
"jacket": 66,
"person": 91,
"truck": 137,
"bottle": 17,
"basket": 6,
"jean": 67,
"wing": 147
}