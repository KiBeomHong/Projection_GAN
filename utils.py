import torchtext.vocab as vocab
import os, csv, sys, gzip, torch, time
import torch.nn as nn
import numpy as np
import scipy.misc
from torch.utils.data import Dataset, DataLoader
import skvideo.io
import pdb
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class UCFsports(Dataset):
	def __init__(self, root_dir, transform=None, T=16):
		self.images = []
		self.texts = []
		self.embedding = []
		self.root_dir = root_dir
		self.transform = transform
		self.T = T

		small = False

		if not small:
			self.action_list = ['Diving', 'Golf', 'SkateBoarding']
		else:
			self.action_list = ['Diving', 'Golf', 'SkateBoarding']



		print('Loading UCFsports metadata...')
		sys.stdout.flush()
		time_start = time.time()
		self.images = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_dir)) for f in files if f.endswith('jpg')])
		
		self.action_map = {}
		for i, action in enumerate(self.action_list):
			self.action_map[action] = i

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image_path = self.images[idx]

		action = os.path.dirname(image_path).split('/')[2]
		
		#text_path = image_path.replace('.jpg','.txt')
		#embedding_path = image_path.replace('.jpg','.npy')

		text_path = os.path.join(self.root_dir,action,'features','0.txt')
		embedding_path = os.path.join(self.root_dir,action,'features','0.npy')

		img = Image.open(image_path).convert('RGB')
		if self.transform:
			img = self.transform(img)

		embedding = torch.Tensor(np.load(embedding_path))
		text = open(text_path).read()

		label = self.action_map[action]
		

		return img, embedding, text, label


def loss_plot(hist, path='.', model_name = 'model', y_max = None):
	try:
		x = range(len(hist['D_loss']))
	except:
		keys = hist.keys()
		lens = [ len(hist[k]) for k in keys if 'loss' in k ]
		maxlen = max(lens)
		x = range(maxlen)

	plt.xlabel('Iter')
	plt.ylabel('Loss')
	plt.tight_layout()

	for key,value in hist.items():
		y = value
		plt.plot(x, y, label=key)

	plt.legend(loc=1)
	plt.grid(True)

	path = os.path.join(path, model_name+'_loss.png')

	plt.savefig(path)

	plt.close()

def save_images(images, size, image_path):
	return imsave(images, size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
