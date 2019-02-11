import os, time, pickle
import torch
import numpy as np
import torch.nn as nn
import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn, optim, autograd
import math
import glob
import argparse
import skvideo
import cv2 as cv

from networks import Generator_Img, Discriminator_Img_SN, Generator_normal_img, Generator_Img_nofc, Generator_normal_img, Discriminator_origin

import pdb

def str2bool(v):
	if v.lower() in ('true','t'):
		return True
	else:
		return False

parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
					 help='set -1 when you use cpu')
parser.add_argument('--ngpu', type=int, default=1,
					 help='set the number of gpu you use')
parser.add_argument('--batch_size', type=int, default=1,
					 help='set batch_size, default: 1')
parser.add_argument('--niter', type=int, default=120000,
					 help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
					 help='set 1 when you use pre-trained models')
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--dataset', type=str, default='UCFsports')
parser.add_argument('--is_bn', type=str2bool, default='True')
parser.add_argument('--cls_weight', type=float, default='10')
parser.add_argument('--relativistic_loss', type=str2bool, default='False')
parser.add_argument('--loss_type', default='hinge', type=str)



args       = parser.parse_args()
cuda       = args.cuda
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train
num_workers = args.num_workers
is_bn = args.is_bn

def timeSince(since):
	now = time.time()
	s = now - since
	d = math.floor(s / ((60**2)*24))
	h = math.floor(s / (60**2)) - d*24
	m = math.floor(s / 60) - h*60 - d*24*60
	s = s - m*60 - h*(60**2) - d*24*(60**2)
	return '%dd %dh %dm %ds' % (d, h, m, s)


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
	torch.cuda.set_device(0)

T = 32#16
L = 32


## set models ##
img_size = 96
nc = 3
ndf = 64 # from dcgan
ngf = 64
d_E = 100
hidden_size = 100 # guess
d_C = 100
d_M = 397
nz  = 100
d_cls = 3
cls_weight = args.cls_weight
Dirac_weight = 10


#loss epoch
criterion = nn.BCELoss()
L1loss = nn.L1Loss()
CEloss = nn.CrossEntropyLoss()
epochs = args.epoch

Gen = Generator_Img(nc, ngf, nz=nz, cls=d_cls, ngpu=ngpu, is_bn=is_bn)
#Gen = Generator_normal_img(nc, ngf, nz=nz, cls=d_cls, ngpu=ngpu, is_bn=is_bn)
Dis_I_projection = Discriminator_Img_SN(nc, ndf=ndf, ngpu=ngpu, num_cls = d_cls)
Dis_I = Discriminator_origin(nc, ndf=ndf, ngpu=ngpu, num_cls = d_cls)
Gen2 = Generator_normal_img(nc, ngf, nz=nz, cls=d_cls, ngpu=ngpu, is_bn = is_bn)


''' prepare for train '''

current_path = os.path.dirname(__file__)
trained_path = os.path.join(current_path, 'trained_models')


def checkpoint(model, optimizer, epoch):
	dir_path = os.path.join(trained_path, args.dataset , args.comment)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	filename = os.path.join(trained_path, args.dataset ,args.comment, '%s_epoch-%d' % (model.__class__.__name__, epoch))
	torch.save(model.state_dict(), filename + '.model')
	torch.save(optimizer.state_dict(), filename + '.state')



''' adjust to cuda '''

if cuda == True:
	criterion.cuda()
	Gen.cuda()
	Dis_I.cuda()
	L1loss.cuda()
	CEloss.cuda()

	Gen2.cuda()
	Dis_I_projection.cuda()

'''dataloader'''
dataset = args.dataset
transform_=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
dataroot = './data' #'../dataset/UCFsports_image'
if dataset == 'UCFsports':
	dataloader = DataLoader(utils.UCFsports(root_dir = dataroot, transform = transform_), batch_size = batch_size, shuffle=True, num_workers = num_workers)

# setup optimizer
lr_D = 0.0002 #After lunch, edit here to same as lr_G!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lr_G = 0.0002
betas=(0.0, 0.999)


optim_Gen = optim.Adam(Gen.parameters(), lr=lr_G, betas=betas)
optim_DisI = optim.Adam(Dis_I.parameters(), lr=lr_D, betas=betas)

optim_Gen2 = optim.Adam(Gen2.parameters(), lr=lr_G, betas=betas)
optim_Dis_I_projection = optim.Adam(Dis_I_projection.parameters(), lr=lr_D, betas=betas)

''' use pre-trained models '''

if pre_train == True:
	Dis_I.load_state_dict(torch.load(trained_path + '/Discriminator_I.model'))
	Gen.load_state_dict(torch.load(trained_path + '/Generator_I.model'))
	optim_DisI.load_state_dict(torch.load(trained_path + '/Discriminator_I.state'))
	optim_Gen.load_state_dict(torch.load(trained_path + '/Generator_I.state'))


def compute_grad2(d_out, x_in):
	_batch_size = x_in.size(0)
	grad_dout = autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
	grad_dout2 = grad_dout.pow(2)
	assert(grad_dout2.size() == x_in.size())
	reg = grad_dout2.view(_batch_size, -1).sum(1)

	return reg

def visualize_results(epoch, z_, y):
		Gen.eval()
		path = os.path.join(current_path, 'generated_videos',dataset, args.comment)
		if not os.path.exists(os.path.join(current_path, 'generated_videos',dataset, args.comment)):
			os.makedirs(os.path.join(current_path, 'generated_videos',dataset, args.comment))

		tot_num_samples = min(16, batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
		samples = Gen(z_,embedding_text)
		
		samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
		
		utils.save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], os.path.join(path, args.comment+'_epoch%03d'%epoch+'_F.png'))



def draw_result(epoch, G_, image_):
		path = os.path.join(current_path, 'generated_videos',dataset, args.comment)
		if not os.path.exists(os.path.join(current_path, 'generated_videos',dataset, args.comment)):
			os.makedirs(os.path.join(current_path, 'generated_videos',dataset, args.comment))

		tot_num_samples = min(16, batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
		samples = G_
		real_samples = image_
		
		samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)	
		real_samples = real_samples.cpu().data.numpy().transpose(0, 2, 3, 1)

		utils.save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], os.path.join(path, args.comment+'_epoch%03d'%epoch+'_F.png'))
		utils.save_images(real_samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], os.path.join(path, args.comment+'_epoch%03d'%epoch+'_R.png'))

def dis_hinge(dis_fake, dis_real):
	loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
	return loss

def gen_hinge(dis_fake, dis_real=None):
	return -torch.mean(dis_fake)

def dis_dcgan(dis_fake, dis_real):
	loss = torch.mean(torch.nn.functional.softplus(-dis_real)) + torch.mean(torch.nn.functional.softplus(dis_fake))
	return loss

def gen_dcgan(dis_fake, dis_real=None):
	return torch.mean(torch.nn.functional.softplus(-dis_fake))

def compute_grad2(d_out, x_in):
	_batch_size = x_in.size(0)
	grad_dout = autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
	grad_dout2 = grad_dout.pow(2)
	assert (grad_dout2.size() == x_in.size())
	reg = grad_dout2.view(_batch_size, -1).sum(1)
	return reg


'''train set'''
#action_list = ['golf','shoot_bow','cartwheel','draw_sword','pushup','swing_baseball']
action_list = ['Diving', 'Golf', 'SkateBoarding']

train_hist = {}
train_hist['D_loss'] = []
train_hist['G_loss'] = []

y_real_ = Variable(torch.ones(batch_size, 1).cuda())
y_fake_ = Variable(torch.zeros(batch_size, 1).cuda())

n_dis = 1

''' train models '''

start_time = time.time()
Dis_I.train()
for epoch in range(epochs):
	
	Gen.train() # If model change, please check here!
	Gen2.train() # if model change, Please check here!!!

	for itr, (image, embedding, text, action) in enumerate(dataloader):		
		if itr == dataloader.dataset.__len__() // batch_size:
			break

		''' prepare real images '''

		# real_videos.size() => (batch_size, nc, T, img_size, img_size)
		embedding_text = Variable(embedding.cuda())
		z_ = Variable(torch.rand(batch_size, nz).cuda())

		y_onehot_ = torch.zeros(batch_size, d_cls)
		y_onehot_.scatter_(1, action.view(-1,1), 1)
		y_onehot = Variable(y_onehot_.cuda())
		
		action = Variable(action.type(torch.LongTensor).cuda())

		image_ = Variable(image.cuda())

		''' Step1 : Discriminator train '''
		D_loss_out = 0
		G_loss_out = 0

		for i in range(n_dis):
			#optim_DisI.zero_grad()
			optim_Dis_I_projection.zero_grad()
			image_.requires_grad_()
			
			#G_ = Gen(z_, embedding_text)
			#G_ = Gen(z_, action)
			G_ = Gen2(z_, y_onehot)
			#G_ = Gen(z_, y_onehot)

			#dis_fake = Dis_I(G_, embedding_text)
			#dis_fake = Dis_I(G_, action)
			#dis_fake, cls_fake = Dis_I(G_)
			dis_fake = Dis_I_projection(G_, action)

			#dis_real = Dis_I(image_, embedding_text)
			#dis_real = Dis_I(image_, action)
			#dis_real, cls_real = Dis_I(image_)
			dis_real = Dis_I_projection(image_, action)

			#D loss calc
			if not args.relativistic_loss:
				dis_fake = dis_fake
				dis_real = dis_real

			else:
				C_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
				C_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)

				dis_fake -= C_xf_tilde
				dis_real -= C_xr_tilde

			if args.loss_type == 'hinge':
				D_loss = dis_hinge(dis_fake, dis_real)
			else:
				#D_loss = dis_dcgan(dis_fake, dis_real)
				GAN_loss = criterion(dis_real, y_real_) + criterion(dis_fake, y_fake_)
				cls_loss = CEloss(cls_real, action)
				D_loss = GAN_loss + cls_loss
				
			D_real_reg = 10 * compute_grad2(dis_real, image_).mean()
			#D_fake_reg = 10 * compute_grad2(dis_fake, G_).mean()
			D_loss += D_real_reg# + D_fake_reg
			#D_loss = D_loss

			if i%5 ==0:
				D_loss_out = D_loss
			D_loss.backward(retain_graph=True)
			
			num_correct_real = torch.sum(dis_real>0.5).item()
			num_correct_fake = torch.sum(dis_fake<0.5).item()
			D_acc = float(num_correct_real + num_correct_fake) / (batch_size*2)
			if D_acc < 100:#0.8:
				#optim_DisI.step()
				optim_Dis_I_projection.step()

		for j in range(1):
			''' Step2 : Generator train '''
			#optim_Gen.zero_grad()
			optim_Gen2.zero_grad()
			#G_ = Gen(z_, embedding_text)
			#dis_fake = Dis_I(G_, embedding_text)
			#G_ = Gen(z_, action)
			G_ = Gen2(z_, y_onehot)
			#dis_fake = Dis_I(G_, action)
			#dis_fake, cls_fake = Dis_I(G_)
			dis_fake = Dis_I_projection(G_, action)

			if not args.relativistic_loss:

				dis_fake = dis_fake
				dis_real = None

			else:
				assert dis_real is not None	

				C_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
				C_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)

				dis_fake -= C_xf_tilde
				dis_real -= C_xr_tilde

			if args.loss_type == 'hinge':
				G_loss = gen_hinge(dis_fake, dis_real)
			else:
				#G_loss = gen_dcgan(dis_fake, y_real_)
				GAN_loss = criterion(dis_fake, y_real_)
				cls_loss = CEloss(cls_fake, action)
				G_loss = GAN_loss + cls_loss

			G_loss.backward()
			#optim_Gen.step()
			optim_Gen2.step()

			if j % 4 ==0:
				G_loss_out = G_loss


		train_hist['D_loss'].append(D_loss_out.item())
		train_hist['G_loss'].append(G_loss_out.item())


		
		if ((itr == 0) or (itr == int(dataloader.dataset.__len__() // batch_size) -1)) & (epoch%1==0): #True:
			print('[%d/%d] (%s) Loss_Di: %.4f Loss_Gi: %.4f  D_acc : %.4f'% (epoch, epochs, timeSince(start_time), D_loss.item(), G_loss.item(),D_acc))

		if ((itr == 0) or (itr == int(dataloader.dataset.__len__() // batch_size) -1)) & (epoch>=5 and epoch%5==0):
			#visualize_results(epoch, z_, embedding_text)
			draw_result(epoch, G_, image_)
			utils.loss_plot(train_hist, os.path.join(current_path, 'generated_videos',dataset, args.comment), args.comment)

		if ((itr == 0) or (itr == int(dataloader.dataset.__len__() // batch_size) -1)) and (epoch % 1000 == 0):
			checkpoint(Dis_I, optim_DisI, epoch)
			checkpoint(Gen, optim_Gen, epoch)