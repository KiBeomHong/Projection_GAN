# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.utils import spectral_norm
from links import CategoricalConditionalBatchNorm2d
import pdb

class Discriminator_Img_SN(nn.Module):
	def __init__(self, nc=3, ndf=32, ngpu=1, num_cls=10):
		super(Discriminator_Img_SN, self).__init__()
		self.ngpu = ngpu
		self.num_cls = num_cls
		self.ndf = ndf
		
		self.optimblock = nn.Sequential(
			spectral_norm(nn.Conv2d(3, ndf, 3, 1, 1, bias=False)),
			nn.ReLU(),
			spectral_norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)),
			nn.AvgPool2d(2),
			nn.ReLU()

		)

		self.new_conv = nn.Sequential(
			#nn.ReLU(),
			spectral_norm(nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=False)),
			nn.AvgPool2d(2),
			nn.ReLU(),

			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 3, 1, 1, bias=False)),
			nn.AvgPool2d(2),
			nn.ReLU(),

			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 3, 1, 1, bias=False)),
			nn.AvgPool2d(2),
			nn.ReLU(),

		)
	
		self.conv = nn.Sequential(
			#96 -> 48 (32)
			spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf),
			nn.LeakyReLU(0.2, inplace=True),

			#48 -> 24 (64)
			spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace=True),

			#24 -> 12 (128)
			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace=True),

			#12 -> 6 (256)
			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace = True),
		)
		
		self.sigmoid = nn.Sigmoid()
		#self.embedding = spectral_norm(nn.Linear(1024, ndf*8,bias=False))
		self.embedding = spectral_norm(nn.Embedding(self.num_cls, ndf*8))
		self.linear = nn.Sequential(
			spectral_norm(nn.Linear(ndf*8, 1)),
			#nn.Sigmoid(),
		)

		#utils.initialize_weights(self)
	def forward(self, x, y):
		#feature = self.conv(x)
		feature = self.optimblock(x)
		feature = self.new_conv(feature)
		#feature = self.activation(feature)
		#Global pooling
		feature = torch.sum(feature, dim=(2,3))
		#pdb.set_trace()
		output = self.linear(feature)
		embedding = self.embedding(y)
		output += torch.sum(embedding*feature, dim=1, keepdim=True)
		#output = self.sigmoid(output)
		return output


class Discriminator_origin(nn.Module):
	def __init__(self, nc=3, ndf=32, ngpu=1, num_cls=10):
		super(Discriminator_origin, self).__init__()
		self.ngpu = ngpu
		self.num_cls = num_cls
		self.ndf = ndf

		self.conv = nn.Sequential(
			#96 -> 48 (32)
			spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf),
			nn.LeakyReLU(0.2, inplace=True),

			#48 -> 24 (64)
			spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace=True),

			#24 -> 12 (128)
			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace=True),

			#12 -> 6 (256)
			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace = True),

		)

		self.convGAN = nn.Sequential(
			spectral_norm(nn.Linear(ndf*8*6*6, 1024)),
			#nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2),
			spectral_norm(nn.Linear(1024,1)),
			nn.Sigmoid(),
		)

		self.convCLS = nn.Sequential(
			spectral_norm(nn.Linear(ndf*8*6*6, 1024)),
			#nn.InstanceNorm1d(1024),
			nn.LeakyReLU(0.2),
			spectral_norm(nn.Linear(1024,self.num_cls)),
		)

		#utils.initialize_weights(self)
	def forward(self, y):
		if isinstance(y.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			feature = nn.parallel.data_parallel(self.conv, y, range(self.ngpu))
			fGAN =  nn.parallel.data_parallel(self.convGAN, feature.view(-1,self.ndf*8*6*6), range(self.ngpu))
			fCLS =  nn.parallel.data_parallel(self.convCLS, feature.view(-1,self.ndf*8*6*6), range(self.ngpu))
		else:
			feature = self.conv(y)
			fGAN = self.convGAN(feature.view(-1,self.ndf*8*6*6))
			fCLS = self.convCLS(feature.view(-1,self.ndf*8*6*6))
		#x = (fGAN, fCLS)
		return fGAN, fCLS #x


class Generator_Img(nn.Module):
	def __init__(self, nc=3, ngf=64, nz=100, cls=6, ngpu=1, is_bn=True):
		super(Generator_Img, self).__init__()
		self.ngpu = ngpu
		self.is_bn = is_bn
		self.cls = cls
		self.nz = nz
		self.ngf = ngf
		self.bottom_width = 6
		self.text_embedding_dim = 1024
		self.num_cls = cls


		self.fc = nn.Sequential(
				nn.Linear(self.nz, 16*self.ngf*self.bottom_width**2),
			)

		self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.activation = nn.ReLU()

		self.deconv1 = nn.Conv2d(ngf*16, ngf*8, 3, 1, 1, bias=False)
		#self.b1 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*8)
		self.b1 = CategoricalConditionalBatchNorm2d(self.num_cls, ngf*16)

		self.deconv2 = nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False)
		#self.b2 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*4)
		self.b2 = CategoricalConditionalBatchNorm2d(self.num_cls, ngf*8)

		self.deconv3 = nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False)
		#self.b3 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*2)
		self.b3 = CategoricalConditionalBatchNorm2d(self.num_cls, ngf*4)

		self.deconv4 = nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False)
		self.b4 = CategoricalConditionalBatchNorm2d(self.num_cls, ngf*2)
		
		self.b5 = nn.BatchNorm2d(ngf)

		self.conv5 = nn.Conv2d(ngf, nc, 3, 1, 1)
		self.tanh = nn.Tanh()
		#utils.initialize_weights(self)

	def forward(self, z, y):
		z = self.fc(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
		#batch norm -> activation -> upsample -> conv
		feature1 = self.Upsample(self.activation(self.b1(z, y)))
		feature1 = self.deconv1(feature1)

		feature2 = self.Upsample(self.activation(self.b2(feature1, y)))
		feature2 = self.deconv2(feature2)

		feature3 = self.Upsample(self.activation(self.b3(feature2, y)))
		feature3 = self.deconv3(feature3)

		feature4 = self.Upsample(self.activation(self.b4(feature3, y)))
		feature4 = self.deconv4(feature4)
		
		output = self.activation(self.b5(feature4))
		output = self.tanh(self.conv5(output))

		return output

class Generator_Img_nofc(nn.Module):
	def __init__(self, nc=3, ngf=64, nz=100, cls=6, ngpu=1, is_bn=True):
		super(Generator_Img_nofc, self).__init__()
		self.ngpu = ngpu
		self.is_bn = is_bn
		self.cls = cls
		self.nz = nz
		self.ngf = ngf
		self.bottom_width = 4
		self.text_embedding_dim = 1024


		self.conv0 = nn.Conv2d(self.nz, ngf*16, 4, 1, 4, bias=False) #not Conv!! 
		
		self.fc1 = nn.Linear(self.nz, ngf*16*4*4, bias=False)
		self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.activation = nn.ReLU()

		self.deconv1 = nn.Conv2d(ngf*16, ngf*8, 3, 1, 1, bias=False)
		self.b1 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*8)

		self.deconv2 = nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False)
		self.b2 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*4)

		self.deconv3 = nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False)
		self.b3 = CategoricalConditionalBatchNorm2d(self.text_embedding_dim, ngf*2)

		self.deconv4 = nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False)
		self.b4 = nn.BatchNorm2d(ngf)

		self.conv5 = nn.Conv2d(ngf, nc, 3, 1, 1)
		self.tanh = nn.Tanh()
		#utils.initialize_weights(self)

	def forward(self, z, y):
		#z = self.conv0(z.view(-1,self.nz,1,1))
		z = self.fc1(z).view(-1, self.ngf*16, self.bottom_width, self.bottom_width)
		feature = self.Upsample(z)
		feature = self.deconv1(feature)
		feature = self.b1(feature,y)
		feature1 = self.activation(feature)

		feature1 = self.Upsample(feature1)
		feature1 = self.deconv2(feature1)
		feature1 = self.b2(feature1,y)
		feature2 = self.activation(feature1)

		feature2 = self.Upsample(feature2)
		feature2 = self.deconv3(feature2)
		feature2 = self.b3(feature2,y)
		feature3 = self.activation(feature2)

		feature3 = self.Upsample(feature3)
		feature4 = self.deconv4(feature3)
		feature4 = self.b4(feature4)
		
		#feature4 = self.activation(feature4)

		output = self.conv5(feature4)
		output = self.tanh(output)
		
		return output

class Generator_normal_img(nn.Module):
	def __init__(self, nc=3, ngf=32, nz=200, cls=6, ngpu=1, is_bn=True):
		super(Generator_normal_img, self).__init__()
		self.ngpu = ngpu
		self.is_bn = is_bn
		self.cls = cls
		self.nz = nz

		self.embedding = nn.Sequential(
				nn.Linear(1024, 100)
		)

		self.fc = nn.Sequential(
				nn.Linear(self.nz+self.cls, self.nz),
		)

		self.deconv = nn.Sequential(

			#1 -> 6 (512)
			nn.Conv2d(self.nz, ngf*16, 4, 1, 4, bias=False),
			nn.BatchNorm2d(ngf*16),
			nn.ReLU(),

			#6 -> 12 (256)
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(ngf*16, ngf*8, 3, 1, 1, bias=False),
			nn.BatchNorm2d(ngf*8),
			nn.ReLU(),

			#12 -> 24 (128)
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False),
			nn.BatchNorm2d(ngf*4),
			nn.ReLU(),

			#24 -> 48 (64)
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False),
			nn.BatchNorm2d(ngf*2),
			nn.ReLU(),

			#48 -> 96 (32)
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(ngf*2, nc, 3, 1, 1, bias=False),
			nn.Sigmoid()
		)

		#utils.initialize_weights(self)

	def forward(self, z, y):
		#y = self.embedding(y)
		z = torch.cat([z,y],1)
		feature = self.fc(z.view(z.size(0),-1))
		output = self.deconv(feature.unsqueeze(2).unsqueeze(3))
		return output


''' utils '''

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


