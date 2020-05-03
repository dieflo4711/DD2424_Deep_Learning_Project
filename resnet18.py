import torch, os
import torch.nn as nn
import torchvision

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from torchvision import models
from torchsummary import summary

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim

# Basic Block
class Conv2dAuto(nn.Conv2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)  
conv = conv3x3(in_channels=32, out_channels=64)

def activation_func(activation):
	return  nn.ModuleDict([
		['relu', nn.ReLU(inplace=True)],
		['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
		['selu', nn.SELU(inplace=True)],
		['none', nn.Identity()]
	])[activation]

# Residual Block
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, activation='relu'):
		super().__init__()
		self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
		self.blocks = nn.Identity()
		self.activate = activation_func(activation)
		self.shortcut = nn.Identity()   

	def forward(self, x):
		residual = x
		if self.should_apply_shortcut: residual = self.shortcut(x)
		x = self.blocks(x)
		x += residual
		x = self.activate(x)
		return x

	@property
	def should_apply_shortcut(self):
		return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
	def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)
		self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
		self.shortcut = nn.Sequential(
			nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
			nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

	@property
	def expanded_channels(self):
		return self.out_channels * self.expansion

	@property
	def should_apply_shortcut(self):
		return self.in_channels != self.expanded_channels

# Basic Block
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
	return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(ResNetResidualBlock):
	"""
	Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
	"""
	expansion = 1
	def __init__(self, in_channels, out_channels, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)
		self.blocks = nn.Sequential(
			conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
			activation_func(self.activation),
			conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
		)

# Bottleneck
class ResNetBottleNeckBlock(ResNetResidualBlock):
	expansion = 4
	def __init__(self, in_channels, out_channels, *args, **kwargs):
		super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
		self.blocks = nn.Sequential(
			conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
			 activation_func(self.activation),
			 conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
			 activation_func(self.activation),
			 conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
		)

# Layer
class ResNetLayer(nn.Module):
	def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
		super().__init__()
		# 'We perform downsampling directly by convolutional layers that have a stride of 2.'
		downsampling = 2 if in_channels != out_channels else 1

		self.blocks = nn.Sequential(
			block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
			*[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
		)

	def forward(self, x):
		x = self.blocks(x)
		return x

# Encoder
class ResNetEncoder(nn.Module):
	"""
	ResNet encoder composed by layers with increasing features.
	"""
	def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], activation='relu', block=ResNetBasicBlock, *args, **kwargs):
		super().__init__()
		self.blocks_sizes = blocks_sizes

		self.gate = nn.Sequential(
			nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(self.blocks_sizes[0]),
			activation_func(activation),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)

		self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
		self.blocks = nn.ModuleList([ 
			ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block,*args, **kwargs),
			*[ResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs) 
			for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
		])


	def forward(self, x):
		x = self.gate(x)
		for block in self.blocks:
			x = block(x)
		return x

# Decoder
class ResnetDecoder(nn.Module):
	""" This class represents the tail of ResNet. It performs a global pooling and maps the output to the
		correct class by using a fully connected layer."""

	def __init__(self, in_features, n_classes):
		super().__init__()
		self.avg = nn.AdaptiveAvgPool2d((1, 1))
		self.decoder = nn.Linear(in_features, n_classes)

	def forward(self, x):
		x = self.avg(x)
		x = x.view(x.size(0), -1)
		x = self.decoder(x)
		return x

# ResNet

class ResNet(nn.Module):
	def __init__(self, in_channels, n_classes, *args, **kwargs):
		super().__init__()
		self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
		self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
	    
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
	return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

def main():
	dataset_dir = 'tiny-imagenet-200/'
	data_dir = 'tiny-imagenet-200/train/'
	valid_size = 0.2

	train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
	test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])

	train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
	test_data = datasets.ImageFolder(data_dir, transform=test_transforms)
	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.shuffle(indices)
	from torch.utils.data.sampler import SubsetRandomSampler
	train_idx, test_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx)
	trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=4)
	testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=4)
	

	#trainset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transforms.ToTensor())
	#trainloader = data.DataLoader(trainset, batch_size=5, shuffle=True)
	#testset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transforms.ToTensor())
	#testloader = data.DataLoader(testset, batch_size=5, shuffle=True)
	#mapping = get_nnumber_to_name(dataset_dir + 'words.txt')

	model = resnet18(3, 200)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for param in model.parameters():
		param.requires_grad = False

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	model.to(device)

	epochs = 1
	steps = 0
	running_loss = 0
	print_every = 10
	train_losses, test_losses = [], []

	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1
			print('steps: ' + str(steps))
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
			#loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % print_every == 0:
				test_loss = 0
				accuracy = 0
				model.eval()

				with torch.no_grad():
					i = 0
					for inputs, labels in testloader:
						i += 1
						print(i)
						inputs, labels = inputs.to(device), labels.to(device)
						logps = model.forward(inputs)
						batch_loss = criterion(logps, labels)
						test_loss += batch_loss.item()

						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape) 
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
						if i == 100:
							break
				train_losses.append(running_loss/len(trainloader))
				test_losses.append(test_loss/len(testloader))                    
				print(f"Epoch {epoch+1}/{epochs}.. "
						f"Train loss: {running_loss/print_every:.3f}.. "
						f"Test loss: {test_loss/len(testloader):.3f}.. "
						f"Test accuracy: {accuracy/len(testloader):.3f}")
				running_loss = 0
				model.train()
			if steps == 11:
				break
	torch.save(model, 'aerialmodel.pth')


def main2():
	dataset_dir = 'tiny-imagenet-200/'
	dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transforms.ToTensor())
	dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)
	#model = models.resnet18(False)
	model = resnet18(3, 1000)
	summary(model, (3, 224, 224))

main2()
