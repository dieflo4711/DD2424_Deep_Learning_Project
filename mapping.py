import torch, os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImageFolderWithPaths(datasets.ImageFolder):
	"""Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder """
	# override the __getitem__ method. this is the method that dataloader calls
	def __getitem__(self, index):
		# this is what ImageFolder normally returns 
		original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
		# the image file path
		path = self.imgs[index][0]
		# make a new tuple that includes original and the path
		tuple_with_path = (original_tuple + (path,))
		return tuple_with_path

def imshow(img):
	img = img / 2 + 0.5 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def get_nnumber_to_name(filepath):
	mapping = {}
	with open(filepath) as f:
		for line in f:
			(nnumber, name) = line.split('	')
			mapping[nnumber] = name
		return mapping

def train_images(dataloader, dataset, mapping):
	dataiter = iter(dataloader)
	images, labels, paths = dataiter.next()

	for i in range(dataloader.batch_size):
		label = labels[i].item()
		nnumber = dataset.classes[labels[i]]
		name = mapping[nnumber]
		print(f'label: {label}, nnumber: {nnumber}, name: {name}')

	imshow(torchvision.utils.make_grid(images))

def get_file_to_nnumber(filepath):
	valmapping = {}
	with open(filepath) as f:
		for line in f:
			info = line.split('	')
			valmapping[info[0]] = info[1] # val_XXX.JPEG: nnumber
		return valmapping

def val_images(dataloader, dataset, mapping, valmapping):
	dataiter = iter(dataloader)
	images, labels, paths = dataiter.next()

	for i in range(dataloader.batch_size):
		file_name = paths[i].split('/')[-1]
		nnumber = valmapping[file_name]
		name = mapping[nnumber]
		print(f'file_name: {file_name}, nnumber: {nnumber}, name: {name}')
	imshow(torchvision.utils.make_grid(images))

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform_1 = transforms.Compose([
	transforms.ToTensor(),
	normalize])

transform_2 = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize
	])

transform_3 = transforms.Compose([
	transforms.RandomResizedCrop(64),
	transforms.ToTensor(),
	normalize
	])

transform_4 = transforms.Compose([
	transforms.RandomResizedCrop(64),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize
	])

if __name__ == "__main__":
	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	dataset_dir = 'tiny-imagenet-200/'
	mapping = get_nnumber_to_name(dataset_dir + 'words.txt')
	valmapping = get_file_to_nnumber(dataset_dir + 'val/val_annotations.txt')

	trainset = ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_1)
	trainloader = data.DataLoader(trainset, batch_size=5, shuffle=False)

	valset = ImageFolderWithPaths(os.path.join(dataset_dir, 'val'), transform=transform_1)
	valloader = data.DataLoader(valset, batch_size=5, shuffle=True)
	
	#val_images(valloader, valset, mapping, valmapping)
	#train_images(trainloader, trainset, mapping)

	# ---------------------------------------
	# data augmentation

	dataset = ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_1)
	dataloader = data.DataLoader(
		data.ConcatDataset([
			ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_1),
			ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_2),
			ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_3),
			ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_4)
			]), batch_size=5, shuffle=True)
	
	train_images(dataloader, dataset, mapping)
	print(len(trainloader))
	print(len(dataloader))
