import torch, os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
	dataset_dir = 'tiny-imagenet-200/'
	dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transforms.ToTensor())
	dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)
	mapping = get_nnumber_to_name(dataset_dir + 'words.txt')

	dataiter = iter(dataloader)
	images, labels = dataiter.next()

	for i in range(5):
		label = labels[i].item()
		nnumber = dataset.classes[labels[i]]
		name = mapping[nnumber]
		print(f'label: {label}, nnumber: {nnumber}, name: {name}')

	imshow(torchvision.utils.make_grid(images))