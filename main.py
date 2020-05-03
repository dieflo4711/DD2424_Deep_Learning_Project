import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from resnet18 import resnet18
from resnet import ResNet


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
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

def get_label(label, dataset):
    nnumber = dataset.classes[label.item()]
    return mapping[nnumber]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if __name__ == "__main__":
    print("Reading data...")
    dataset_dir = './data/tiny-imagenet-200/'
    trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=5, shuffle=True)
    print("Loaded")
    mapping = get_nnumber_to_name(dataset_dir + 'words.txt')


    #net = resnet18(3, 200).cuda()
    net = ResNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print(str(i) + " of " + str(len(trainloader)))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())

            loss = criterion(outputs.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

