import os
import time
import copy
import torch
import numpy as np
import mapping as mp
from torch import nn
from torch import optim
from resnet18 import resnet18
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        file_name = self.imgs[index][0].split('\\')[-1]  # .split('/')[-1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (file_name,))
        return tuple_with_path


def plot_graph(train_loss, val_loss, train_acc, val_acc):
    epochs_axis = np.arange(epochs) + 1
    plt.figure(1)
    plt.title("Training & validation loss")
    plt.plot(epochs_axis, train_loss, label="train")
    plt.plot(epochs_axis, val_loss, label="val")
    plt.ylim(0, 7)
    plt.xlim(1, epochs)
    plt.xticks(np.array([1, 5, 10, 15, 20]))
    plt.yticks(np.array([0, 2, 4, 6]))
    plt.legend(loc='upper left')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig('./results/loss.png')
    plt.figure(2)
    plt.title("Training & validation accuracy")
    plt.plot(epochs_axis, train_acc, label="train")
    plt.plot(epochs_axis, val_acc, label="val")
    plt.xlim(1, epochs)
    plt.xticks(np.array([1, 5, 10, 15, 20]))
    plt.yticks(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    plt.legend(loc='upper left')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig('./results/acc.png')
    #plt.show()


def test_model(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            if use_imageNet:
                images, labels, file_names = data
                labels = mp.get_labels(labels, file_names, valmapping, nnumber_to_idx)
            else:
                images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test accuracy: %d %%' % (
            100 * correct / total))


def train_model(model, criterion, optimizer, num_epochs=10):
    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_at_epoch = 0

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(dataloaders[phase], 1):
                if use_imageNet:
                    inputs, labels, file_names = data
                    if phase == 'validation':
                        labels = mp.get_labels(labels, file_names, valmapping, nnumber_to_idx)
                else:
                    inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'validation':
                val_loss[epoch] = running_loss / len(dataloaders[phase].dataset)
                val_acc[epoch] = running_corrects.float() / len(dataloaders[phase].dataset)
                if val_acc[epoch] > best_acc:
                    best_acc = val_acc[epoch]
                    best_at_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                lr_scheduler.step(val_loss[epoch])
            else:
                train_loss[epoch] = running_loss / len(dataloaders[phase].dataset)
                train_acc[epoch] = running_corrects.float() / len(dataloaders[phase].dataset)
                #print('Training accuracy at epoch %d: %0.3f' % (epoch + 1, train_acc[epoch]))
                #lr_scheduler.step()

    model.load_state_dict(best_model_wts)
    print('Best validation accuracy (epoch %d): %0.3f' % (best_at_epoch + 1, best_acc))
    print('Validation accuracy at epoch %d: %0.3f' % (num_epochs, val_acc[-1]))
    plot_graph(train_loss, val_loss, train_acc, val_acc)
    return model


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
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

transform_5 = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms.ToTensor(),
    normalize])

dataset_dir = './data/tiny-imagenet-200/'
epochs = 20 #20
eta = 0.001 #pow(10, -4)#3e-4 #0.001
weight_decay = pow(10, -4)
batch_size = 256  #35
num_workers = 6
use_imageNet = True

def tiny_imagenet(overfit=False, augment=False):
    trainset = ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform)
    valset = ImageFolderWithPaths(os.path.join(dataset_dir, 'val'), transform=transform)
    if overfit:
        trainset, _ = torch.utils.data.random_split(trainset, [500, len(trainset)-500])
        nnumber_to_idx = dict(zip(trainset.dataset.classes, np.arange(len(trainset.dataset.classes))))
    else:
        nnumber_to_idx = dict(zip(trainset.classes, np.arange(len(trainset.classes))))
    valset, testset = torch.utils.data.random_split(valset, [5000, 5000])
    if augment:
        trainloader = DataLoader(
            ConcatDataset([
                ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform),
                ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_2),
                ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_3),
                ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_4),
                ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_5)
            ]), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valmapping = mp.get_file_to_nnumber(dataset_dir + 'val/val_annotations.txt')
    model = resnet18(3, 200)
    return model, trainloader, valloader, testloader, nnumber_to_idx, valmapping

def cifar_10():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model = resnet18(3, 10)
    return model, trainloader, valloader, testloader


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_imageNet:
        model, trainloader, valloader, testloader, nnumber_to_idx, valmapping = \
            tiny_imagenet(overfit=False, augment=True)
    else:
        model, trainloader, valloader, testloader = cifar_10()

    dataloaders = {
        "train": trainloader,
        "validation": valloader
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    start = time.time()
    model = train_model(model, criterion, optimizer, num_epochs=epochs)
    end = time.time()
    print("Training time: %d min" % ((end - start)/60))
    test_model(model, testloader)
