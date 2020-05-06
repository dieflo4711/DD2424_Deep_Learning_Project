import os
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from resnet18 import resnet18


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


def get_file_to_nnumber(filepath):
    valmapping = {}
    with open(filepath) as f:
        for line in f:
            info = line.split('	')
            valmapping[info[0]] = info[1]  # val_XXX.JPEG: nnumber
        return valmapping


def get_labels(labels, file_names, valmapping, nnumber_to_idx):
    for i in range(len(labels)):
        nnumber = valmapping[file_names[i]]
        labels[i] = nnumber_to_idx[nnumber]
    return labels


def load_saved_model(net):
    net.load_state_dict(torch.load(model_dir))


def plot_graph(train_loss, val_loss, train_acc, val_acc):
    epochs_axis = np.arange(epochs) + 1
    plt.figure(1)
    plt.title("Training & validation loss")
    plt.plot(epochs_axis, train_loss, label="train")
    plt.plot(epochs_axis, val_loss, label="val")
    plt.ylim(0, 7)
    # plt.yticks(np.array([0, 2, 4, 6]))
    plt.legend(loc='upper left')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.figure(2)
    plt.title("Training & validation accuracy")
    plt.plot(epochs_axis, train_acc, label="train")
    plt.plot(epochs_axis, val_acc, label="val")
    plt.legend(loc='upper left')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()


def test_model(net, testloader):
    print('Testing model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, file_names = data
            labels = get_labels(labels, file_names, valmapping, nnumber_to_idx)
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def train_model(net, train_loader, val_loader, trainset, valset):
    train_loss = np.zeros(epochs)
    train_acc = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=eta, momentum=momentum)

    print("Started training...")
    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
                loader = train_loader
            else:
                net.eval()  # Set model to evaluate mode
                loader = val_loader

            running_loss = 0.0
            running_acc = 0

            for i, data in enumerate(loader, 1):
                print(str(i) + " of " + str(len(train_loader)) + " epoch: " + str(str(epoch + 1)))
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
                inputs, labels, file_names = data
                if phase == 'val':
                    labels = get_labels(labels, file_names, valmapping, nnumber_to_idx)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, acc_index = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(acc_index == labels.data)
            if phase == 'train':
                train_loss[epoch] = running_loss / len(trainset)
                train_acc[epoch] = running_acc.double() / len(trainset)
            else:
                val_loss[epoch] = running_loss / len(valset)
                val_acc[epoch] = running_acc.double() / len(valset)
    print('Finished Training')
    # Or save the most accuracte model based on val data?
    torch.save(net.state_dict(), model_dir)
    plot_graph(train_loss, val_loss, train_acc, val_acc)


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

# Global variables
dataset_dir = './data/tiny-imagenet-200/'
model_dir = './model/net.pth'
epochs = 10
eta = 0.001
momentum = 0.9
batch_size = 256

if __name__ == "__main__":
    print("Reading data...")
    trainset = ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform)
    valset = ImageFolderWithPaths(os.path.join(dataset_dir, 'val'), transform=transform)
    # testset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)

    # trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])
    # testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Reduce trainset & valiset for debugging
    #trainset, _ = torch.utils.data.random_split(trainset, [1000, len(trainset) - 1000])
    #valset, _ = torch.utils.data.random_split(valset, [500, len(valset) - 500])
    # testset, _ = torch.utils.data.random_split(testset, [500, len(testset) - 500])
    
    #train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    train_loader = data.DataLoader(
        data.ConcatDataset([
            ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform),
            ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_2),
            ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_3),
            ImageFolderWithPaths(os.path.join(dataset_dir, 'train'), transform=transform_4)
            ]), batch_size=batch_size, shuffle=True)

    val_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    # test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    print("Done reading")

    nnumber_to_idx = dict(zip(trainset.classes, np.arange(len(trainset.classes))))
    valmapping = get_file_to_nnumber(dataset_dir + 'val/val_annotations.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet18(3, 200).to(device)

    # print(dict([(value, key) for key, value in dict(trainset.classes).items()]))
    #for data in val_loader:
    #    images, labels, file_names = data
    #    labels = get_labels(labels, file_names, valmapping, nnumber_to_idx)
    #    print(labels)
    #    break


    train_model(net, train_loader, val_loader, trainset, valset)
    # load_saved_model(net)
    # test_model(net, test_loader)
    test_model(net, val_loader)
