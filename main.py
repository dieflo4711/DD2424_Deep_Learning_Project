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
            images, labels = data
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
            running_acc = 0.0

            for i, data in enumerate(loader, 1):
                print(str(i) + " of " + str(len(train_loader)) + " epoch: " + str(str(epoch + 1)))
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Global variables
dataset_dir = './data/tiny-imagenet-200/'
model_dir = './model/net.pth'
epochs = 10
eta = 0.001
momentum = 0.9
batch_size = 5

if __name__ == "__main__":
    print("Reading data...")
    trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    valset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform)
    testset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
    # Reduce trainset & valiset for debugging
    trainset, _ = torch.utils.data.random_split(trainset, [1000, len(trainset) - 1000])
    valset, _ = torch.utils.data.random_split(valset, [500, len(valset) - 500])
    testset, _ = torch.utils.data.random_split(testset, [500, len(testset) - 500])
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    print("Done reading")
    mapping = get_nnumber_to_name(dataset_dir + 'words.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet18(3, 200).to(device)

    train_model(net, train_loader, val_loader, trainset, valset)
    #load_saved_model(net)
    #test_model(net, test_loader)
