import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    train_size = len(trainset)

    trainset, validset = torch.utils.data.random_split(
        trainset, (int(0.8 * train_size), int(0.2 * train_size))
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=4, shuffle=True, num_workers=2
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainset, trainloader, validset, validloader, testset, testloader, classes



def train(trainloader, validloader, net, lr, momentum):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(15):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.view(-1, 3072))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        train_acc = accuracy(trainloader, net)    
        valid_acc =  accuracy(validloader, net)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        # print('Train Accuracy: ' + str(train_acc))
        # print('Validation Accuracy: ' + str(valid_acc))

    print('Finished Training')
    return train_acc_list, valid_acc_list


def accuracy(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.view(-1, 3072))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return correct / total

trainset, trainloader, validset, validloader, testset, testloader, classes = load_data()

net1 =  torch.nn.Sequential(
    torch.nn.Linear(3072, 10, 10)
)

'''
M=500, acc = 65%
M=525, acc = 69%
'''

M = 520
acc=0
while acc < 0.7:
    net2 = torch.nn.Sequential(
    torch.nn.Linear(3072, M, M),
    torch.nn.ReLU(),
    torch.nn.Linear(M, 10, 10)
)
    train_acc_list2, valid_acc_list2=train(trainloader, validloader, net2, lr=0.001, momentum=0.9)
    acc = valid_acc_list[-1]
    M += 5


acc = 0
lr = 0.001
train_list = np.array([])
val_list = np.array([])
while acc < 0.7:
    train_acc_list1, valid_acc_list1 = train(trainloader,validloader, net1, lr=lr, momentum=0.9)
    acc = valid_acc_list1[-1]
    train_list = np.append(train_list, train_acc_list1)
    val_list = np.append(val_list, valid_acc_list1)
    
    lr*=0.8
