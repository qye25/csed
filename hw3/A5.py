import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import pandas as pd
import sys
import torch

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def load_dataset():
    mndata = MNIST("/Users/qye/Desktop/CSE 546/hw1-folder/code/data")
    # mndata = MNIST('./data/')
    X_train, labels_train = map(torch.tensor, mndata.load_training())
    X_test, labels_test = map(torch.tensor, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test


def neural_net1(x, y, w0, b0, w1, b1):
    acc = 0
    optimizer = torch.optim.Adam([w0, b0, w1, b1], lr=0.01)
    i = 0
    loss_list = []
    while acc < 0.99:

        h_relu = torch.nn.functional.relu(torch.matmul(x, w0.t()) + b0)
        y_pred = torch.matmul(h_relu, w1.t()) + b1

        acc = accuracy(y_pred, y)

        loss = torch.nn.functional.cross_entropy(y_pred, y)
        loss_list.append(loss.detach().item())

        if i % (100 // 10) == 0:
            print("{},\t{:.2f},\t{:.2f}".format(i, loss.item(), acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    return w0, b0, w1, b1, loss_list


def neural_net2(x, y, w0, b0, w1, b1, w2, b2):
    acc = 0
    optimizer = torch.optim.Adam([w0, b0, w1, b1, w2, b2], lr=0.01)
    i = 0
    loss_list = []
    while acc < 0.99:

        h1_relu = torch.nn.functional.relu(torch.matmul(x, w0.t()) + b0)
        h2_relu = torch.nn.functional.relu(torch.matmul(h1_relu, w1.t()) + b1)
        y_pred = torch.matmul(h2_relu, w2.t()) + b2

        acc = accuracy(y_pred, y)

        loss = torch.nn.functional.cross_entropy(y_pred, y)
        loss_list.append(loss.detach().item())

        if i % (100 // 10) == 0:
            print("{},\t{:.2f},\t{:.2f}".format(i, loss.item(), acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    return w0, b0, w1, b1, w2, b2, loss_list


def accuracy(y_pred, labels):
    labels_pred = torch.argmax(y_pred, dim=1).detach().numpy()
    accu = (labels_pred == labels.numpy()).mean()
    return accu


###############################################
# a
X_train, labels_train, X_test, labels_test = load_dataset()

d = 784
h = 64
k = 10
alpha1 = 1 / np.sqrt(d)
alpha2 = 1 / np.sqrt(h)

w0 = torch.FloatTensor(h, d).uniform_(-alpha1, alpha1).requires_grad_()
b0 = torch.FloatTensor(h).uniform_(-alpha1, alpha1).requires_grad_()
w1 = torch.FloatTensor(k, h).uniform_(-alpha2, alpha2).requires_grad_()
b1 = torch.FloatTensor(k).uniform_(-alpha2, alpha2).requires_grad_()


w0, b0, w1, b1, loss_list = neural_net1(X_train, labels_train, w0, b0, w1, b1)

plt.plot(loss_list)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training plot on wide, shallow network")
plt.savefig('A5a.png')
plt.show()

h_relu = torch.nn.functional.relu(torch.matmul(X_test, w0.t()) + b0)
y_pred = torch.matmul(h_relu, w1.t()) + b1

acc = accuracy(y_pred, labels_test)
print("Testing Accuracy: ")
print(acc)

loss = torch.nn.functional.cross_entropy(y_pred, labels_test)
print("Testing Loss: ")
print(loss.detach().item())

########################################################
# b

d = 784
h = 32
k = 10
alpha1 = 1 / np.sqrt(d)
alpha2 = 1 / np.sqrt(h)

w0 = torch.FloatTensor(h, d).uniform_(-alpha1, alpha1).requires_grad_()
b0 = torch.FloatTensor(h).uniform_(-alpha1, alpha1).requires_grad_()
w1 = torch.FloatTensor(h, h).uniform_(-alpha2, alpha2).requires_grad_()
b1 = torch.FloatTensor(h).uniform_(-alpha2, alpha2).requires_grad_()
w2 = torch.FloatTensor(k, h).uniform_(-alpha2, alpha2).requires_grad_()
b2 = torch.FloatTensor(k).uniform_(-alpha2, alpha2).requires_grad_()

# X_train, labels_train, X_test, labels_test = load_dataset()

w0, b0, w1, b1, w2, b2, loss_list = neural_net2(
    X_train, labels_train, w0, b0, w1, b1, w2, b2
)

plt.plot(loss_list)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training plot on narrow, deeper network")
plt.savefig('A5b.png')
plt.show()

h1_relu = torch.nn.functional.relu(torch.matmul(X_test, w0.t()) + b0)
h2_relu = torch.nn.functional.relu(torch.matmul(h1_relu, w1.t()) + b1)
y_pred = torch.matmul(h2_relu, w2.t()) + b2

acc = accuracy(y_pred, labels_test)
print("Testing Accuracy: ")
print(acc)

loss = torch.nn.functional.cross_entropy(y_pred, labels_test)
print("Testing Loss: ")
print(loss.detach().item())
