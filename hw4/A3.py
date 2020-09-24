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

def get_image_sample(x, y, k=10):
    n, d = x.shape
    x_sample =torch.zeros((k, d))
    for i in range(10):
        idx = np.where(y==i)[0][0]
        x_sample[i] =x[idx]
    return x_sample

def plot(x, x_recon, width, length):
    k, d = x.shape
    plt.gcf().set_size_inches(width, length)
    for i in range(2*k):
        idx = int(i/2)
        plt.subplot(k, 2, i+1)
        if i%2==0:
            plt.imshow(x[idx].reshape(28,28))
        else:
            plt.imshow(x_recon[idx].reshape(28,28))
    # plt.savefig('')
    plt.show()

def F(x, model):
    n, d = x.shape
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(100):
        x_recon = model(x)
        loss = torch.nn.functional.mse_loss(x_recon, x)

        # if i % (100 // 10) == 0:
        #     print("{},\t{:.2f}".format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, loss.detach().item()

# def F2(x, h):
#     n, d = x.shape

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     for i in range(100):
#         x_recon = model(x)
#         loss = torch.nn.functional.mse_loss(x_recon, x)

#         # if i % (100 // 10) == 0:
#         #     print("{},\t{:.2f}".format(i, loss.item()))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     return model, loss.detach().item()


X_train, labels_train, X_test, labels_test = load_dataset()
x_sample = get_image_sample(X_train, labels_train, k=10)
n, d = X_train.shape

#########################################################
# a

for h in [32, 64,128]:
    linear_model =  torch.nn.Sequential(
        torch.nn.Linear(d, h),
        torch.nn.Linear(h, d)
    )
    trained_linear, loss = F(X_train ,linear_model)
    print(loss)
    x_reconstruction = trained_linear(x_sample).detach().numpy()
    plot(x_sample, x_reconstruction, 5,20)

#########################################################
# b

for h in [32, 64,128]:
    non_linear_model  = torch.nn.Sequential(
        torch.nn.Linear(d, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d),
        torch.nn.ReLU()
    )
    trained_non_linear, loss=F(X_train,non_linear_model)
    print(loss)
    # x_sample = get_image_sample(X_train, labels_train, k=10)
    x_reconstruction = trained_non_linear(x_sample).detach().numpy()
    plot(x_sample, x_reconstruction, 5,20)

#########################################################
# c
pred_linear = trained_linear(X_test)
loss = torch.nn.functional.mse_loss(pred_linear, X_test)
print(loss.item())

pred_linear = trained_non_linear(X_test)
loss = torch.nn.functional.mse_loss(pred_linear, X_test)
print(loss.item())