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
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test


def reconstruct(x, mu, v, k):
    return mu + (x - mu).dot(v[:k].T).dot(v[:k])


def error(x, reconstruct_x):
    norm = np.linalg.norm(x - reconstruct_x, axis=1) ** 2
    return norm.mean()


#################
# a
X_train, labels_train, X_test, labels_test = load_dataset()
mu = X_train.mean(axis=0)
cov = np.cov(X_train.T)
u, s, v = np.linalg.svd(cov, full_matrices=True)

for i in [1, 2, 10, 30, 50]:
    lam_i = round(float(s[i - 1]), 4)
    print("lambda_" + str(i) + " = " + str(lam_i))

eigen_sum = s.sum()
print("Sum of lambdas = " + str(round(eigen_sum, 4)))

#################
# c

k = 100

err_train = []
err_test = []
sum_lam = []
for i in range(1, k + 1):
    recon_train = reconstruct(X_train, mu, v, i)
    recon_test = reconstruct(X_test, mu, v, i)
    err_train.append(error(X_train, recon_train))
    err_test.append(error(X_test, recon_test))
    sum_lam.append(1 - (s[:i].sum() / eigen_sum))

plt.plot(err_train)
plt.plot(err_test)
plt.xlabel("k")
plt.ylabel("Reconstruction error")
plt.title("Reconstruction error vs. k")
plt.legend(["Train", "Test"])
plt.savefig("A6_c1.png")
plt.show()

plt.plot(range(1, k + 1), sum_lam)
plt.xlabel("k")
plt.legend(["$1-{\sum_{i=1}^{k}\lambda_{i}}/{\sum_{i=1}^{d}\lambda_{i}}$"])
# plt.title('$1-{\sum_{i=1}^{k}\lambda_{i}}/{\sum_{i=1}^{d}\lambda_{i}}$ vs. k')
plt.savefig("A6_c2.png")
plt.show()


#################
# d

for i in range(10):
    plt.gcf().set_size_inches(15, 18)
    plt.subplot(4, 3, i + 1).set_title("Eigenvector " + str(i + 1))
    plt.imshow(v[i].reshape(28, 28))
plt.savefig("A6d.png")

#################
# e

digit2 = X_train[labels_train == 2][0].reshape(784, 1)
digit6 = X_train[labels_train == 6][0].reshape(784, 1)
digit7 = X_train[labels_train == 7][0].reshape(784, 1)
k_list = [5, 15, 40, 100]
digit_list = [digit2, digit6, digit7]

for digit in digit_list:
    plt.gcf().set_size_inches(25, 20)

    plt.subplot(5, 1, 1).set_title("Original Image")
    plt.imshow(digit.reshape(28, 28))
    for i in range(len(k_list)):
        recon = reconstruct(digit.T, mu, v, k_list[i])
        plt.subplot(5, 1, i + 2).set_title("Reconstructed Image k = " + str(k_list[i]))
        plt.imshow(recon.reshape(28, 28))
    plt.show()


np.rank(X_train)