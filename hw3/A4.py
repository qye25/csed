import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import pandas as pd
import sys

k = 10
n = 20
d = 5
x = np.random.uniform(0, 10, size=(n, d))


def load_dataset():
    mndata = MNIST("/Users/qye/Desktop/CSE 546/hw1-folder/code/data")
    # mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test


def k_means(x, k, eps):
    n, d = x.shape
    obj_list = []
    obj = sys.float_info.max
    old_obj = 0
    it = 0
    # randomly select k centers
    index = np.random.choice(n, k, replace=False)
    c = x[index]
    old_c = np.zeros_like(c)

    while np.linalg.norm(c-old_c, axis=1).max() > eps:
        old_c = c.copy()
        old_obj = obj
        dist = []
        for cj in c:
            dist_j = np.linalg.norm(cj - x, axis=1)**2
            dist.append(dist_j)

        dist = np.array(dist)
        group = dist.argmin(axis=0)
        obj = dist.min(axis=0).sum()

        for j in range(k):
            mu_j = x[group == j].mean(axis=0)
            c[j] = mu_j

        obj_list.append(obj)
        it += 1
        # print(it)
        # print(np.linalg.norm(c - old_c, axis=1).max())

    return c, obj_list


def error(center, x):
    dist = []
    for cj in center:
        dist_j = np.linalg.norm(cj - x, axis=1)**2
        dist.append(dist_j)
    dist = np.array(dist)
    # return dist.min(axis=1).mean()
    return dist.min(axis=0).mean()


X_train, labels_train, X_test, labels_test = load_dataset()

c1, obj1 = k_means(X_train, k=10, eps=0.005)
# c, clusters, obj = k_means(x,k=10, eps=0.005)
plt.plot(obj1)
plt.xlabel("Iterations")
plt.ylabel("Objective")
plt.savefig('A4b-1_new.png')
plt.show()


print(error(c1, X_train))
print(error(c1, X_test))

fig, axs = plt.subplots(5, 2, figsize=(10, 20))
i = 0
for row in range(5):
    for col in range(2):
        axs[row, col].imshow(c1[i].reshape((28, 28)))
        i += 1
plt.savefig('A4b-2_new.png')
plt.show()


k_list = [2, 4, 8, 16, 32, 64]
train_error = []
test_error = []
for k in k_list:
    print(k)
    c, obj = k_means(X_train, k, eps=0.005)
    train_error.append(error(c, X_train))
    test_error.append(error(c, X_test))

plt.plot(k_list, train_error, "o--")
plt.plot(k_list, test_error)
plt.xlabel("K")
plt.ylabel("Errors")
plt.legend(["Training Error", "Testing Error"])
plt.savefig('A4c_error.png')
plt.show()
