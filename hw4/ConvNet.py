import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, M = 100, k = 5, N = 14):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, M, k)
        self.pool = nn.MaxPool2d(N, stride=2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.vec_size = M * int(((33-k)/N)**2)
        self.fc2 = nn.Linear(self.vec_size, 10, 10)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.vec_size)
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x