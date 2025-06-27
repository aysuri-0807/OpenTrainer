import torch
from torch import nn, save, load
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim import Adam
from torchvision.transforms import ToTensor

#Start by choosing the dataset to draw from a built-in library
#CIFAR 10, CIFAR 100, EMNIST,F-MNIST, MNIST

SET_CHOICE = int(input("Please choose from the following datasets using the associated number:\n1 - CIFAR-10\n2 - CIFAR-100\n3-EMNIST\n4-F-MNIST\n5-MNIST"))

classes_set = [None]
root_set = [None]


train = datasets.FashionMNIST(root=root_set[SET_CHOICE], download=True, train=True, transform=ToTensor())
data = DataLoader(train)
classes = classes_set[SET_CHOICE]


class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1, 32, (3,3)),1
            nn.ReLU(),
        nn.Conv2d(32, 64,(3,3)),
        nn.ReLU(),
        nn.Conv2d(64, 64,(3,3)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*(22)*(22), 10),
    )
  def forward(self, x):
    return (self.model(x))

modl = NeuralNet().to('cuda')
optimizer = Adam(modl.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
iters = 1