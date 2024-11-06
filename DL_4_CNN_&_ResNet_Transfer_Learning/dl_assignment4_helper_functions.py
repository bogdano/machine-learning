import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim


def trainMyModel(net, lr, trainloader, n_epochs):
  # Attempt to put your neural network onto the GPU. Do not worry if there is no
  # GPU available on CoLab; for models of our size it won't be prohibitively
  # slow and you can proceed with a CPU
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"Training on {device}")
  net.to(device)
  for epoch in range(n_epochs):
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # store running loss
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # print statistics
      running_loss += loss.item()
      if i % 100 == 99:    # print every 100 mini-batches
        print('[epoch: %d, batch: %3d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0

  print('✨ Finished Training ✨')
  return net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def testMyModel(trainedNet, testloader):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = trainedNet(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  acc = 100 * correct / total
  
  print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {acc} %')
  return acc