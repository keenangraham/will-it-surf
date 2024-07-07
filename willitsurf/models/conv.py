import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)
        self.mp1 = nn.MaxPool2d(2)
        self.f1 = nn.Flatten()
        self.lsm1 = nn.LogSoftmax(dim=1)
        self.rl1 = nn.ReLU()

    def forward(self, x):
        print(x.shape)
        x = self.cn1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.cn2(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.mp1(x)
        print(x.shape)
        x = self.dp1(x)
        print(x.shape)
        x = self.f1(x)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.rl1(x)
        print(x.shape)
        x = self.dp2(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        x = self.lsm1(x)


def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred_prob = model(X)
        loss = F.nll_loss(pred_prob, y)
        loss.backward()
        optim.step()
        if b_i % 10 == 0:
            print(
                f'epoch: {epoch}, {b_i * len(X)}, {len(train_dataloader.dataset)},'
                f' {100.0 * b_i / len(train_dataloader)}, {loss.item()}'
            )

def test(model, device, test_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss += F.nll_loss(
                pred_prob,
                y,
                reduction='sum'
            ).item()
            pred = pred_prob.argmax(dim=1, keepdim=True)
            success += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(test_dataloader.dataset)
    print(
        f'Test dataset: Overall loss: {loss} '
        f'Overall accuracy: {success}/{len(test_dataloader.dataset)} '
        f'({100.0 * success / len(test_dataloader.dataset)})'
    )
