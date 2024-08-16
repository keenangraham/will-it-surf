import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(3, 6, 10, 4)
        self.cn2 = nn.Conv2d(6, 12, 5, 2)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        logger.debug(f'input {x.shape}')
        x = F.relu(self.cn1(x))
        logger.debug(f'after cn1 {x.shape}')
        x = F.max_pool2d(x, (2, 2))
        logger.debug(f'after maxpool1 {x.shape}')
        x = F.relu(self.cn2(x))
        logger.debug(f'after cn2 {x.shape}')
        x = F.max_pool2d(x, (2, 2))
        logger.debug(f'after maxpool2 {x.shape}')
        x = torch.flatten(x, 1)
        logger.debug(f'after flatten {x.shape}')
        x = F.relu(self.fc1(x))
        logger.debug(f'after fc1 {x.shape}')
        x = F.relu(self.fc2(x))
        logger.debug(f'after fc2 {x.shape}')
        x = self.fc3(x)
        logger.debug(f'after fc3 {x.shape}')
        return x


def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred_prob = model(X)
        loss = F.cross_entropy(pred_prob, y)
        loss.backward()
        optim.step()
        if b_i % 10 == 0:
            print(
                f'epoch: {epoch}, {b_i * len(X)}, {len(train_dataloader.dataset)},'
                f' {100.0 * b_i / len(train_dataloader)}, {loss.item()}'
            )


def validate(model, device, val_dataloader):
    model.eval()
    success = 0
    counter = 0
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            _, pred = torch.max(pred_prob, 1)
            counter += y.size(0)
            success += (pred == y).sum().item()
    print(f'Accuracy on val dataset: {(100 * success / counter)}')


def test(model, device, test_dataloader):
    model.eval()
    success = 0
    counter = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            _, pred = torch.max(pred_prob, 1)
            counter += y.size(0)
            success += (pred == y).sum().item()
    print(f'Accuracy on test dataset: {(100 * success / counter)}')
