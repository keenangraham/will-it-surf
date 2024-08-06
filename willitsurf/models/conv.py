import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 10, 4)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.10)
        self.dropout2 = nn.Dropout(0.25)
        self.fullyconnected1 = nn.Linear(327488, 64)
        self.fullyconnected2 = nn.Linear(64, 2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.flatten1 = nn.Flatten()
        self.logsoftmax1 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        logger.debug(f'input {x.shape}')
        x = self.conv1(x)
        logger.debug(f'after conv1 {x.shape}')
        x = F.relu(x)
        logger.debug(f'after relu1 {x.shape}')
        x = self.conv2(x)
        logger.debug(f'after conv2 {x.shape}')
        x = F.relu(x)
        logger.debug(f'after relu2 {x.shape}')
        x = self.maxpool1(x)
        logger.debug(f'after maxpool1 {x.shape}')
        x = self.dropout1(x)
        logger.debug(f'after dropout {x.shape}')
        x = self.flatten1(x)
        logger.debug(f'after flatten {x.shape}')
        x = self.fullyconnected1(x)
        logger.debug(f'after linear1 {x.shape}')
        x = F.relu(x)
        logger.debug(f'after relu3 {x.shape}')
        x = self.dropout2(x)
        logger.debug(f'after dropout2 {x.shape}')
        x = self.fullyconnected2(x)
        logger.debug(f'after linear2 {x.shape}')
        x = self.logsoftmax1(x)
        logger.debug(f'after logsoftmax {x.shape}')
        return x


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


'''
import torch; import numpy as np; import PIL; from willitsurf.models.conv import ConvNet
x = torch.from_numpy(np.array(PIL.Image.open('./assets/data/raw/slm-2024-07-09-15-36-31-image-2.png')).transpose(2, 0, 1))
d = (x / 255.0).unsqueeze(0).repeat(5, 1, 1, 1)
c = ConvNet()
c.forward(x)
'''

# Normalize train_X.mean() / 256.0, train_x.std() / 256.0
def make_train_dataloader(train_data):
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=True
    )


def make_test_dataloader(test_data):
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=True
    )

# torch.manual_seed(0)
# device = torce.device('cpu')
# model = ConvNet()
# optimizer = optim.Adadelta(model.parameters(), lr=0.5)

#for epoch in range(1, 3):
#    train(model, device, train_dataloader, optimizer, epoch)
#    trest(model, device, test_dataloader)

'''
from torch.utils.data import DataLoader
from torch.optim import Adadelta
from willitsurf.models.conv import ConvNet
from willitsurf.dataset import SurfImageDataset
from willitsurf.models.conv import train
s = SurfImageDataset('./assets/data/labels/annotations.tsv', './assets/data/raw')
d = DataLoader(s, batch_size=10, shuffle=True)
c = ConvNet()
c.to('mps')
a = Adadelta(c.parameters(), lr=0.5)
train(c, 'cpu', d, a, 1)
from willitsurf.models.conv import test
test(c, 'cpu', d)
'''
