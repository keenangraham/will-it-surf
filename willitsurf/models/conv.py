import torch

import torch.nn as nn

import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            k_size: int = 3,
            stride: int = 1,
            p: float = 0.10,
            use_maxpool: bool = True,
            use_batch_norm: bool = True,
            use_dropout: bool = False,
    ):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.conv = nn.Conv2d(d_in, d_out, k_size, stride)
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(2)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(d_out)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        x = self.conv(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(3, 8, k_size=9, stride=4),
                ConvBlock(8, 16, stride=2),
                ConvBlock(16, 32),
#               ConvBlock(32, 64),
#               ConvBlock(64, 128),
            ],
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3584, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.do1 = nn.Dropout(0.15)
        self.do2 = nn.Dropout(0.05)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.do2(x)
        x = self.fc3(x)
        return x


def train(model, device, train_dataloader, optim, epoch):
    model.train()
    for b_i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y = y.view(-1, 1).float()
        optim.zero_grad()
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=torch.tensor(4.5)
        )
        loss.backward()
        optim.step()
        if b_i % 10 == 0:
            print(
                f'epoch: {epoch}, {b_i * len(X)}, {len(train_dataloader.dataset)},'
                f' {100.0 * b_i / len(train_dataloader)}, {loss.item()}'
            )


def validate(model, device, val_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            y = y.view(-1, 1).float()
            logits = model(X)
            loss += F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=torch.tensor(4.5),
                reduction='sum',
            ).item()
            pred = (logits > 0).float()
            success += (pred == y).sum().item()
    loss /= len(val_dataloader.dataset)
    print(
        f'Val dataset: Overall loss: {loss} '
        f'Overall accuracy: {success}/{len(val_dataloader.dataset)} '
        f'({100.0 * success / len(val_dataloader.dataset)})'
    )


def test(model, device, test_dataloader):
    model.eval()
    loss = 0
    success = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y = y.view(-1, 1).float()
            logits = model(X)
            loss += F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=torch.tensor(4.5),
                reduction='sum'
            ).item()
            pred = (logits > 0).float()
            success += (pred == y).sum().item()
    loss /= len(test_dataloader.dataset)
    print(
        f'Test dataset: Overall loss: {loss} '
        f'Overall accuracy: {success}/{len(test_dataloader.dataset)} '
        f'({100.0 * success / len(test_dataloader.dataset)})'
    )


def make_train_dataloader(train_data):
    return torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )


def make_test_dataloader(test_data):
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=True
    )


def shape_given_input(b, c, w, h):
    x = torch.randn((b, c, w, h))

    def fw_hook(module, input, output):
        print(f'Shape of output to {module} is {output.shape}.')

    with torch.device("meta"):
        model = ConvNet()
        x = torch.randn((32, 3, w, h))

    for name, layer in model.named_modules():
        layer.register_forward_hook(fw_hook)

    model(x)
