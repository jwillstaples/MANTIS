import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time

from chess_standard.board_chess_pypi import OUTPUT_LENGTH


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.backBone = nn.ModuleList([ResBlock(64) for i in range(3)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, OUTPUT_LENGTH),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 64, 1),
            nn.Tanh(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return F.softmax(policy, dim=1), value

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                init.zeros_(m.bias)


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


def test():
    print("start")
    N, ch, R, C = 10, 3, 8, 8
    rand = torch.randn((N, ch, R, C))
    net = ChessNet()
    p, v = net(rand)
    assert p.shape == torch.Size([N, OUTPUT_LENGTH])
    assert v.shape == torch.Size([N, 1])
    print("Works!")


def test2():
    print("Start")
    times = 10

    st = time.time()
    for _ in range(times):
        N, ch, R, C = 1, 3, 8, 8
        rand = torch.randn((N, ch, R, C))
        net = ChessNet()
        p, v = net(rand)
    print(f"time: {time.time() - st}")

    st = time.time()
    times *= 10
    for _ in range(times):
        N, ch, R, C = 1, 3, 8, 8
        rand = torch.randn((N, ch, R, C))
        net = ChessNet()
        p, v = net(rand)
    print(f"time: {time.time() - st}")

if __name__ == "__main__":
    test()