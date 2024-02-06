import torch
import torch.nn as nn

# Define the critic (discriminator) and generator networks
class Critic(nn.Module):
    def __init__(self, channels_img, features):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            # Input: N x channels_img x 32 x 32
            nn.Conv2d(channels_img, features, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.LeakyReLU(0.2),
            self._block(features, features * 2, 4, 2, 1),  # Output: 8x8
            self._block(features * 2, features * 4, 4, 2, 1),  # Output: 4x4
        )
        self.dis = nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=0)  # Output: 1x1
        self.classifier = nn.Linear(512*4, 10)
        self.softmax = nn.Softmax(dim=1)

        self.apply(self.init_weights)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.critic(x)
        class_x = x.view(-1, 32*4*4*4)
        return self.dis(x), self.softmax(self.classifier(class_x))
    def init_weights(self, m):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            init.zeros_(m.bias)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=110, out_features=384*4*4),
            nn.ReLU(inplace=True),

            Reshape((-1, 384, 4, 4)),

            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=96, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(self.init_weights)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)

    def init_weights(self, m):
      if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            init.zeros_(m.bias)

def test():
    N, in_channels, H, W = 100, 3, 32, 32
    x = torch.randn((N, in_channels, H, W))
    critic = Critic(in_channels, 32)
    assert critic(x)[0].shape == (N, 1, 1, 1)
    print(critic(x)[1].shape)
    assert critic(x)[1].shape == (N, 10)
    gen = Generator()
    noise_dim = 110
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")

test()