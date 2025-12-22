import torch
import torch.nn as nn


# --------------------------------------------------
# Basic Conv Block
# --------------------------------------------------
class Basic_C2D_Block(nn.Module):
    def __init__(self, in_dim, out_dim, k_size, stride, is_BN):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=k_size, stride=stride, padding=k_size // 2
        )
        self.bn = nn.BatchNorm2d(out_dim) if is_BN else nn.Identity()
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# --------------------------------------------------
# Residual Block
# --------------------------------------------------
class Res_C2D_Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, stride=1):
        super().__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(
                Basic_C2D_Block(
                    in_dim if i == 0 else out_dim,
                    out_dim,
                    k_size=3,
                    stride=stride if i == 0 else 1,
                    is_BN=False,
                )
            )
        self.blocks = nn.Sequential(*layers)

        self.shortcut = None
        if in_dim != out_dim or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim),
            )

        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.act(self.blocks(x) + identity)


# --------------------------------------------------
# Actor-Critic Network (policy logits + value)
# --------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        channels, _, _ = input_shape  # supports frame stack (e.g., 4)

        self.conv1 = Basic_C2D_Block(channels, 32, k_size=8, stride=4, is_BN=False)
        self.res1 = Res_C2D_Block(32, 64, num_blocks=2, stride=2)
        self.res2 = Res_C2D_Block(64, 128, num_blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value
