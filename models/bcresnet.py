"""BC-ResNet baseline sharing the recovered BRACE macro architecture."""

from __future__ import annotations

import torch.nn as nn

from models.bracenet import BCResBlock


class BCResNet(nn.Module):
    """Standard BCResNet with no dual-stream wrappers or unused projections."""

    def __init__(self, base_c=64, num_classes=35):
        super().__init__()
        self.blocks_per_stage = [2, 2, 4, 4]
        self.channels = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.channels[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True),
        )
        self.stages = nn.ModuleList()
        for stage_index, num_blocks in enumerate(self.blocks_per_stage):
            blocks = nn.ModuleList()
            in_channels = self.channels[stage_index]
            out_channels = self.channels[stage_index + 1]
            for block_offset in range(num_blocks):
                stride = (
                    (2, 1)
                    if stage_index in {1, 2} and block_offset == 0
                    else (1, 1)
                )
                blocks.append(
                    BCResBlock(
                        in_channels,
                        out_channels,
                        stage_index,
                        stride,
                    )
                )
                in_channels = out_channels
            self.stages.append(blocks)
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.channels[-2],
                self.channels[-2],
                (5, 5),
                bias=False,
                groups=self.channels[-2],
                padding=(0, 2),
            ),
            nn.Conv2d(self.channels[-2], self.channels[-1], 1, bias=False),
            nn.BatchNorm2d(self.channels[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channels[-1], num_classes, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(x.shape)}")
        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.classifier(x).flatten(1)

