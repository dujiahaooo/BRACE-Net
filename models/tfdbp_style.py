"""Independent, parameter-matchable [36]-style topology control.

This is not an official TF-DBPResNet reproduction.  It keeps BRACE-Full's
macro stages and head, and replaces only Stage 3--4 dual-stream slots with
the serial topology TFCA36(DBBR36(x)).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.bracenet import BCResBlock


class TFCA36(nn.Module):
    """Coordinate attention in the order documented by [36], Sec. 3.4."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.shared_projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True),
        )
        self.frequency_projection = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.time_projection = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, frequency_bins, _ = x.shape
        frequency_descriptor = x.mean(dim=3, keepdim=True)
        time_descriptor = x.mean(dim=2, keepdim=True).transpose(2, 3)
        embedding = self.shared_projection(
            torch.cat((frequency_descriptor, time_descriptor), dim=2)
        )
        frequency_embedding = embedding[:, :, :frequency_bins, :]
        time_embedding = embedding[:, :, frequency_bins:, :]
        frequency_gate = torch.sigmoid(self.frequency_projection(frequency_embedding))
        time_gate = torch.sigmoid(self.time_projection(time_embedding)).transpose(2, 3)
        return x * frequency_gate * time_gate


class _AxisBranch36(nn.Module):
    """One DBBR axis branch with projection, axis convolution and 1-D gate."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernel_size: tuple[int, int],
        modeled_axis: str,
    ):
        super().__init__()
        if modeled_axis not in {"time", "frequency"}:
            raise ValueError(f"unsupported modeled_axis: {modeled_axis}")
        self.modeled_axis = modeled_axis
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )
        padding = tuple((size - 1) // 2 for size in kernel_size)
        self.axis_convolution = nn.Sequential(
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size,
                padding=padding,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )
        # [36] specifies two independent Conv1D+BN paths but not their
        # kernel sizes.  Kernel 1 is the most conservative, least assumptive
        # choice; the axis context has already been modeled above.
        self.sigmoid_path = nn.Sequential(
            nn.Conv1d(branch_channels, branch_channels, 1, groups=branch_channels, bias=False),
            nn.BatchNorm1d(branch_channels),
        )
        self.tanh_path = nn.Sequential(
            nn.Conv1d(branch_channels, branch_channels, 1, groups=branch_channels, bias=False),
            nn.BatchNorm1d(branch_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        features = self.axis_convolution(projected)
        if self.modeled_axis == "time":
            compact = features.mean(dim=2)
            gate = torch.sigmoid(self.sigmoid_path(compact)) * torch.tanh(
                self.tanh_path(compact)
            )
            broadcast = gate.unsqueeze(2)
        else:
            compact = features.mean(dim=3)
            gate = torch.sigmoid(self.sigmoid_path(compact)) * torch.tanh(
                self.tanh_path(compact)
            )
            broadcast = gate.unsqueeze(3)
        return projected + broadcast


class DBBR36(nn.Module):
    """Parallel time/frequency branches followed by 1x1 fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        time_channels: int,
        frequency_channels: int,
        time_kernel: int,
        frequency_kernel: int,
    ):
        super().__init__()
        frequency_stride = stride[0]
        if in_channels != out_channels or frequency_stride != 1:
            self.input_projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=(frequency_stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.input_projection = nn.Identity()
        self.time_branch = _AxisBranch36(
            out_channels, time_channels, (1, time_kernel), "time"
        )
        self.frequency_branch = _AxisBranch36(
            out_channels, frequency_channels, (frequency_kernel, 1), "frequency"
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(time_channels + frequency_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_input = self.input_projection(x)
        # Both modules deliberately receive the same tensor.  Verification
        # hooks use this boundary to reject accidental serial rewrites.
        time_features = self.time_branch(shared_input)
        frequency_features = self.frequency_branch(shared_input)
        return self.fusion(torch.cat((time_features, frequency_features), dim=1))


class TFDBPStyleTopologyBlock(nn.Module):
    """The required serial relation TFCA36(DBBR36(x))."""

    def __init__(self, dbbr: DBBR36, tfca: TFCA36):
        super().__init__()
        self.dbbr = dbbr
        self.tfca = tfca

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tfca(self.dbbr(x))


class TFDBPStyleTopologyControl(nn.Module):
    """BRACE macro graph with Stage 3--4 slots replaced by [36]-style blocks."""

    blocks_per_stage = (2, 2, 4, 4)
    frequency_kernels = (5, 5, 7, 7)
    time_kernels = (9, 11, 13, 15)

    def __init__(
        self,
        base_c: int = 40,
        num_classes: int = 35,
        time_channels: Sequence[int] = (134, 120),
        frequency_channels: Sequence[int] = (16, 8),
        tfca_hidden_channels: Sequence[int] = (17, 20),
    ):
        super().__init__()
        if not (
            len(time_channels) == len(frequency_channels) == len(tfca_hidden_channels) == 2
        ):
            raise ValueError("Stage 3--4 width sequences must each contain two values")
        self.base_c = base_c
        self.num_classes = num_classes
        self.time_channels = tuple(time_channels)
        self.frequency_channels = tuple(frequency_channels)
        self.tfca_hidden_channels = tuple(tfca_hidden_channels)
        self.channels = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        stride_stages = {1, 2}

        self.stem = nn.Sequential(
            nn.Conv2d(1, self.channels[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for stage_index, block_count in enumerate(self.blocks_per_stage):
            blocks = nn.ModuleList()
            in_channels = self.channels[stage_index]
            out_channels = self.channels[stage_index + 1]
            for block_offset in range(block_count):
                stride = (
                    (2, 1)
                    if stage_index in stride_stages and block_offset == 0
                    else (1, 1)
                )
                if stage_index < 2:
                    block = BCResBlock(
                        in_channels,
                        out_channels,
                        stage_index,
                        stride,
                    )
                else:
                    deep_index = stage_index - 2
                    dbbr = DBBR36(
                        in_channels,
                        out_channels,
                        stride,
                        self.time_channels[deep_index],
                        self.frequency_channels[deep_index],
                        self.time_kernels[stage_index],
                        self.frequency_kernels[stage_index],
                    )
                    tfca = TFCA36(out_channels, self.tfca_hidden_channels[deep_index])
                    block = TFDBPStyleTopologyBlock(dbbr, tfca)
                blocks.append(block)
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
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channels[-1], num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(x.shape)}")
        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.classifier(x)
        return x.view(-1, x.shape[1])

