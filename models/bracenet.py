import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subspectralnorm import SubSpectralNorm


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_index,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        use_batch_norm=True,
        use_subspectral_norm=False,
    ):
        super().__init__()
        self.block_index = block_index

        def get_padding(kernel, dilated):
            dilation = 1
            padding = (kernel - 1) // 2
            if dilated and kernel > 1:
                dilation = int(2 ** self.block_index)
                padding = dilation * padding
            return padding, dilation

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            dilation = []
            for kernel in kernel_size:
                pad, dil = get_padding(kernel, use_dilation)
                padding.append(pad)
                dilation.append(dil)
        else:
            padding, dilation = get_padding(kernel_size, use_dilation)

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            )
        ]
        if use_subspectral_norm:
            layers.append(SubSpectralNorm(out_channels, 5))
        elif use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_index, stride):
        super().__init__()
        self.transition_block = in_channels != out_channels
        kernel_size = (3, 3)

        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_channels, out_channels, block_index, 1, 1))
            in_channels = out_channels

        layers.append(
            ConvBNReLU(
                in_channels,
                out_channels,
                block_index,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_channels,
                use_subspectral_norm=True,
                activation=False,
            )
        )
        self.frequency_branch = nn.Sequential(*layers)
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_branch = nn.Sequential(
            ConvBNReLU(
                out_channels,
                out_channels,
                block_index,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_channels,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        shortcut = x
        x = self.frequency_branch(x)
        pooled = x
        x = self.temporal_pool(x)
        x = self.temporal_branch(x)
        x = x + pooled
        if not self.transition_block:
            x = x + shortcut
        return F.relu(x, True)


class TACE2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True),
        )
        self.frequency_projection = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.time_projection = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def forward(self, x):
        _, _, freq_bins, _ = x.shape
        pooled_freq = x.mean(dim=3, keepdim=True)
        pooled_time = x.mean(dim=2, keepdim=True).transpose(2, 3)
        embedding = self.shared(torch.cat([pooled_freq, pooled_time], dim=2))
        attention_freq = torch.sigmoid(self.frequency_projection(embedding[:, :, :freq_bins, :]))
        attention_time = torch.sigmoid(self.time_projection(embedding[:, :, freq_bins:, :])).transpose(2, 3)
        return x * attention_freq * attention_time


class BCDualStreamBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_index,
        stride,
        use_dual=True,
        use_tfca=True,
        use_ssn=True,
        use_extra_res=True,
    ):
        super().__init__()
        self.use_dual = use_dual
        self.use_tfca = use_tfca
        self.use_extra_res = use_extra_res and use_dual

        freq_stride = stride[0] if isinstance(stride, (list, tuple)) else stride
        if in_channels != out_channels or freq_stride != 1:
            self.shortcut_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(freq_stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut_projection = nn.Identity()

        self.local_branch = BCResBlock(in_channels, out_channels, block_index, stride)

        if use_dual:
            if use_tfca:
                self.global_branch = TACE2D(out_channels)
            else:
                mid_channels = max(out_channels // 8, 8)
                self.global_branch = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, mid_channels, 1, bias=False),
                    nn.SiLU(True),
                    nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                    nn.Sigmoid(),
                )

            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(True),
            )
            if use_ssn:
                groups = 4 if out_channels % 4 == 0 else 2 if out_channels % 2 == 0 else 1
                self.post_norm = nn.GroupNorm(groups, out_channels)
            else:
                self.post_norm = nn.BatchNorm2d(out_channels)

            if self.use_extra_res:
                self.extra_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = self.shortcut_projection(x)
        local_features = self.local_branch(x)

        if not self.use_dual:
            return local_features

        if self.use_tfca:
            global_features = self.global_branch(shortcut)
        else:
            global_features = self.global_branch(shortcut) * shortcut

        fused = self.fusion(torch.cat([local_features, global_features], dim=1))
        fused = self.post_norm(fused)
        if self.use_extra_res:
            fused = fused + self.extra_scale * local_features
        return F.relu(fused + shortcut, inplace=True)


class BCDualNet(nn.Module):
    def __init__(
        self,
        base_c=40,
        num_classes=35,
        use_dual=True,
        use_tfca=True,
        use_ssn=True,
        use_extra_res=True,
        dual_start_stage=2,
    ):
        super().__init__()
        self.blocks_per_stage = [2, 2, 4, 4]
        self.channels = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        stride_stages = {1, 2}

        self.stem = nn.Sequential(
            nn.Conv2d(1, self.channels[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True),
        )

        self.stages = nn.ModuleList()
        for stage_index, num_blocks in enumerate(self.blocks_per_stage):
            use_stride = stage_index in stride_stages
            use_dual_here = use_dual and stage_index >= dual_start_stage
            blocks = nn.ModuleList()
            in_channels = self.channels[stage_index]
            out_channels = self.channels[stage_index + 1]
            for block_offset in range(num_blocks):
                stride = (2, 1) if use_stride and block_offset == 0 else (1, 1)
                blocks.append(
                    BCDualStreamBlock(
                        in_channels,
                        out_channels,
                        stage_index,
                        stride,
                        use_dual=use_dual_here,
                        use_tfca=use_tfca,
                        use_ssn=use_ssn,
                        use_extra_res=use_extra_res,
                    )
                )
                in_channels = out_channels
            self.stages.append(blocks)

        self.classifier = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-2], (5, 5), bias=False, groups=self.channels[-2], padding=(0, 2)),
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
            raise ValueError(f'Expected a 3D or 4D tensor, got shape {tuple(x.shape)}')

        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.classifier(x)
        return x.view(-1, x.shape[1])
