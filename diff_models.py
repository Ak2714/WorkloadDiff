import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AFF(nn.Module):
    # multi-feature fusion
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


# weight initialization
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv1d(in_channels, out_channels, kernel_size, padding, dilation):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config):
        super().__init__()
        inputdim = 1
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection_noise = Conv1d_with_init(inputdim, self.channels, 1)
        self.input_projection_conditioner = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.relu_cond = nn.ReLU()
        self.relu_noise = nn.ReLU()

        # Residual layer connection
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    dilation=2 ** (i % 10),
                    dilation_exp=i
                )
                for i in range(config["layers"])
            ]
        )


    def forward(self, x):
        # separation of conditions and predictions.
        conditioner, x = torch.split(x, 1, dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        conditioner = conditioner.reshape(B, inputdim, K * L)

        # Conv1x1
        x = self.input_projection_noise(x)
        conditioner = self.input_projection_conditioner(conditioner)

        # ReLU
        x = self.relu_noise(x)
        conditioner = self.relu_cond(conditioner)

        x = x.reshape(B, self.channels, K, L)
        skip = []

        for layer in self.residual_layers:
            x, skip_connection, conditioner = layer(x, conditioner)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = x.reshape(B, self.channels, K * L)
        # Conv1x1
        x = self.output_projection1(x)  # (B,channel,K*L)
        # ReLU
        x = F.relu(x)
        # Conv1x1
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class linear_layer(nn.Module):
    def __init__(self, input_dim=24, embed_dim=24):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


def add_conv(in_ch, out_ch, ksize, stride, leaky=True, lgt=24):
    stage = nn.Sequential()
    stage.add_module('linear', nn.Linear(lgt, lgt))
    stage.add_module('layer_norm', nn.LayerNorm(lgt))

    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class my_afpn(nn.Module):
    def __init__(self, feature_num, in_channels=64, out_channels=64):
        super().__init__()
        residual_channels = 64
        self.feature_num = feature_num
        self.dilated_conv_01 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)
        self.dilated_conv_02 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)
        self.dilated_conv_03 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)
        self.dilated_conv_04 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)
        self.dilated_conv_05 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)
        self.dilated_conv_06 = Conv1d(residual_channels, residual_channels, 3, padding=1, dilation=2)

        # dilation=2
        self.dilated_conv_transpose_04 = nn.ConvTranspose1d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=(3,), padding=(0,), stride=(1,))
        self.dilated_conv_transpose_05 = nn.ConvTranspose1d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=(5,), padding=(0,), stride=(1,))
        self.dilated_conv_transpose_06 = nn.ConvTranspose1d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=(7,), padding=(0,), stride=(1,))

        self.feature_fuse_01 = AFF(channels=residual_channels)
        self.feature_fuse_02 = AFF(channels=residual_channels)

        in_ch = 64
        out_ch = 64
        branch_num = 3
        pre_lgt = 24
        self.weight_level_0 = add_conv(in_ch, out_ch, 1, 1, leaky=False, lgt=pre_lgt)
        self.weight_level_1 = add_conv(in_ch, out_ch, 1, 1, leaky=False, lgt=pre_lgt)
        self.weight_level_2 = add_conv(in_ch, out_ch, 1, 1, leaky=False, lgt=pre_lgt)
        self.weight_levels = nn.Conv1d(in_ch * branch_num, branch_num, kernel_size=1, stride=1, padding=0)


    def forward(self, conditioner):
        cur_feature = conditioner[:, :, self.feature_num, :]

        # 1th layer
        y1 = self.dilated_conv_01(cur_feature)
        y2 = self.dilated_conv_02(cur_feature)
        y3 = self.dilated_conv_03(cur_feature)

        # 2th layer
        y2 = self.dilated_conv_04(y2)
        y3 = self.dilated_conv_05(y3)

        # 3th layer
        y3 = self.dilated_conv_06(y3)

        # 4th layer
        y1 = self.dilated_conv_transpose_04(y1)
        y2 = self.dilated_conv_transpose_05(y2)
        y3 = self.dilated_conv_transpose_06(y3)

        # 5th layer
        level_0_weight_v = self.weight_level_0(y1)
        level_1_weight_v = self.weight_level_1(y2)
        level_2_weight_v = self.weight_level_2(y3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        conditioner = y1 * levels_weight[:, 0:1, :] + y2 * levels_weight[:, 1:2, :] + y3 * levels_weight[:, 2:, :]
        return conditioner


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, dilation=None, dilation_exp=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        residual_channels = 64
        self.feature_fuse = AFF(channels=residual_channels)
        self.feature_fuse1 = AFF(channels=residual_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dilated_conv = Conv1d(residual_channels, residual_channels, 3, padding=dilation, dilation=dilation)
        self.dilated_conv_128 = Conv1d(residual_channels * 2, residual_channels * 2, 3, padding=dilation,
                                       dilation=dilation)

        feature_num = 4
        pre_lgt = 24
        hidden_size = feature_num * pre_lgt

        self.norm = torch.nn.LayerNorm(hidden_size)
        self.drop = torch.nn.Dropout(0.05)
        self.layer = torch.nn.Linear(1, 1)

        self.afpn_01 = my_afpn(feature_num=0)
        self.afpn_02 = my_afpn(feature_num=1)
        self.afpn_03 = my_afpn(feature_num=2)
        self.afpn_04 = my_afpn(feature_num=3)

        # feature fusion
        in_ch = 64
        out_ch = 64
        branch_num = 2
        self.weight_level_0 = add_conv(in_ch, out_ch, 1, 1, leaky=False, lgt=96)
        self.weight_level_1 = add_conv(in_ch, out_ch, 1, 1, leaky=False, lgt=96)
        self.weight_levels = nn.Conv1d(in_ch * branch_num, branch_num, kernel_size=1, stride=1, padding=0)

    def forward(self, x, conditioner):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        y = x
        y = self.dilated_conv(y)
        conditioner = conditioner.reshape(base_shape)

        # 1th feature
        y1 = self.afpn_01(conditioner)
        # 2th feature
        y2 = self.afpn_02(conditioner)
        # 3th feature
        y3 = self.afpn_03(conditioner)
        # 4th feature
        y4 = self.afpn_04(conditioner)

        conditioner = torch.cat([y1.unsqueeze(2), y2.unsqueeze(2), y3.unsqueeze(2), y4.unsqueeze(2)], dim=2).reshape(B, channel, K * L)
        # attention fusion
        y = self.drop(self.norm(self.feature_fuse(y, conditioner)))
        y = self.mid_projection(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)

        # output_projection
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        # Input to the next layer
        return (x + residual) / math.sqrt(2.0), skip, conditioner
