# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E402
import os
import sys

import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common

from physicsnemo.models.diffusion import SongUNetPosLtEmbd


def setup_model_learnable_embd(N_pos, lt_steps, lt_channels, img_resolution=128):
    # Smaller architecture variant with learnable positional embeddings
    # (more similar to CorrDiff example)
    model = SongUNetPosLtEmbd(
        img_resolution=img_resolution,
        in_channels=2 + N_pos,
        out_channels=2,
        model_channels=32,
        channel_mult_emb=2,
        gridtype="learnable",
        N_grid_channels=N_pos,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        use_apex_gn=True,
        amp_mode=True,
        prob_channels=[1],
    )
    return model


def setup_model_ddm_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=128):
    model = SongUNetPosLtEmbd(
        img_resolution=img_resolution,
        in_channels=2 + N_pos,
        out_channels=2,
        gridtype="test",
        N_grid_channels=N_pos,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        use_apex_gn=True,
        amp_mode=True,
        prob_channels=[1],
    )
    return model


def setup_model_ncsn_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=128):
    model = SongUNetPosLtEmbd(
        img_resolution=img_resolution,
        in_channels=2 + N_pos,
        out_channels=2,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
        gridtype="sinusoidal",
        N_grid_channels=N_pos,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        use_apex_gn=True,
        amp_mode=True,
        prob_channels=[1],
    )
    return model


# Test forward pass with AMP, Apex GN, and compile
@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_forward(device):
    torch.manual_seed(0)
    H, W = 32, 64
    offset_H, offset_W = 45, 12
    N_pos, lt_steps, lt_channels = 4, 3, 8
    model = (
        setup_model_learnable_embd(N_pos, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    # Compile model
    model = common.torch_compile_model(model)

    input_image = torch.ones([1, 2, H, W]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = None
    idx_H = torch.arange(offset_H, offset_H + H)
    idx_W = torch.arange(offset_W, offset_W + W)
    global_index = torch.stack(torch.meshgrid(idx_H, idx_W, indexing="ij"), dim=0)[
        None
    ].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels, global_index)
    assert output_image.shape == (1, 2, H, W)

    # TODO: add non-regression test
    return


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_global_indexing(device):
    torch.manual_seed(0)
    H, W = 32, 64
    offset_H, offset_W = 45, 12

    # Test with the DDM++ UNet model
    N_pos, lt_steps, lt_channels = 2, 3, 8
    model = (
        setup_model_ddm_plus_plus(N_pos, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    input_image = torch.ones([1, 2, H, W]).to(device)
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)
    idx_H = torch.arange(offset_H, offset_H + H)
    idx_W = torch.arange(offset_W, offset_W + W)
    global_index = torch.stack(torch.meshgrid(idx_H, idx_W, indexing="ij"), dim=0)[
        None
    ].to(device)

    pos_embed = model.positional_embedding_indexing(
        input_image, global_index, lead_time_label
    )
    assert pos_embed.shape == (1, N_pos + lt_channels, H, W)
    assert torch.equal(pos_embed[:, :N_pos, ...], global_index)
    assert torch.equal(pos_embed[:, N_pos:, ...], model.lt_embd[lead_time_label.int()])

    # Test with architecture variant with learnable positional embeddings
    # (more similar to CorrDiff example)
    N_pos, lt_steps, lt_channels = 4, 3, 8
    model = (
        setup_model_learnable_embd(N_pos, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    input_image = torch.ones([1, 2, H, W]).to(device)
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)
    idx_H = torch.arange(offset_H, offset_H + H)
    idx_W = torch.arange(offset_W, offset_W + W)
    global_index = torch.stack(torch.meshgrid(idx_H, idx_W, indexing="ij"), dim=0)[
        None
    ].to(device)

    pos_embed = model.positional_embedding_indexing(
        input_image, global_index, lead_time_label
    )
    assert pos_embed.shape == (1, N_pos + lt_channels, H, W)
    assert torch.equal(
        pos_embed[0, :N_pos, :, :],
        model.pos_embd[:, offset_H : offset_H + H, offset_W : offset_W + W],
    )
    assert torch.equal(pos_embed[:, N_pos:, :, :], model.lt_embd[lead_time_label.int()])


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_constructor(device):

    # Test DDM++ with square shape
    N_pos, lt_steps, lt_channels = 2, 3, 8
    H = W = 16
    model = (
        setup_model_ddm_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=H)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    input_image = torch.ones([1, 2, H, W]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = None
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels, lead_time_label)
    assert output_image.shape == (1, 2, H, W)

    # Test DDM++ with rectangular shape
    H, W = 16, 32
    model = (
        setup_model_ddm_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=[H, W])
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    noise_labels = torch.randn([1]).to(device)
    class_labels = None
    input_image = torch.ones([1, 2, H, W]).to(device)
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels, lead_time_label)
    assert output_image.shape == (1, 2, H, W)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_position_embedding(device):

    # Test with NCSN++ model
    H = W = 16
    N_pos, lt_steps, lt_channels = 100, 3, 8
    model = (
        setup_model_ncsn_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=H)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (N_pos, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)

    # Test with learnable positional embeddings
    H, W = 16, 32
    N_pos, lt_steps, lt_channels = 40, 3, 8
    model = (
        setup_model_learnable_embd(N_pos, lt_steps, lt_channels, img_resolution=[H, W])
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (N_pos, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_optims(device):
    """Test Song UNet optimizations"""

    def setup_model():
        H = W = 16
        N_pos, lt_steps, lt_channels = 100, 3, 8
        model = (
            setup_model_ncsn_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=H)
            .to(device)
            .to(memory_format=torch.channels_last)
        )
        input_image = torch.ones([1, 2, H, W]).to(device)
        noise_labels = torch.randn([1]).to(device)
        class_labels = None
        lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)
        return model, [input_image, noise_labels, class_labels, lead_time_label]

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))

    # Check JIT
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_checkpoint(device):
    """Test Song UNet checkpoint save/load"""
    H = W = 16
    N_pos, lt_steps, lt_channels = 40, 3, 8

    model_1 = (
        setup_model_learnable_embd(N_pos, lt_steps, lt_channels, img_resolution=H)
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    model_2 = (
        setup_model_learnable_embd(N_pos, lt_steps, lt_channels, img_resolution=H)
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    input_image = torch.ones([1, 2, H, W]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = None
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (*[input_image, noise_labels, class_labels, lead_time_label],),
        enable_autocast=True,
    )


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0"])
def test_son_unet_deploy(device):
    """Test Song UNet deployment support"""
    H = W = 16
    N_pos, lt_steps, lt_channels = 40, 3, 8
    model = (
        setup_model_ncsn_plus_plus(N_pos, lt_steps, lt_channels, img_resolution=H)
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    input_image = torch.ones([1, 2, H, W]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = None
    lead_time_label = torch.randint(0, lt_steps, (1,)).to(device)

    assert common.validate_onnx_export(
        model, (*[input_image, noise_labels, class_labels, lead_time_label],)
    )
    assert common.validate_onnx_runtime(
        model, (*[input_image, noise_labels, class_labels, lead_time_label],)
    )
