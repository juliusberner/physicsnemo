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

import os
import sys
from typing import Any, Dict, Tuple

import pytest
import torch

import physicsnemo
from physicsnemo.models.diffusion.layers import UNetBlock

script_path: str = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common  # noqa: E402


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """
    Helper function to instantiate a model with reproducible random parameters.
    """
    model: physicsnemo.Module = cls(**kwargs)
    gen: torch.Generator = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


class UNetBlockModule(physicsnemo.Module):
    """
    A wrapper around UNetBlock with attention that has a factory method to
    create a model with reproducible random parameters.
    """

    _overridable_args: set[str] = {"use_apex_gn", "fused_conv_bias"}

    def __init__(
        self,
        arch_type: str = "UNetBlock_type_1",
        use_apex_gn: bool = False,
        fused_conv_bias: bool = False,
    ):
        super().__init__()
        C_in, Ne = 16, 8
        C_out = C_in * 2
        if arch_type == "UNetBlock_type_1":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        elif arch_type == "UNetBlock_type_2":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                attention=True,
                num_heads=2,
                channels_per_head=16,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x, emb):
        return self.unet_block(x, emb)


def generate_data(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W, Ne = 4, 16, 24, 24, 8
    x: torch.Tensor = torch.randn(B, C_in, H, W).to(device)
    emb: torch.Tensor = torch.randn(B, Ne).to(device)
    return x, emb


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_block_non_regression(device):
    """
    Test that UNetBlock can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """

    TEST_PARAMS: Tuple[Dict[str, Any], ...] = (
        {
            "arch_type": "UNetBlock_type_1",
            "use_apex_gn": False,
            "fused_conv_bias": False,
        },
        {
            "arch_type": "UNetBlock_type_2",
            "use_apex_gn": False,
            "fused_conv_bias": False,
        },
        {
            "arch_type": "UNetBlock_type_1",
            "use_apex_gn": False,
            "fused_conv_bias": True,
        },
        {
            "arch_type": "UNetBlock_type_2",
            "use_apex_gn": False,
            "fused_conv_bias": True,
        },
    )
    if device == "cuda:0":
        TEST_PARAMS += (
            {
                "arch_type": "UNetBlock_type_1",
                "use_apex_gn": True,
                "fused_conv_bias": True,
            },
            {
                "arch_type": "UNetBlock_type_2",
                "use_apex_gn": True,
                "fused_conv_bias": True,
            },
        )

    for test_params in TEST_PARAMS:
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        model: UNetBlockModule = UNetBlockModule.factory(
            arch_type=test_params["arch_type"],
            use_apex_gn=test_params["use_apex_gn"],
            fused_conv_bias=test_params["fused_conv_bias"],
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "UNetBlock_type_1":
            assert model.unet_block.in_channels == 16, err_msg
            assert model.unet_block.out_channels == 32, err_msg
            assert model.unet_block.emb_channels == 8, err_msg
            assert model.unet_block.attention is False, err_msg
            assert model.unet_block.num_heads == 0, err_msg
            assert model.unet_block.dropout == 0.0, err_msg
            assert model.unet_block.skip_scale == 1.0, err_msg
        elif test_params["arch_type"] == "UNetBlock_type_2":
            assert model.unet_block.in_channels == 16, err_msg
            assert model.unet_block.out_channels == 32, err_msg
            assert model.unet_block.emb_channels == 8, err_msg
            assert model.unet_block.attention is True, err_msg
            assert model.unet_block.num_heads == 2, err_msg
            assert model.unet_block.dropout == 0.0, err_msg
            assert model.unet_block.skip_scale == 1.0, err_msg

        x, emb = generate_data(device)
        out: torch.Tensor = model(x, emb)

        assert common.validate_accuracy(
            out,
            file_name=f"output_diffusion_unet_block_{test_params['arch_type']}-v1.0.1.pth",
        ), err_msg
