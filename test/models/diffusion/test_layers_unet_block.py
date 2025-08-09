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
from pathlib import Path
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
        for name, param in model.named_parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


# # For checkpoint generation with v1.0.1
# class UNetBlockModule(physicsnemo.Module):
#     """
#     A wrapper around UNetBlock with attention that has a factory method to
#     create a model with reproducible random parameters.
#     """

#     def __init__(
#         self,
#         arch_type: str = "UNetBlock_type_1",
#     ):
#         super().__init__()
#         C_in, Ne = 16, 8
#         C_out = C_in * 2
#         if arch_type == "UNetBlock_type_1":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#             )
#         elif arch_type == "UNetBlock_type_2":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#                 attention=True,
#                 num_heads=2,
#                 channels_per_head=16,
#             )
#         elif arch_type == "UNetBlock_type_3":
#             self.unet_block = UNetBlock(
#                 in_channels=C_in,
#                 out_channels=C_out,
#                 emb_channels=Ne,
#                 attention=True,
#                 channels_per_head=C_out,
#             )
#     factory: classmethod = classmethod(_instantiate_model)

#     def forward(self, x, emb):
#         return self.unet_block(x, emb)


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
        # Default parameters (no attention)
        if arch_type == "UNetBlock_type_1":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                use_apex_gn=use_apex_gn,
                fused_conv_bias=fused_conv_bias,
            )
        # Attention with 2 heads
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
        # Attention with single head and skip_scale != 1.0
        elif arch_type == "UNetBlock_type_3":
            self.unet_block = UNetBlock(
                in_channels=C_in,
                out_channels=C_out,
                emb_channels=Ne,
                attention=True,
                channels_per_head=C_out,
                skip_scale=0.5,
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


# For data with v1.0.1
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_block_generate_data(device):
    """
    Test that UNetBlock can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """

    TEST_PARAMS: Tuple[Dict[str, Any], ...] = (
        {"arch_type": "UNetBlock_type_1"},
        {"arch_type": "UNetBlock_type_2"},
        {"arch_type": "UNetBlock_type_3"},
    )

    for test_params in TEST_PARAMS:
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        model: UNetBlockModule = UNetBlockModule.factory(
            arch_type=test_params["arch_type"]
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "UNetBlock_type_1":
            assert model.unet_block.in_channels == 16, err_msg
            assert model.unet_block.out_channels == 32, err_msg
            assert model.unet_block.emb_channels == 8, err_msg
            # assert model.unet_block.attention is False, err_msg
            assert model.unet_block.num_heads == 0, err_msg
            assert model.unet_block.dropout == 0.0, err_msg
            assert model.unet_block.skip_scale == 1.0, err_msg
        elif test_params["arch_type"] == "UNetBlock_type_2":
            assert model.unet_block.in_channels == 16, err_msg
            assert model.unet_block.out_channels == 32, err_msg
            assert model.unet_block.emb_channels == 8, err_msg
            # assert model.unet_block.attention is True, err_msg
            assert model.unet_block.num_heads == 2, err_msg
            assert model.unet_block.dropout == 0.0, err_msg
            assert model.unet_block.skip_scale == 1.0, err_msg
        elif test_params["arch_type"] == "UNetBlock_type_3":
            assert model.unet_block.in_channels == 16, err_msg
            assert model.unet_block.out_channels == 32, err_msg
            assert model.unet_block.emb_channels == 8, err_msg
            # assert model.unet_block.attention is True, err_msg
            assert model.unet_block.num_heads == 1, err_msg
            assert model.unet_block.dropout == 0.0, err_msg
            assert model.unet_block.skip_scale == 0.5, err_msg

        model.save(
            f"checkpoint_diffusion_unet_block_{test_params['arch_type']}-v1.0.1.mdlus"
        )

        x, emb = generate_data(device)
        out: torch.Tensor = model(x, emb)

        torch.save(
            out, f"output_diffusion_unet_block_{test_params['arch_type']}-v1.0.1.pth"
        )


@pytest.mark.parametrize(
    ("device", "use_apex_gn"),
    [
        ("cuda:0", False),
        ("cuda:0", True),
        ("cpu", False),
    ],
    ids=["gpu", "gpu-apexgn", "cpu"],
)
@pytest.mark.parametrize("fused_conv_bias", [False, True], ids=["non_fused", "fused"])
@pytest.mark.parametrize(
    "arch_type",
    ["UNetBlock_type_1", "UNetBlock_type_2", "UNetBlock_type_3"],
    ids=["arch1", "arch2", "arch3"],
)
def test_unet_block_non_regression(arch_type, device, use_apex_gn, fused_conv_bias):
    """
    Test that UNetBlock can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """

    model: UNetBlockModule = UNetBlockModule.factory(
        arch_type=arch_type,
        use_apex_gn=use_apex_gn,
        fused_conv_bias=fused_conv_bias,
    ).to(device)

    # Check that the model is instantiated correctly
    if arch_type == "UNetBlock_type_1":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is False
        assert model.unet_block.num_heads == 0
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "UNetBlock_type_2":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 2
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 1.0
    elif arch_type == "UNetBlock_type_3":
        assert model.unet_block.in_channels == 16
        assert model.unet_block.out_channels == 32
        assert model.unet_block.emb_channels == 8
        assert model.unet_block.attention is True
        assert model.unet_block.num_heads == 1
        assert model.unet_block.dropout == 0.0
        assert model.unet_block.skip_scale == 0.5
    x, emb = generate_data(device)
    out: torch.Tensor = model(x, emb)

    # DEBUG
    # file_name: str = str(
    #     Path(__file__).parents[1].resolve()
    #     / Path("data")
    #     / Path(f"checkpoint_diffusion_unet_block_{test_params['arch_type']}-v1.0.1.mdlus")
    # )
    # model_ref: physicsnemo.Module = physicsnemo.Module.from_checkpoint(file_name=file_name)
    # out_ref: torch.Tensor = model_ref(x, emb)

    assert common.validate_accuracy(
        out,
        file_name=f"output_diffusion_unet_block_{arch_type}-v1.0.1.pth",
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_block_non_regression_from_checkpoint(device):
    """
    Tests simple loading and non-regression of a checkpoint generated with the
    UNetBlock class. Also tests the API to override ``use_apex_gn`` and
    ``fused_conv_bias`` when loading the checkpoint.
    """

    # TODO : currently only test overrding use_apex_gn from False to True. Need to
    # add test to do the opposite (that is load checkpoint with use_apex_gn=True and
    # override it to False)
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

        file_name: str = str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path(
                f"checkpoint_diffusion_unet_block_{test_params['arch_type']}-v1.0.1.mdlus"
            )
        )

        model: physicsnemo.Module = physicsnemo.Module.from_checkpoint(
            file_name=file_name,
            override_args={
                "use_apex_gn": test_params["use_apex_gn"],
                "fused_conv_bias": test_params["fused_conv_bias"],
            },
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
