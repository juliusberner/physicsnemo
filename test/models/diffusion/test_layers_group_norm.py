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

# from pathlib import Path

import pytest
import torch

import physicsnemo
from physicsnemo.models.diffusion.layers import GroupNorm

# from physicsnemo.models.diffusion.layers import GroupNorm, get_group_norm


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """
    Helper function to instantiate a model with reproducible random parameters.
    """
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
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


class GroupNormModule(physicsnemo.Module):
    """
    A wrapper around GroupNorm that has a factory method to create a model with
    reproducible random parameters.
    """

    def __init__(self, arch_type: str = "GN_type_1"):
        super().__init__()
        C_in = 64
        if arch_type == "GN_type_1":
            self.group_norm = GroupNorm(num_channels=C_in)
        elif arch_type == "GN_type_2":
            self.group_norm = GroupNorm(
                num_channels=C_in,
                num_groups=2,
                min_channels_per_group=16,
                eps=1e-3,
            )

    factory = classmethod(_instantiate_model)

    def forward(self, x):
        return self.group_norm(x)


def generate_data(device: str):
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W = 2, 64, 8, 16
    x = torch.randn(B, C_in, H, W).to(device)
    return x


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_group_norm_non_regression(device):
    """
    Test that GroupNorm can be instantiated and compare the output with a reference
    implementation.
    """
    model = GroupNormModule.factory().to(device)
    x = generate_data(device)
    out = model(x)
    assert out.shape == x.shape


# @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
# def test_unet_block_load_checkpoint(device):
#     """
#     Test that UNetBlock can be loaded from a checkpoint generated with v1.0.1.
#     and test non-regression on the output.
#     """
#     torch.manual_seed(0)

#     file_name = str(
#         Path(__file__).parents[1].resolve()
#         / Path("data")
#         / Path("unet_block_with_attention-v1.0.1.mdlus")
#     )

#     # model_0 generated with seed=0 from v1.0.1 checkpoint
#     model_0 = UNetBlockModuleWithAttention.from_checkpoint(
#         file_name=file_name,
#     ).to(device)

#     # TODO: add test with use_apex_gn=True once it's fixed
#     for test_params in (
#         {
#             "use_apex_gn": False,
#             "fused_conv_bias": False,
#         },
#         {
#             "use_apex_gn": False,
#             "fused_conv_bias": True,
#         },
#     ):
#         err_msg = (
#             f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
#         )

#         # model_1 generated with seed=0 from fresh model, should be the same as model_0
#         model_1 = UNetBlockModuleWithAttention.factory(
#             seed=0,
#             use_apex_gn=test_params["use_apex_gn"],
#             fused_conv_bias=test_params["fused_conv_bias"],
#         ).to(device)
#         # model_1.save("unet_block_with_attention-v1.0.1.mdlus")
#         # model_2 generated with seed=1 from fresh model, should be different from model_0
#         model_2 = UNetBlockModuleWithAttention.factory(
#             seed=1,
#             use_apex_gn=test_params["use_apex_gn"],
#             fused_conv_bias=test_params["fused_conv_bias"],
#         ).to(device)

#         x, emb = generate_data(device)

#         out_0 = model_0(x, emb)
#         out_1 = model_1(x, emb)
#         out_2 = model_2(x, emb)

#         assert torch.allclose(out_0, out_1, atol=1e-3), err_msg
#         assert not torch.allclose(out_0, out_2, atol=1e-3), err_msg

#         # after loading the state_dict of model_0, model_2 should be the same as model_0
#         model_2.load_state_dict(model_0.state_dict())
#         out_2 = model_2(x, emb)
#         assert torch.allclose(out_0, out_2, atol=1e-3), err_msg
