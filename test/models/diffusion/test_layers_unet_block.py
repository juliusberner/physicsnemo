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

from pathlib import Path

import pytest
import torch

import physicsnemo
from physicsnemo.models.diffusion.layers import UNetBlock


class UNetBlockModuleWithAttention(physicsnemo.Module):
    """
    A wrapper around UNetBlock with attention that has a factory method to
    create a model with reproducible random parameters.
    """

    def __init__(
        self,
        use_apex_gn: bool = False,
        fused_conv_bias: bool = False,
    ):
        super().__init__()
        C_in, Ne = 16, 8
        C_out = C_in * 2
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

    def forward(self, x, emb):
        return self.unet_block(x, emb)

    @classmethod
    def factory(cls, seed: int = 0, **kwargs):
        """Create an instance with reproducible random parameters."""
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


def generate_data(device: str):
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W, Ne = 4, 16, 24, 24, 8
    x = torch.randn(B, C_in, H, W).to(device)
    emb = torch.randn(B, Ne).to(device)
    return x, emb


# TODO: add non-regression test against reference data
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_block_load_checkpoint(device):
    """
    Test that UNetBlock can be loaded from a checkpoint generated with v1.0.1.
    and test non-regression on the output.
    """
    torch.manual_seed(0)

    file_name = str(
        Path(__file__).parents[1].resolve()
        / Path("data")
        / Path("unet_block_with_attention-v1.0.1.mdlus")
    )

    # TODO: add test with use_apex_gn=True once it's fixed
    # model_0 generated with seed=0 from v1.0.1 checkpoint
    model_0 = UNetBlockModuleWithAttention.from_checkpoint(
        file_name=file_name,
    ).to(device)
    # model_1 generated with seed=0 from fresh model, should be the same as model_0
    model_1 = UNetBlockModuleWithAttention.factory(
        seed=0,
        use_apex_gn=False,
        fused_conv_bias=False,
    ).to(device)
    # model_1.save("unet_block_with_attention-v1.0.1.mdlus")
    # model_2 generated with seed=1 from fresh model, should be different from model_0
    model_2 = UNetBlockModuleWithAttention.factory(
        seed=1,
        use_apex_gn=False,
        fused_conv_bias=False,
    ).to(device)

    x, emb = generate_data(device)

    out_0 = model_0(x, emb)
    out_1 = model_1(x, emb)
    out_2 = model_2(x, emb)

    assert torch.allclose(out_0, out_1, atol=1e-4)
    assert not torch.allclose(out_0, out_2, atol=1e-4)

    # after loading the state_dict of model_0, model_2 should be the same as model_0
    model_2.load_state_dict(model_0.state_dict())
    out_2 = model_2(x, emb)
    assert torch.allclose(out_0, out_2, atol=1e-4)

    # TODO: add identical test with use_apex_gn=True once it's fixed
    # TODO: add test with fused_conv_bias=True once it's fixed (nor now not
    # working, expected behavior or not?)
