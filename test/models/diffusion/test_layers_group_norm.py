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
from physicsnemo.models.diffusion.layers import GroupNorm, get_group_norm

script_path = os.path.abspath(__file__)
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

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x):
        return self.group_norm(x)


class GetGroupNormModule(physicsnemo.Module):
    """
    A wrapper around get_group_norm that has a factory method to create a model with
    reproducible random parameters.
    """

    _overridable_args: set[str] = {"use_apex_gn"}

    def __init__(self, arch_type: str = "GN_type_1", use_apex_gn: bool = False):
        super().__init__()
        C_in = 64
        if arch_type == "GN_type_1":
            self.group_norm = get_group_norm(
                num_channels=C_in,
                use_apex_gn=use_apex_gn,
            )
        elif arch_type == "GN_type_2":
            self.group_norm = get_group_norm(
                num_channels=C_in,
                num_groups=2,
                min_channels_per_group=16,
                eps=1e-3,
                use_apex_gn=use_apex_gn,
            )

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x):
        return self.group_norm(x)


def generate_data(device: str):
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W = 2, 64, 8, 16
    x: torch.Tensor = torch.randn(B, C_in, H, W).to(device)
    return x


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_group_norm_non_regression(device):
    """
    Test that GroupNorm can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """

    for test_params in (
        {"arch_type": "GN_type_1"},
        {"arch_type": "GN_type_2"},
    ):
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        model: GroupNormModule = GroupNormModule.factory(
            arch_type=test_params["arch_type"]
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "GN_type_1":
            assert model.group_norm.num_groups == 16, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-5, err_msg
        elif test_params["arch_type"] == "GN_type_2":
            assert model.group_norm.num_groups == 2, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-3, err_msg

        x: torch.Tensor = generate_data(device)
        out: torch.Tensor = model(x)

        assert common.validate_accuracy(
            out,
            file_name=f"output_diffusion_group_norm_{test_params['arch_type']}-v1.0.1.pth",
        ), err_msg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_get_group_norm_non_regression(device):
    """
    Test that get_group_norm can be instantiated and compare the output with a
    reference output generated with v1.0.1 (with the GroupNorm class).
    """

    TEST_PARAMS: Tuple[Dict[str, Any], ...] = (
        {"arch_type": "GN_type_1", "use_apex_gn": False},
        {"arch_type": "GN_type_2", "use_apex_gn": False},
    )
    if device == "cuda:0":
        TEST_PARAMS += (
            {"arch_type": "GN_type_1", "use_apex_gn": True},
            {"arch_type": "GN_type_2", "use_apex_gn": True},
        )

    for test_params in TEST_PARAMS:
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        model: GetGroupNormModule = GetGroupNormModule.factory(
            arch_type=test_params["arch_type"],
            use_apex_gn=test_params["use_apex_gn"],
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "GN_type_1":
            assert model.group_norm.num_groups == 16, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-5, err_msg
        elif test_params["arch_type"] == "GN_type_2":
            assert model.group_norm.num_groups == 2, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-3, err_msg

        x: torch.Tensor = generate_data(device)
        out: torch.Tensor = model(x)

        assert common.validate_accuracy(
            out,
            file_name=f"output_diffusion_group_norm_{test_params['arch_type']}-v1.0.1.pth",
        ), err_msg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_get_group_norm_non_regression_from_checkpoint(device):
    """
    Tests simple loading and non-regression of a checkpoint generated with the
    get_group_norm class. Also tests the API to override ``use_apex_gn`` to
    use Apex-based group norm when loading the checkpoint.
    """

    # TODO : currently only test overrding use_apex_gn from False to True. Need to
    # add test to do the opposite (that is load checkpoint with use_apex_gn=True and
    # override it to False)
    TEST_PARAMS: Tuple[Dict[str, Any], ...] = (
        {"arch_type": "GN_type_1", "chkpt_use_apex_gn": False, "use_apex_gn": False},
        {"arch_type": "GN_type_2", "chkpt_use_apex_gn": False, "use_apex_gn": False},
    )
    if device == "cuda:0":
        TEST_PARAMS += (
            {"arch_type": "GN_type_1", "chkpt_use_apex_gn": False, "use_apex_gn": True},
            {"arch_type": "GN_type_2", "chkpt_use_apex_gn": False, "use_apex_gn": True},
        )

    for test_params in TEST_PARAMS:
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        file_name: str = str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path(
                f"checkpoint_diffusion_get_group_norm_{test_params['arch_type']}_"
                f"use_apex_gn_{test_params['chkpt_use_apex_gn']}.mdlus"
            )
        )

        model: physicsnemo.Module = physicsnemo.Module.from_checkpoint(
            file_name=file_name,
            override_args={"use_apex_gn": test_params["use_apex_gn"]},
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "GN_type_1":
            assert model.group_norm.num_groups == 16, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-5, err_msg
        elif test_params["arch_type"] == "GN_type_2":
            assert model.group_norm.num_groups == 2, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-3, err_msg

        x: torch.Tensor = generate_data(device)
        out: torch.Tensor = model(x)

        assert common.validate_accuracy(
            out,
            file_name=f"output_diffusion_group_norm_{test_params['arch_type']}-v1.0.1.pth",
        ), err_msg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_get_group_norm_non_regression_from_group_norm_checkpoint(device):
    """
    Test that get_group_norm can be used to load a checkpoint generated with
    v1.0.1 and the GroupNorm class instead of get_group_norm. This also test
    the API to override ``use_apex_gn`` to use Apex-based group norm when
    loading the checkpoint.
    """

    TEST_PARAMS: Tuple[Dict[str, Any], ...] = (
        {"arch_type": "GN_type_1", "use_apex_gn": False},
        {"arch_type": "GN_type_2", "use_apex_gn": False},
    )
    if device == "cuda:0":
        TEST_PARAMS += (
            {"arch_type": "GN_type_1", "use_apex_gn": True},
            {"arch_type": "GN_type_2", "use_apex_gn": True},
        )

    for test_params in TEST_PARAMS:
        err_msg: str = (
            f"Failed with: {', '.join(f'{k}={v}' for k, v in test_params.items())}"
        )

        file_name: str = str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path(
                f"checkpoint_diffusion_group_norm_{test_params['arch_type']}-v1.0.1.mdlus"
            )
        )

        model: physicsnemo.Module = physicsnemo.Module.from_checkpoint(
            file_name=file_name,
            override_args={"use_apex_gn": test_params["use_apex_gn"]},
        ).to(device)

        # Check that the model is instantiated correctly
        if test_params["arch_type"] == "GN_type_1":
            assert model.group_norm.num_groups == 16, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-5, err_msg
        elif test_params["arch_type"] == "GN_type_2":
            assert model.group_norm.num_groups == 2, err_msg
            assert model.group_norm.weight.shape == (64,), err_msg
            assert model.group_norm.bias.shape == (64,), err_msg
            assert model.group_norm.eps == 1e-3, err_msg

        x: torch.Tensor = generate_data(device)
        out: torch.Tensor = model(x)

        assert common.validate_accuracy(
            out,
            file_name=f"output_diffusion_group_norm_{test_params['arch_type']}-v1.0.1.pth",
        ), err_msg
