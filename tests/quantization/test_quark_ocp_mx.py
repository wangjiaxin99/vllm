# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.quark.schemes import quark_ocp_mx
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)


WEIGHT_MXFP4_SPEC = {"dtype": "fp4"}
INPUT_DYNAMIC_MXFP4_SPEC = {"dtype": "fp4", "is_dynamic": True}


def _vllm_config_with_linear_backend(linear_backend: str) -> VllmConfig:
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = linear_backend
    return vllm_config


def test_quark_ocp_mx_respects_linear_backend_emulation(monkeypatch):
    monkeypatch.setattr(quark_ocp_mx.current_platform, "supports_mx", lambda: True)
    init_kernel = Mock(side_effect=AssertionError("native kernel initialized"))
    monkeypatch.setattr(quark_ocp_mx, "init_mxfp4_linear_kernel", init_kernel)

    vllm_config = _vllm_config_with_linear_backend("emulation")
    with set_current_vllm_config(vllm_config):
        scheme = QuarkOCP_MX(WEIGHT_MXFP4_SPEC, INPUT_DYNAMIC_MXFP4_SPEC)

    assert scheme.emulate
    assert not hasattr(scheme, "ocp_mx_linear")
    init_kernel.assert_not_called()


def test_quark_ocp_mx_auto_uses_native_mxfp4_kernel(monkeypatch):
    monkeypatch.setattr(quark_ocp_mx.current_platform, "supports_mx", lambda: True)
    kernel = Mock()
    init_kernel = Mock(return_value=kernel)
    monkeypatch.setattr(quark_ocp_mx, "init_mxfp4_linear_kernel", init_kernel)

    vllm_config = _vllm_config_with_linear_backend("auto")
    with set_current_vllm_config(vllm_config):
        scheme = QuarkOCP_MX(WEIGHT_MXFP4_SPEC, INPUT_DYNAMIC_MXFP4_SPEC)

    assert not scheme.emulate
    assert scheme.ocp_mx_linear is kernel
    init_kernel.assert_called_once_with()
