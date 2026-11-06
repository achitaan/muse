"""Exercise actual patched imports, phase gating and dormant fallback behavior."""
from inspect import Parameter, signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.muse_spark_v12 import native_marlin
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.models import muse_spark


def test_no_accepted_regime_allocates_nothing():
    assert not native_marlin.ACCEPTED_SERVING_SHAPES
    owner = native_marlin.ServingOwner()
    with patch.object(native_marlin, "NativeWorkspace", side_effect=AssertionError("unexpected allocation")):
        owner.bind(object())
    assert owner.workspace is None and owner.bound_bytes == 0


@pytest.mark.parametrize("method", [MuseSpark1xFp4Experts.forward,
                                    MuseSpark1xFp4Experts._forward_marlin,
                                    muse_spark.MuseSpark1xMoE._forward_experts])
def test_keyword_only_default_fallback(method):
    arg = signature(method).parameters["allow_decode_kernel"]
    assert arg.default is False and arg.kind == Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("phase", ["decode", "verify", "extend", "missing"])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("tp_ep", [(8, 8), (4, 4), (8, 4)])
def test_model_phase_and_parallel_gates(phase, sp, tp_ep):
    x = torch.empty((128, 8192), dtype=torch.bfloat16)
    call = MagicMock(return_value=x)
    instance = SimpleNamespace(finalize_ar_workspace=None, tp_size=tp_ep[0], ep_size=tp_ep[1],
                               moe_tp_size=1, _forward_experts=call,
                               _reduce_and_unpool=MagicMock(return_value=x))
    mode = SimpleNamespace(is_decode=lambda: phase == "decode",
                           is_target_verify=lambda: phase == "verify")
    batch = None if phase == "missing" else SimpleNamespace(forward_mode=mode)
    with patch.object(muse_spark, "get_forward", return_value=SimpleNamespace(sp_active=sp)):
        muse_spark.MuseSpark1xMoE._forward_tp(instance, x, forward_batch=batch)
    expected = phase in ("decode", "verify") and not sp and tp_ep == (8, 8)
    assert call.call_args.kwargs["allow_decode_kernel"] == expected


@pytest.mark.parametrize("tokens", [0, 1, 8, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257])
def test_dormant_dispatch_and_short_prefill(tokens):
    x = torch.empty((tokens, 4096), dtype=torch.bfloat16)
    ids = torch.zeros((tokens, 8), dtype=torch.int32)
    for flag in (False, True):
        assert native_marlin.serving_expert_outputs(object(), x, ids, allow_decode_kernel=flag) is None


@pytest.mark.parametrize("allow", [False, True])
def test_actual_marlin_method_keeps_stock_without_prepared_state(allow):
    layer = MuseSpark1xFp4Experts.__new__(MuseSpark1xFp4Experts)
    torch.nn.Module.__init__(layer)
    layer._native_marlin_state = None
    norm = object()
    layer._post_expert_norm = norm
    layer._post_expert_norm_weight = torch.ones(4096, dtype=torch.bfloat16)
    layer._post_expert_norm_eps = 1e-8
    layer._post_expert_norm_weight_first = True
    x = torch.empty((8, 4096), dtype=torch.bfloat16)
    z = torch.empty((8, 8, 4096), dtype=torch.bfloat16)
    ids = torch.zeros((8, 8), dtype=torch.int32)
    scores = torch.ones((8, 8)) / 8
    stock = MagicMock(return_value=(z, ids))
    layer._marlin_expert_outputs = stock
    import sglang.srt.layers.muse_spark_fp4_moe as module
    with patch.object(module, "fused_post_expert_norm_weighted_sum", return_value=x) as finalize:
        result = layer._forward_marlin(x, scores, ids, norm, allow_decode_kernel=allow)
    stock.assert_called_once_with(x, scores, ids)
    assert result is x
    assert finalize.call_args.args[0] is z
