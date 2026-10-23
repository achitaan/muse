# okc: problem=muse_bf16_proj
"""Compute both permitted BF16 router components in one cuBLAS GEMM."""
import torch


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_packed_t"] = torch.cat((hi, lo), dim=0).T.contiguous()
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    n, layer = cfg["n"], cfg["layer"]
    if layer == "router":
        # Each half has its own FP32 GEMM accumulator. The output sum is FP32.
        products = torch.mm(x, state["router_packed_t"], out_dtype=torch.float32)
        full_n = state["router"].shape[0]
        return products[:, :n] + products[:, full_n:full_n+n]
    return torch.mm(x, state[layer][:n].T)
