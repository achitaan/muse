"""Experimental direct-Marlin WGS state. Explicitly prepared, serialized use only.

This module is a diagnostic artifact. No production path imports it until all
correctness, replay, performance and whole-server gates have passed.
"""
from pathlib import Path
from functools import lru_cache
import hashlib


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256((root / "native_marlin_wgs.cu").read_bytes()
                            + (root / "native_marlin_tiles.cuh").read_bytes()).hexdigest()[:12]
    return load(
        name="muse_native_marlin_" + digest,
        sources=[str(root / "native_marlin_wgs.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                           "-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1",
                           "-DOKC_PRMTSC=1", "-DOKC_NS=3", "-DOKC_NSF=3", "--ptxas-options=-v"],
        extra_ldflags=["-lcuda"], verbose=True,
    )


class NativeWorkspace:
    """One owner per model/rank; all calls and graph replays must serialize.

    Keep this object alive while any captured graph references its storage.
    A second model or concurrent execution stream requires a separate owner;
    it is outside the 256-MiB single-model/rank configuration.
    """
    def __init__(self, exemplar):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare workspace before capture")
        self.ext = load_extension()
        self.scratch = self.ext.workspace(exemplar)
        self.resources = self.ext.initialize(exemplar)
        self.out = torch.empty((256, 8, 4096), device=exemplar.device, dtype=torch.bfloat16)
        self.local = torch.empty((256, 8), device=exemplar.device, dtype=torch.int32)
        self.empty_debug = torch.empty(0, device=exemplar.device, dtype=torch.bfloat16)
        self.bytes = sum(t.numel() * t.element_size()
                         for t in (*self.scratch, self.out, self.local))
        self.device = exemplar.device


class NativeState:
    def __init__(self, layer, *, offset, workspace=None):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NativeState must be prepared before capture")
        self.workspace = workspace or NativeWorkspace(layer.w13_weight)
        if self.workspace.device != layer.w13_weight.device:
            raise ValueError("shared workspace belongs to another device")
        self.ext = self.workspace.ext
        self.offset = offset
        self.weights = []
        for name in ("w13", "w2"):
            self.weights.extend((getattr(layer, name + "_weight"),
                                 getattr(layer, name + "_weight_scale"),
                                 getattr(layer, name + "_weight_scale_2").float()))
        self.maps = self.ext.maps(self.weights)
        self.scratch = self.workspace.scratch
        self.resources = self.workspace.resources
        self.workspace_bytes = self.workspace.bytes + self.maps.numel() + 256
        if self.workspace_bytes > 256 * 1024**2:
            raise RuntimeError("workspace budget exceeded")

    def run(self, x, ids, out, *, local, debug, phases=2):
        self.ext.run(self.weights, self.scratch, self.maps, x, ids, out, local, debug,
                     self.offset, self.resources[0], phases)


def eligible(x, *, allow_decode_kernel=False, state=None, accepted=False):
    # Unaccepted experimental code cannot become an automatic serving dispatch.
    return (accepted and allow_decode_kernel and state is not None
            and 32 <= x.shape[0] <= 256 and x.shape[1] == 4096)


# No regime has passed the serving-equivalence gate. Do not populate this from
# operator speed alone. In particular, T=32 regressed for representative routes.
ACCEPTED_SERVING_SHAPES = frozenset()


class ServingOwner:
    """Model-owned preparation hook. Empty acceptance set allocates nothing."""
    def __init__(self):
        self.workspace = None
        self.bound_bytes = 0

    def bind(self, layer):
        import torch
        if not ACCEPTED_SERVING_SHAPES:
            return
        if not (layer.use_marlin and not layer.expert_tp
                and layer.global_num_experts == 256 and layer.num_local_experts == 32
                and layer.top_k == 8 and layer.hidden_size == 4096
                and layer.intermediate_size_per_partition == 4096
                and layer.params_dtype == torch.bfloat16
                and torch.cuda.get_device_capability(layer.w13_weight.device) == (9, 0)):
            return
        if self.workspace is None:
            self.workspace = NativeWorkspace(layer.w13_weight)
        state = NativeState(layer, offset=layer.local_expert_offset, workspace=self.workspace)
        self.bound_bytes += state.maps.numel() + 256
        if self.workspace.bytes + self.bound_bytes > 256 * 1024**2:
            raise RuntimeError("per-rank workspace plus all layer descriptors exceeds cap")
        layer._native_marlin_state = state


def serving_expert_outputs(state, x, ids, *, allow_decode_kernel=False):
    import torch
    if not (allow_decode_kernel and state is not None and x.dim() == 2
            and x.shape[0] in ACCEPTED_SERVING_SHAPES and x.shape[1] == 4096
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and ids.dtype == torch.int32 and ids.is_contiguous()
            and ids.shape == (x.shape[0], 8)):
        return None
    workspace = state.workspace
    out, local = workspace.out[:x.shape[0]], workspace.local[:x.shape[0]]
    state.run(x, ids, out, local=local, debug=workspace.empty_debug)
    return out, local
