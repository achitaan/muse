"""Experimental direct-Marlin WGS state. Explicitly prepared, serialized use only.

This module is a diagnostic artifact. No production path imports it until all
correctness, replay, performance and whole-server gates have passed.
"""
from pathlib import Path
import hashlib
import time


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


class NativeState:
    def __init__(self, layer, *, offset):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NativeState must be prepared before capture")
        self.ext = load_extension()
        self.offset = offset
        self.weights = []
        for name in ("w13", "w2"):
            self.weights.extend((getattr(layer, name + "_weight"),
                                 getattr(layer, name + "_weight_scale"),
                                 getattr(layer, name + "_weight_scale_2").float()))
        self.maps = self.ext.maps(self.weights)
        self.scratch = self.ext.workspace(layer.w13_weight)
        self.resources = self.ext.initialize(layer.w13_weight)
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in self.scratch) + self.maps.numel() + 256
        if self.workspace_bytes + 256 * 8 * 4096 * 2 > 256 * 1024**2:
            raise RuntimeError("workspace budget exceeded")

    def run(self, x, ids, out, *, local, debug, phases=2):
        self.ext.run(self.weights, self.scratch, self.maps, x, ids, out, local, debug,
                     self.offset, self.resources[0], phases)


def eligible(x, *, allow_decode_kernel=False, state=None, accepted=False):
    # Unaccepted experimental code cannot become an automatic serving dispatch.
    return (accepted and allow_decode_kernel and state is not None
            and 32 <= x.shape[0] <= 256 and x.shape[1] == 4096)
