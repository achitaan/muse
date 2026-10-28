
# Local experiment: map global top-k IDs to this rank's 32 experts in one launch.
import triton
import triton.language as tl


@triton.jit
def _achita_local_ids_kernel(IDS, START, OUT, COUNT: tl.constexpr,
                              LOCAL_COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + offsets, mask=offsets < COUNT, other=0)
    start = tl.load(START)
    local = ids - start
    local = tl.where((local >= 0) & (local < LOCAL_COUNT), local, -1)
    tl.store(OUT + offsets, local, mask=offsets < COUNT)


def _achita_map_local(ids, local_start):
    result = torch.empty_like(ids)
    _achita_local_ids_kernel[(triton.cdiv(ids.numel(), 256),)](
        ids, local_start, result, ids.numel(), LOCAL_EXPERTS, 256, num_warps=4)
    return result
