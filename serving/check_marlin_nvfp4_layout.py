"""Check the layout oracle on CPU or against pinned stock SGLang on CUDA."""

import argparse
from pathlib import Path
import subprocess

from marlin_nvfp4_layout import (
    SGLANG_COMMIT, encode_scale_byte, scale_address, weight_address,
)


def check_cpu():
    # Hand-derived anchors from the repacker's lane/warp and nibble order.
    for k, n, expected in (
        (0, 0, (0, 0)), (8, 0, (0, 4)), (1, 0, (0, 16)),
        (0, 8, (0, 8)), (0, 16, (1, 0)), (2, 0, (4, 0)),
        (0, 64, (128, 0)), (16, 0, (256, 0)),
    ):
        assert weight_address(k, n, 128) == expected
    assert [scale_address(0, n, 128) for n in (0, 8, 16, 24, 1, 64)] == [
        0, 2, 1, 3, 8, 64
    ]
    assert scale_address(1, 0, 128) == 128
    assert [encode_scale_byte(x) for x in (0x00, 0x01, 0x38, 0x7E)] == [
        0x00, 0x68, 0xB0, 0xF6
    ]

    for size_k, size_n in ((16, 64), (48, 192)):
        nibbles = set()
        for k in range(size_k):
            for n in range(size_n):
                word, shift = weight_address(k, n, size_n)
                nibbles.add(8 * word + shift // 4)
        assert nibbles == set(range(size_k * size_n))
        scales = {
            scale_address(k_group, n, size_n)
            for k_group in range(size_k // 16)
            for n in range(size_n)
        }
        assert scales == set(range(size_k * size_n // 16))
    assert len({encode_scale_byte(x) for x in range(127)}) == 127
    print("CPU layout anchors, coverage and scale encoding passed.")


def verify_sglang_checkout(checkout, expected_patch=None):
    import sglang

    checkout = checkout.resolve()
    source_dir = checkout / "python" / "sglang"
    if not Path(sglang.__file__).resolve().is_relative_to(source_dir):
        raise RuntimeError(f"import SGLang from {source_dir} via PYTHONPATH")
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "python/sglang"],
        text=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SGLANG_COMMIT:
        raise RuntimeError(f"expected an unmodified SGLang {SGLANG_COMMIT} checkout")
    if expected_patch is None:
        if status:
            raise RuntimeError("stock checkout has changes")
        return
    actual_diff = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "HEAD", "--", "python/sglang"]
    )
    def patch_id(data):
        return subprocess.check_output(["git", "patch-id", "--stable"], input=data).split()[0]
    if patch_id(actual_diff) != patch_id(expected_patch.read_bytes()):
        raise RuntimeError("S1 checkout differs from the expected reviewed patch")
    if any(line.startswith("??") for line in status.splitlines()):
        raise RuntimeError("stage S1's new test file before verification")


def check_gpu(checkout):
    verify_sglang_checkout(checkout)
    import torch
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    from sglang.srt.layers.quantization.marlin_utils_fp4 import nvfp4_marlin_process_scales

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("run this check on a Hopper GPU")

    size_k, size_n = 64, 192
    labels = torch.arange(size_k * size_n, dtype=torch.int32).reshape(size_k, size_n)
    permutation = torch.empty(0, dtype=torch.int32, device="cuda")
    addresses = [
        weight_address(k, n, size_n) for k in range(size_k) for n in range(size_n)
    ]
    words = torch.tensor([word for word, _ in addresses])
    shifts = torch.tensor([shift for _, shift in addresses])
    # Four nibble planes encode a unique label for every logical (K, N).
    for plane in range(4):
        codes = (labels >> (4 * plane)) & 15
        packed = torch.zeros(size_k // 8, size_n, dtype=torch.int32)
        for lane in range(8):
            packed |= codes[lane::8] << (4 * lane)
        resident = gptq_marlin_repack(
            packed.cuda(), permutation, size_k, size_n, 4
        ).cpu().flatten()
        torch.testing.assert_close(
            (resident[words] >> shifts) & 15, codes.flatten().to(torch.int64), rtol=0, atol=0
        )

    raw = torch.arange(size_k * size_n // 16, dtype=torch.int32)
    raw = (raw % 127).to(torch.uint8).reshape(size_k // 16, size_n)
    permuted = marlin_permute_scales(
        raw.cuda().view(torch.float8_e4m3fn).to(torch.bfloat16),
        size_k, size_n, group_size=16,
    )
    resident = nvfp4_marlin_process_scales(permuted).view(torch.uint8).cpu().flatten()
    expected = torch.empty_like(resident)
    for k_group in range(size_k // 16):
        for n in range(size_n):
            expected[scale_address(k_group, n, size_n)] = encode_scale_byte(
                raw[k_group, n].item()
            )
    torch.testing.assert_close(resident, expected, rtol=0, atol=0)
    print(f"Stock CUDA repack and scale bytes matched on {torch.cuda.get_device_name()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-checkout", type=Path, help="also run the CUDA comparison")
    args = parser.parse_args()
    check_cpu()
    if args.sglang_checkout is not None:
        check_gpu(args.sglang_checkout)
