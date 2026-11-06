"""Build a separate, dormant experimental patch against pinned SGLang sources.

Supply a clean checkout at --sglang-checkout. Never edits that checkout.
"""
import argparse
import difflib
from pathlib import Path
import subprocess

PIN = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"


def edit_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def build(checkout, *, source_dir=None, output=None):
    here = Path(__file__).resolve().parent
    source_dir = source_dir or here
    root = "python/sglang/"
    files = {}
    def base(path):
        return subprocess.check_output(["git", "-C", str(checkout), "show", f"{PIN}:{path}"]).decode()

    path = root + "srt/layers/muse_spark_fp4_moe.py"
    original = base(path)
    text = edit_once(original, "        self.helion_verify_bank = None\n", """        self.helion_verify_bank = None
        self._native_marlin_state = None
        self._native_marlin_owner = None
""")
    marker = "    def _forward_marlin("
    start, end = text.index(marker), text.index("    def prepare_input(")
    part = text[start:end]
    part = edit_once(part, "        pre_expert_norm: Optional[nn.Module] = None,\n", """        pre_expert_norm: Optional[nn.Module] = None,
        *,
        allow_decode_kernel: bool = False,
""")
    part = edit_once(part, """        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids
        )
""", """        native = None
        if allow_decode_kernel and self._native_marlin_state is not None:
            from sglang.kernels.ops.muse_spark_v12.native_marlin import (
                serving_expert_outputs,
            )
            native = serving_expert_outputs(
                self._native_marlin_state, hidden_states, topk_ids,
                allow_decode_kernel=allow_decode_kernel,
            )
        if native is None:
            expert_outputs, local_topk_ids = self._marlin_expert_outputs(
                hidden_states, topk_scores, topk_ids
            )
        else:
            expert_outputs, local_topk_ids = native
""")
    text = text[:start] + part + text[end:]
    start = text.index("    def forward(")
    part = text[start:]
    part = edit_once(part, "        ar_workspace: FinalizePushWorkspace | None = None,\n", """        ar_workspace: FinalizePushWorkspace | None = None,
        allow_decode_kernel: bool = False,
""")
    part = edit_once(part, "                pre_expert_norm=pre_expert_norm,\n            )\n", """                pre_expert_norm=pre_expert_norm,
                allow_decode_kernel=allow_decode_kernel and ar_workspace is None,
            )
""")
    text = text[:start] + part
    files[path] = original, text

    path = root + "srt/models/muse_spark.py"
    original = base(path)
    text = edit_once(original, """        output = self._forward_experts(
            hidden_states,
            router_input,
            ar_workspace=ar_workspace,
        )
""", """        output = self._forward_experts(
            hidden_states,
            router_input,
            ar_workspace=ar_workspace,
            allow_decode_kernel=(
                forward_batch is not None
                and (forward_batch.forward_mode.is_decode()
                     or forward_batch.forward_mode.is_target_verify())
                and self.tp_size == 8 and self.ep_size == 8
                and self.moe_tp_size == 1 and not get_forward().sp_active
                and ar_workspace is None
            ),
        )
""")
    start, end = text.index("    def _forward_experts("), text.index("    def _forward_experts(") + 300
    part = edit_once(text[start:end], "        ar_workspace=None,\n", "        ar_workspace=None,\n        allow_decode_kernel: bool = False,\n")
    text = text[:start] + part + text[end:]
    text = edit_once(text, """                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
""", """                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
                allow_decode_kernel=allow_decode_kernel,
""")
    text = edit_once(text, """        side_stream = None if _is_cpu else torch.cuda.Stream()
        for layer_idx in range(config.num_hidden_layers):
""", """        side_stream = None if _is_cpu else torch.cuda.Stream()
        from sglang.kernels.ops.muse_spark_v12.native_marlin import ServingOwner
        self._native_marlin_owner = ServingOwner()
        for layer_idx in range(config.num_hidden_layers):
""")
    text = edit_once(text, """            if layer.is_moe_layer:
                layer.mlp.side_stream = side_stream
""", """            if layer.is_moe_layer:
                layer.mlp.side_stream = side_stream
                if layer.mlp.nvfp4_checkpoint:
                    layer.mlp.experts._native_marlin_owner = self._native_marlin_owner
""")
    files[path] = original, text

    path = root + "srt/layers/quantization/muse_spark_nvfp4_moe.py"
    original = base(path)
    text = edit_once(original, """            elif hasattr(layer, native):
                delattr(layer, native)
        return result
""", """            elif hasattr(layer, native):
                delattr(layer, native)
        # Bind only after stock has repacked and collapsed global scales.
        # The model-scoped owner allocates nothing while its accepted set is empty.
        owner = getattr(layer, "_native_marlin_owner", None)
        if owner is not None:
            owner.bind(layer)
        return result
""")
    files[path] = original, text
    for name in ("native_marlin.py", "native_marlin_wgs.cu", "native_marlin_tiles.cuh"):
        files[root + "kernels/ops/muse_spark_v12/" + name] = "", (source_dir / name).read_text()
    chunks = []
    for name, (old, new) in files.items():
        chunks.append(f"diff --git a/{name} b/{name}\n")
        if not old:
            chunks.append("new file mode 100644\n")
        chunks.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile="a/" + name if old else "/dev/null",
                                          tofile="b/" + name))
    output = output or here / "moe_direct_marlin.experimental.patch.diff"
    output.write_text("".join(chunks), newline="\n")
    print(output, output.stat().st_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-checkout", type=Path, required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    build(args.sglang_checkout, source_dir=args.source_dir, output=args.output)
