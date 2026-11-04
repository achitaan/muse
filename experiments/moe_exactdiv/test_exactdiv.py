"""Compile/execute extracted index helpers on CPU and run source-isolation gates.

No Torch, CUDA compiler, GPU access, benchmark harness, or OKC submission.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_best/moe_m3_scored_2.3797.py"
CAND = HERE / "moe_exactdiv.py"


def sources(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in {"CUDA_SRC", "MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"}:
                values[name] = ast.literal_eval(node.value)
    return tree, values


def function(src, name):
    match = re.search(r"^(?:__device__|constexpr)[^\n]*\b" + re.escape(name) + r"\([^\n]*\)\s*\{", src, re.M)
    if not match:
        raise AssertionError(f"missing function {name}")
    start = src.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (src[end] == "{") - (src[end] == "}")
        end += 1
    return src[match.start():end]


def canonical(src):
    src = re.sub(r"/\*.*?\*/|//[^\n]*", "", src, flags=re.S)
    return re.sub(r"\s+", " ", src).strip()


def model_header(mutation=None):
    base, cand = sources(BASE)[1]["CUDA_SRC"], sources(CAND)[1]["CUDA_SRC"]
    parts = []
    for name in ("OKC_WGSA", "OKC_WGSB"):
        parts.append(re.search(r"^#define " + name + r" .*$", base, re.M)[0])
    for namespace, src in (("reference", base), ("candidate", cand)):
        begin = src.index("constexpr unsigned long long wgs_pack8(")
        end = src.index("#define OKC_WGSREORD", begin)
        parts += [f"namespace {namespace} {{", src[begin:end]]
        if namespace == "candidate":
            helper = function(src, "wgs_div_u32")
            if mutation == "no_corrections":
                helper = re.sub(r"  while \(product > n\).*\n", "", helper)
                helper = re.sub(r"  while \(product \+ d <= n\).*\n", "", helper)
            # Observation only: count correction iterations in the actual
            # extracted CUDA helper. No operand/result expression changes.
            helper = helper.replace("  uint32_t q =", "  ++division_calls;\n  unsigned corrections = 0;\n  uint32_t q =")
            helper = helper.replace("{ --q; product -= d; }", "{ ++corrections; --q; product -= d; }")
            helper = helper.replace("{ ++q; product += d; }", "{ ++corrections; ++q; product += d; }")
            helper = helper.replace("  return q;", "  correction_steps += corrections;\n  max_corrections = std::max(max_corrections, corrections);\n  return q;")
            parts.append(helper)
        for name in ("wgs_prefix", "wgs_stage_at", "wgs_range", "wgs_ranges", "wgs_cta_of"):
            text = function(src, name)
            if mutation == "inverse_boundary" and namespace == "candidate" and name == "wgs_cta_of":
                assert "xe * (uint32_t)G - 1u" in text
                text = text.replace("xe * (uint32_t)G - 1u", "xe * (uint32_t)G")
            parts.append(text)
        parts.append("}")
    return "\n".join(parts) + "\n"


class ExactDivTests(unittest.TestCase):
    def test_native_actual_helpers_and_mutations(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no native C++ compiler")
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            directory = Path(directory)
            for enabled, mutation in ((1, None), (0, None), (1, "no_corrections"), (1, "inverse_boundary")):
                (directory / "helpers.hpp").write_text(model_header(mutation))
                binary = directory / "model"
                build = subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off", f"-DOKC_EXACTDIV={enabled}", "-I", str(directory), str(HERE / "test_exactdiv.cpp"), "-o", str(binary)], text=True, capture_output=True)
                self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                if mutation:
                    self.assertNotEqual(result.returncode, 0, f"mutation {mutation} escaped")
                    self.assertIn("expected=", result.stderr)
                    print(f"MUTATION {mutation} rejected: {result.stderr.strip()}", flush=True)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("PASS EXACTDIV=", result.stdout)
                    print(result.stdout.strip(), flush=True)

    def test_only_scheduler_cuda_functions_change(self):
        a, b = sources(BASE)[1]["CUDA_SRC"], sources(CAND)[1]["CUDA_SRC"]
        self.assertNotIn("JOINFULL", b)
        touched = ("wgs_stage_at", "wgs_range", "wgs_ranges", "wgs_cta_of")
        for name in touched:
            a, b = a.replace(function(a, name), ""), b.replace(function(b, name), "")
        b = b.replace(function(b, "wgs_div_u32"), "")
        b = b.replace("#ifndef OKC_EXACTDIV\n#define OKC_EXACTDIV 1  // exact quotient from a fast FP32 estimate, scheduler indices only\n#endif\n", "")
        self.assertEqual(canonical(a), canonical(b))

    def test_host_ast_unchanged(self):
        base, cand = sources(BASE)[0], sources(CAND)[0]
        for tree in (base, cand):
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CUDA_SRC" for t in n.targets):
                    n.value = ast.Constant("CUDA source checked separately")
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    n.value = n.value.replace("v216x_pair1dq8_exactdiv_", "v216x_pair1dq8_")
        self.assertEqual(ast.dump(base, include_attributes=False), ast.dump(cand, include_attributes=False))

    def test_non_wgs_preprocessing_unchanged(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no C++ preprocessor")
        a, b = sources(BASE)[1], sources(CAND)[1]
        self.assertIn("#define OKC_FT_T 64", a["CUDA_SRC"])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            directory = Path(directory)
            for name in ("cuda_bf16.h", "cuda_runtime.h"):
                (directory / name).write_text("")
            for key in ("TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"):
                for prmt in (0, 1):
                    outputs = []
                    for values in (a, b):
                        text = re.sub(r"^#include [<\"](?:torch|ATen|c10)/.*$", "", values["CUDA_SRC"], flags=re.M)
                        proc = subprocess.run([compiler, "-E", "-P", "-x", "c++", "-I", str(directory), "-DOKC_NS=3", "-DOKC_NSF=3", f"-DOKC_PRMTSC={prmt}", *values[key], "-"], input=text, text=True, capture_output=True, check=True)
                        outputs.append(proc.stdout)
                    self.assertEqual(*outputs, (key, prmt))

    def test_main_control_prunes_general_division_paths(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no C++ preprocessor")
        values = sources(CAND)[1]
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            directory = Path(directory)
            for name in ("cuda_bf16.h", "cuda_runtime.h"):
                (directory / name).write_text("")
            text = re.sub(r"^#include [<\"](?:torch|ATen|c10)/.*$", "", values["CUDA_SRC"], flags=re.M)
            for enabled in (0, 1):
                proc = subprocess.run([compiler, "-E", "-P", "-x", "c++", "-I", str(directory), "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_PRMTSC=1", f"-DOKC_EXACTDIV={enabled}", *values["MAIN_DEFINES"], "-"], input=text, text=True, capture_output=True, check=True)
                divide = function(proc.stdout, "wgs_div_u32")
                owner = function(proc.stdout, "wgs_cta_of")
                self.assertEqual("return n / d;" in divide, not enabled)
                self.assertEqual("/ tot" in owner, not enabled)
                if enabled:
                    self.assertIn("xe * (uint32_t)G - 1u", owner)
                    self.assertNotIn(" / ", owner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
