"""Freeze one single-file Decode tail candidate, without importing or running it."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
BASE = '307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'
WORK = ROOT / '.tools/decoder-moe-tail-20261029'


def main():
    original = (ROOT / 'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert hashlib.sha256(original).hexdigest() == BASE
    text = original.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
    root = ast.get_source_segment(text, node)
    begin = root.index(' expert_output = _experts(')
    end = root.index(' return (next_attn_input, stream, topk_ids)', begin)
    old = root[begin:end]
    new = (" if _dt_eligible(expert_input, stream, state):\n"
           "  next_attn_input, stream, _, _ = _dt_call(expert_input, topk_weights, local_ids, stream, state)\n"
           " else:\n" + ''.join(' ' + line if line.strip() else line for line in old.splitlines(keepends=True)))
    assert text.count(old) == 1
    prefix = text.replace(old, new)
    assert prefix.replace(new, old) == text
    cuda = (WORK / 'prototype.cu').read_bytes()
    native_binding = json.loads((WORK / 'binding.json').read_bytes())
    sha = hashlib.sha256(cuda).hexdigest()
    assert sha == native_binding['prototype_CUDA_sha256']
    signature = re.search(r'void okc_run_tail\([^{}]+\) \{', cuda.decode())[0][:-2]
    cpp = '#include <torch/extension.h>\n' + signature + ';\nstd::vector<int64_t> okc_tail_resources();\n'
    addon = (ROOT / 'experiments/prepared/decoder_moe_tail_addon.py').read_text()
    addon = addon.replace('SOURCE_TAG', sha[:12]).replace('CUDA_TAG', sha)
    payload = base64.b64encode(zlib.compress(cuda, 9)).decode()
    constants = '\n_DT_CUDA = _zsrc(' + repr(payload) + ')\n_DT_CPP = ' + repr(cpp) + '\n'
    frozen = (prefix + constants + addon).encode()
    tree = ast.parse(frozen)
    compile(tree, '<frozen_decode_tail>', 'exec')
    assert len(frozen) < 1_000_000
    final_sha = hashlib.sha256(frozen).hexdigest()
    (WORK / 'submission.py').write_bytes(frozen)
    binding = dict(parent_decode_sha256=native_binding['parent_decode_sha256'],
                   packaged_parent_sha256=BASE, source_sha256=final_sha, source_bytes=len(frozen),
                   CUDA_sha256=sha, CPP_sha256=hashlib.sha256(cpp.encode()).hexdigest(),
                   addon_template_sha256=hashlib.sha256((ROOT / 'experiments/prepared/decoder_moe_tail_addon.py').read_bytes()).hexdigest(),
                   root_old=old, root_new=new, exact_root_inverse=True,
                   complete_other_packaged_source_preserved=True,
                   launch_arguments=10, tail_tensor_arguments=9, query_values=28,
                   changed_tokens=[1,8], supported_tiny_neighbors=[2,3,4,5,6,7],
                   unchanged_fallback='All >8, unsupported norm/cache/layout/device and PRMT-ineligible paths',
                   original_dense_and_all_expert_arithmetic_retained=True,
                   added_BF16_weight_layout_bytes=1024*4096*2,
                   GPU_admitted=False, GPU_jobs=0,
                   pending=['host ABI CPU audit','fresh official identities','resource SASS review',
                            'complete result/guard adversarial fixtures','frozen launch driver'])
    (WORK / 'submission-binding.json').write_bytes((json.dumps(binding,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in binding.items() if k not in ('root_old','root_new')},indent=2))


if __name__ == '__main__':
    main()
