# FS build/logging follow-up

Checks391e3613/70380847 hid the useful diagnostic by logging only the exception
tail. The later user-run probe ddcd55fb exposed the actual failure:
`ATen/ATen.h:5:2: error: #error C++20 or later compatible compiler is required to use ATen`.
It was the CUDA TU's forced `-std=c++17`, not malformed host declarations.
The user already removed that flag in the delivered files and then obtained
successful FS execution in71a878e1/70fa067e.

This follow-up preserves that fix and synchronizes the generation templates,
which still contained the obsolete flag. The candidate and diagnostic now
print only error-containing diagnostic lines, excluding Torch's generic build
wrapper, truncated to600 characters. Missing diagnostics get a short explicit
message instead of a ninja command dump. fsdiag's embedded candidate is updated.

Host declarations and CUDA source are byte-identical to the GPU-tested version:
torch/extension.h, torch::Tensor, std::vector, int64_t and bool only, no CUDA
types in the host declarations, no duplicate inline PYBIND11_MODULE, and each
export declared exactly once with a matching CUDA definition.

Validation:
- `python -B .fs_offline/test_host_build.py`: passes runtime-flag, declaration,
  logging, fallback and embedding tests; g++ C++20 `-fsyntax-only` passes for
  generated main.cpp and binding.cpp using minimal stub headers.
- `python -B .fs_offline/test_diag.py`: passes three-TU linkage and timing mocks.
- `python -B .fs_offline/test_fs.py`: passes all previous CPU numerical/layout tests.
- Stub-header syntax checking is NOT a full Torch2.14 header/ABI or linked
  extension build. No GPU execution or git commands performed in this follow-up.

Current hashes are in manifest.json. fsdiag2.py is untouched.
