"""Emit a CPU C++ signature contract for both graph API generations.

Pipe to clang++ -std=c++20 -fsyntax-only -x c++ - -DCUDART_VERSION=13000
(also 12080). Signatures are supplied by the judge report; this does not replace
compilation against NVIDIA headers, which are unavailable on this machine.
"""
import ast
from pathlib import Path
import re

tree = ast.parse(Path(__file__).with_name('decode_layer_abl.py').read_text())
src = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='_DA_CUDA' for t in n.targets))
guard = re.search(r'#if CUDART_VERSION >= 13000.*?#endif',src,re.S).group()
capture = re.search(r'ck\(DA_CAPTURE_INFO\(.*?\)\);',src,re.S).group()
edges = re.findall(r'ck\(DA_GRAPH_EDGES\(.*?\)\);',src,re.S)
assert len(edges)==2
print(r'''
#include <cstddef>
struct Stream; struct Graph; struct Node; struct EdgeData;
using cudaStream_t = Stream*;
using cudaGraph_t = Graph*;
using cudaGraphNode_t = Node*;
using cudaGraphEdgeData = EdgeData;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusActive };
using cudaError_t = int;
namespace at::cuda { cudaStream_t getCurrentCUDAStream(); }
void ck(cudaError_t);
#if CUDART_VERSION >= 13000
// Intentionally no suffixed names: reproduces the CUDA 13 failure mode.
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus*,
  unsigned long long*, cudaGraph_t*, const cudaGraphNode_t**,
  const cudaGraphEdgeData**, std::size_t*);
cudaError_t cudaGraphGetEdges(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*,
  cudaGraphEdgeData*, std::size_t*);
#else
cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t, cudaStreamCaptureStatus*,
  unsigned long long*, cudaGraph_t*, const cudaGraphNode_t**,
  const cudaGraphEdgeData**, std::size_t*);
cudaError_t cudaGraphGetEdges_v2(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*,
  cudaGraphEdgeData*, std::size_t*);
#endif
''')
print(guard)
print('void check() { cudaStreamCaptureStatus status; cudaGraph_t g = nullptr;')
print('std::size_t edges = 0, after = 0;')
print(capture)
print('\n'.join(edges))
print('}')
