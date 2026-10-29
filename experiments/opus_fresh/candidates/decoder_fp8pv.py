# okc: problem=muse_decode_layer
# Vjeux okc-2026, Apache-2.0, f35a41ee1c13e316dabbf772830d47b23b453393.
# NathenFernandes/muse-bf16-proj, MIT, 7385e5d8f70980cfd5322ea254edc0b2b8ca90f7.
# Local changes: integration/cache lifetime, strict BF16 MoE/attention boundaries,
# fused pooler/router only at 9..32 tokens. Original FP32 small-token router
# retained; attention fallback restores BF16 boundaries. Sequential scratch replay.
# Python metadata compacted for 1 MB; attention CUDA restores BF16 boundaries.
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# 
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
#    1. Definitions.
# 
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
# 
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
# 
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
# 
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
# 
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
# 
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
# 
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
# 
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
# 
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
# 
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
# 
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
# 
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
# 
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
# 
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
# 
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
# 
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
# 
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
# 
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
# 
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
# 
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
# 
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
# 
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
# 
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
# 
#    END OF TERMS AND CONDITIONS
# 
#    APPENDIX: How to apply the Apache License to your work.
# 
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
# 
#    Copyright [yyyy] [name of copyright owner]
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 
# 
# MIT License
# 
# Copyright (c) 2026 Nathen Fernandes
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# 
# Additional local change: full FP32 router/top-8 fusion for 1..8 tokens; no activation or weight quantization.
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

class _native_moe_Weights(TypedDict):
 w13: torch.Tensor
 w13_sf: torch.Tensor
 w13_gs: torch.Tensor
 w2: torch.Tensor
 w2_sf: torch.Tensor
 w2_gs: torch.Tensor
 post_norm_w: torch.Tensor

class _native_moe_Config(TypedDict):
 bs: int
 d_hidden: int
 d_expert: int
 n_experts: int
 local_start: int
 local_count: int
 top_k: int
 sharding: str

class _native_moe_Inputs(TypedDict):
 hidden_states: torch.Tensor
 topk_ids: torch.Tensor
 topk_weights: torch.Tensor
 config: _native_moe_Config
_native_moe_HIDDEN = 4096
_native_moe_TOP_K = 8
_native_moe_N_EXPERTS_TOTAL = 256
_native_moe_LOCAL_START = 0
_native_moe_LOCAL_EXPERTS = 32
_native_moe_SHARDING = 'ep8'
_native_moe_CHUNK_TOKENS = 384
_native_moe_EMAX = _native_moe_CHUNK_TOKENS * _native_moe_TOP_K
_native_moe_JMAX = 32 + _native_moe_EMAX // 64 + 8
_native_moe_WGS_SLOTS = 160
_native_moe_WGS_FIXW = 4096
_native_moe_WGS_FIXF = 2 * _native_moe_WGS_SLOTS * 4 * _native_moe_WGS_FIXW + 2 * _native_moe_WGS_SLOTS * 4 + 136 * 16
_native_moe_WGS_XROWS = _native_moe_EMAX + 8 * (32 + _native_moe_EMAX // 32 + 8)
_native_moe_MAIN_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1')
_native_moe_TINY_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_L2H=0', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')
_native_moe_TINY_MAX_M = 4
_native_moe_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')
_native_moe_TINY2_MAX_M = 8
_native_moe_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')
_native_moe_MID_MAX_M = 16
_native_moe_MAIN_NS = 3
_native_moe_MAIN_FALLBACK_DEFINES = ('-DOKC_CPF=0', '-DOKC_CPFF=0', '-DOKC_DYNK=7', '-DOKC_EARLYN=2', '-DOKC_EPI4=0', '-DOKC_FCOSTB=35', '-DOKC_FCOSTBB=36', '-DOKC_FINCOL=0')
_native_moe_PDL = False
_native_moe_PRMT_SCALES = True
_native_moe_DQ7 = True
_native_moe_DIAG = False
_native_moe_DIAG_VARIANTS = {}
_native_moe_DIAG_BW = False
_native_moe_DIAG_BW_MODES = (5, 8, 12, 16) + tuple(range(17, 25))
_native_moe_ALT_SRCS = {}
_native_moe__REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]
_native_moe_CUDA_SRC = '\n// ---------------------------------------------------------------- roadmap\n// Compile-time knobs (the OKC_* values the host actually passes, plus\n// defaults for the rest) -> Params / Smem -> small helpers (dequant,\n// mma, mbarrier, L2 hints) -> route() -> skinny ring (two variants:\n// dynamic-pool SKDYN / static, see below) -> fat stream (cp.async +\n// mma.sync, wgmma variant) -> WS helper warp -> WGS (RS-wgmma,\n// producer/consumer) -> run_phase -> moe_fused -> C++ launcher okc_run.\n// v2: dead #if branches pruned for the only builds the host compiles\n// (MAIN / TINY / TINY2 / MID, each with and without -DOKC_PRMTSC=1, plus\n// the MAIN fallback; BW with no -D), sm_90a only; dead functions,\n// defines and BW switch cases removed. Taken-path code is unchanged.\n// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.\n#include <cuda_bf16.h>\n#include <type_traits>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define OKC_DEBUG 0\n#define OKC_USE_TMA 1\n#ifndef OKC_NACC\n#define OKC_NACC 2\n#endif\n#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores\n                     // OKC_FTSC2; the TINY builds measured -0.5%, so off by default)\n#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0\'s barrier inits\n#define OKC_WGSFINSS 1  // WGS: the final pass forms each route\'s sum of squares (the phase-B epilogue only writes y)\n#ifndef OKC_FINCOL\n#define OKC_FINCOL 1\n#endif\n#define OKC_WGMIN 9\n#define OKC_XPRE 1\n#define FAST1_ARCH 1\n#ifndef OKC_FAST1\n#define OKC_FAST1 3\n#endif\n#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)\n#ifndef OKC_CPFF\n#define OKC_CPFF 1\n#endif\n#ifndef OKC_FCOSTBB\n#define OKC_FCOSTBB OKC_FCOSTB\n#endif\n#define OKC_FCOSTSB OKC_FCOST\n#define OKC_DYN 1\n#ifndef OKC_DYNK\n#define OKC_DYNK 4\n#endif\n#define OKC_TDYN 0\n#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)\n#define OKC_FT_T 64\n#ifndef OKC_SPLIT\n#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)\n#endif\n#ifndef OKC_TINY\n#define OKC_TINY 0\n#endif\n#ifndef OKC_NW\n#define OKC_NW 8\n#endif\n#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n#define OKC_BIGM 64\n#define OKC_XPF 0\n#ifndef OKC_CPF\n#define OKC_CPF 1\n#endif\n#define OKC_FATSTATIC 1\n#define OKC_FCOST 40\n#ifndef OKC_FCOSTB\n#define OKC_FCOSTB 40\n#endif\n#ifndef OKC_WS\n#define OKC_WS 0  // tiny build: one extra helper warp issues the ring\'s TMA copies and runs every skinny epilogue\n#endif\n#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped\n#ifndef OKC_WGS\n#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp\n#endif\n#define SILU_DIV(a, b) __fdividef((a), (b))\n#ifndef OKC_UWARP\n#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers\n#endif\n#if OKC_UWARP\n#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)\n#else\n#define WARP_IDX(t) ((t) >> 5)\n#endif\n#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n#ifndef OKC_FINPF\n#define OKC_FINPF 0   // skinny builds: final item\'s topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)\n#endif\n#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp\n#define OKC_NOLOAD 0\n\nnamespace okc {\n\n#ifndef OKC_L2H\n#define OKC_L2H 1\n#endif\n#if OKC_L2H\n// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in\n// the same call (evict last). Hints only: the math and the memory traffic\'s results are unchanged.\n#define L2H_SUFFIX ".L2::cache_hint"\n#define L2H_OPERAND ", %4"\n#define L2H_ARG , "l"(0x12F0000000000000ull)\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {\n  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");\n}\n__device__ __forceinline__ void st_keep_f32(float* a, float v) {\n  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");\n}\n#else\n#define L2H_SUFFIX ""\n#define L2H_OPERAND ""\n#define L2H_ARG\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }\n__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n#endif\n\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers\n#if OKC_WS || OKC_WGS\n// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)\n#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory")\n#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")\n#else\n#define CTA_SYNC() __syncthreads()\n#endif\nconstexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)\n#define OKC_SKT 8\nconstexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)\nconstexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\nstatic_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\nconstexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)\nconstexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks\nconstexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;\nconstexpr int RB_A_LOG = 9, RB_B_LOG = 8;\nstatic_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");\nconstexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A\n// dynamic tail: the last pool units of a phase\'s skinny work are claimed at run time (OKC_DYNK per CTA)\n__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }\nstatic_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");\nstatic_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)\nconstexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;\nconstexpr int KS = 256;         // fat: K per x stage (4 chunks)\nconstexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)\nconstexpr float TWO62 = 4.611686018427388e18f;\n\nstruct Params {\n  const uint4* wA; const uint4* sA; const float* gsA;\n  const uint4* wB; const uint4* sB; const float* gsB;\n  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here\n  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)\n  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)\n  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)\n  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]\n  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]\n  int* xready;                           // WGS: per job, rows of xA written\n  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)\n  uint32_t zero;  // always 0 (opaque to the compiler)\n  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;\n  int M, lo, nloc;\n  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit\n  int* doneA;      // [JMAX]\n  __nv_bfloat16* inter;  // [EMAX, ID]\n  float* ybuf;     // [EMAX, HD]\n  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1\n  float* ss1;      // [EMAX]\n  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};\n\nstruct Smem {\n  uint32_t lut[256];\n#if !OKC_WGS\n  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums\n#else\n  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)\n#endif\n#if OKC_WS\n  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free\n  int wsr[4];                          // WS: this CTA\'s phase A and phase B unit ranges\n#endif\n#if OKC_WGS\n  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty\n  int wglast[4];      // WGS: this warpgroup completed a split unit\n  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job\n  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job\n  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)\n  uint32_t wtot;      // WGS: phase-A total cost\n  uint32_t wtotB;     // WGS: phase-B total cost\n#else\n  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)\n#endif\n#if OKC_WGS\n  int wgr[4];         // WGS: this CTA\'s phase A and phase B unit ranges\n#endif\n  int seq, fseq, preissued, preissuedB;\n  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue\'s job -> expert map)\n  void* xbuf_ptr;\n  int cnt[64], off[64];\n  int wc[NW][32];\n  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];\n  int njF, njS;\n  int gen;          // the grid barrier\'s generation at launch start (+1 per launch): picks this launch\'s ss copy\n  float* ssp;       // this launch\'s ss copy\n  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)\n  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)\n  signed char eid[EMAX];\n  short ent[EMAX];   // sorted entries (token*8+slot) by expert\n  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n\n// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide\n// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up\n__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {\n  const int p = a * c;\n  int q = __float2int_rz((float)p * rG);\n  if (q * G > p) --q;\n  else if ((q + 1) * G <= p) ++q;\n  return q;\n}\n__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)\n__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n\n__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,\n                                         uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {\n  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);\n  int sz = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(s), "l"(gmem), "r"(sz));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ unsigned long long gtimer() {\n  unsigned long long t;\n  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));\n  return t;\n}\n#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\nextern __shared__ __align__(16) unsigned char smem_raw[];\n__device__ __forceinline__ float* ss_cur();  // this launch\'s ss copy (defined after Smem users)\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n  return v;\n}\n\n// Exact FP4 placement adapted from Lee Penkman, lee101/openkernels,\n// c565e373d34575c5f7df862b05f8be7619078b0f (Apache-2.0).\n// The corresponding prepare-time bit permutation is a bijection of all 32 bits.\n__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n  r1 = __funnelshift_l(q, q, 10) & M;\n  r2 = __funnelshift_l(q, q, 13) & M;\n  r3 = ((q & 0x001B001Bu) * 0x1084u) & M;\n}\n\n\n// ---------------------------------------------------------------- routing prologue (every CTA, identical result)\n__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);\n  const int E = p.M * 8;\n  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;\n  const int i0 = warp * CH, i1 = min(E, i0 + CH);\n  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;\n#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)\n  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores\n    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;\n    int4 ev[LPT];\n    const int E4 = E >> 2;\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);\n    }\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      if (i4 < E4) {\n        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};\n        char4 c;\n        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);\n        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);\n        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);\n        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);\n        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;\n      }\n    }\n  }\n#else\n#pragma unroll 8\n  for (int i = tid; i < E; i += NT) {\n    int e = __ldg(p.topk_ids + i) - p.lo;\n    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);\n  }\n#endif\n  CTA_SYNC();\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n  }\n  CTA_SYNC();\n  if (warp == 0) {\n    int c[NW];\n    int tot = 0;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }\n    int incl = tot;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }\n    int off = incl - tot;\n    S.cnt[lane] = tot;\n    S.off[lane] = off;\n    int o2 = off;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }\n    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each\n#if OKC_SPLIT\n    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");\n    const int nfj = 0;\n    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights\n#else\n    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;\n    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;\n#endif\n    int fi = nfj, si = nsj;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);\n      if (lane >= d) { fi += v1; si += v2; }\n    }\n    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);\n    int fj = fi - nfj, sj = nfTot + si - nsj;\n    for (int q = 0; q < nfj; ++q) {\n      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);\n    }\n#if OKC_SPLIT\n    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n#else\n    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }\n#endif\n    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }\n    __syncwarp();\n#if !OKC_WGS  // (the WGS build has no fat queue)\n    // fat queue order: largest n-tile count first (LPT), stable by job index\n    int pos = 0;\n    for (int key = FT_T / 8; key >= 1; --key) {\n      for (int f0 = 0; f0 < nfTot; f0 += 32) {\n        const int f = f0 + lane;\n        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;\n        const unsigned bal = __ballot_sync(0xffffffffu, m);\n        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;\n        pos += __popc(bal);\n      }\n    }\n#endif\n  }\n  CTA_SYNC();\n  early_issue(p, S);\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0) {\n      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));\n      S.ent[q] = (short)i;\n      S.pos[i] = (short)q;\n    } else if (i < i1) {\n      S.pos[i] = -1;\n    }\n    __syncwarp();\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n    __syncwarp();\n  }\n  CTA_SYNC();\n}\n\n// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)\n// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile\n// Local change: match Marlin\'s BF16 accumulator/global-scale boundaries.\n// Local change: pair independent outputs, retaining both BF16 rounding boundaries.\n// CUDA Math API: __hmul2_rn is RN-even and prevents contraction with additions.\n__device__ __forceinline__ float2 achita_marlin_out2(float encoded_acc0, float encoded_acc1, float encoded_scale) {\n  const __nv_bfloat162 acc = __floats2bfloat162_rn(encoded_acc0 * TWO62, encoded_acc1 * TWO62);\n  const __nv_bfloat16 gs = __float2bfloat16_rn(encoded_scale * 0x1p-62f);\n  return __bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)));\n}\n__device__ __forceinline__ float probe_q(float v) {\n  return v;\n}\n__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    if (tk < ntok) {\n      const float2 achita_pair_0 = achita_marlin_out2(r[h], r[2 + h], gsc);\n      float gt = achita_pair_0.x, up = achita_pair_0.y;\n      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));\n    }\n  }\n}\n// returns per-token partial sum of squares (2 values) for this lane\'s two tokens\n__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,\n                                      float* sq) {\n  const float w0 = __bfloat162float(p.normw[n0]), w1 = __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    const float2 achita_pair_1 = achita_marlin_out2(r[h], r[2 + h], gsc);\n    float v0 = achita_pair_1.x * w0, v1 = achita_pair_1.y * w1;\n    sq[h] = v0 * v0 + v1 * v1;\n    if (tk < ntok) {\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);\n    }\n  }\n}\n__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);\n  }\n  if (g == 0) {\n#pragma unroll\n    for (int h = 0; h < 2; ++h)\n      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);\n  }\n}\n\n\n\n// ---------------------------------------------------------------- skinny path: shared-memory weight ring\n// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.\n// Units of the CTA\'s range stream through OKC_NS smem slots (TMA bulk copies on sm_90).\n// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.\nconstexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n#if OKC_PRMTSC\n// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the\n// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes\n// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.\n#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)\n// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the\n// ALU pipe, which the FP4 decode saturates\n#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)\n#endif\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"\n      ::"r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n  const size_t blk = (size_t)S.jE[j] * rbs + rb;\n  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,\n                                            uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], st = S.jS[j];\n  {\n    const int r = g;\n    if (r < cnt) {\n      const __nv_bfloat16* row;\n      if (PB) row = p.inter + (size_t)(st + r) * ID;\n      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) {\n        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);\n        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);\n      }\n    } else {\n#pragma unroll\n      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n    }\n  }\n}\n\nconstexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16\ntemplate <bool PB>\n__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,\n                                             __nv_bfloat16* x1s) {\n  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4\n  const int cnt = S.jC[j], st = S.jS[j];\n  for (int i = lane; i < 8 * 64; i += 32) {\n    const int r = i >> 6, c = i & 63;\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (8 + r < cnt) {\n      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n      v = PB ? __ldcg(src) : __ldg(src);\n    }\n    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;\n  }\n  __syncwarp();\n}\n\n// skinny_ring exists twice, deliberately: the SKDYN variant claims tail units\n// from a run-time atomic pool, the #else variant issues a fixed static\n// range. The bodies share most of their epilogue code, but the pool-claim\n// logic interleaves with the ring refill in ways a merge would risk\n// changing; keep them in sync by hand.\n#if SKDYN\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows\n  const int rsh = PB ? RB_B_LOG : RB_A_LOG;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  // cross-phase prefill: with no fat jobs, phase A\'s last ring refills already fetch this CTA\'s first B units\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight\n  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;\n  bool exh = poolBase >= nSall;\n  int* pctr = p.ctr + (PB ? 1 : 0);\n  auto take = [&]() -> int {\n    if (snext < v1) return snext++;\n    if (exh) return -1;\n    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);\n    const int v = poolBase + cl;\n    if (v >= nSall) { exh = true; pend = -2; return -1; }\n    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill\n    if (!PB) {\n      const int jj = njF + (v >> rsh);\n      S.dync[jj] += 1;\n      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);\n    }\n    return v;\n  };\n  auto refill = [&](int sl) {\n    const int v = take();\n    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }\n    else if (cpf && preB < nB) {\n      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);\n      S.uslot[sl] = BFLAG | (v0B + preB);\n      ++preB;\n    } else {\n      S.uslot[sl] = -1;\n    }\n  };\n  if (tid == 0)\n    for (int i = pre; i < OKC_NS; ++i) {\n      const int sl = (seq0 + i) % OKC_NS;\n      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) {\n    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);\n    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);\n    nwc0 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n  }\n  int nproc = 0;\n  for (int i = 0;; ++i) {\n    int v;\n    if (i < nU) {\n      v = v0 + i;  // static units: no slot lookup\n    } else {\n      v = S.uslot[(seq0 + i) % OKC_NS];\n      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B\'s prefetched units follow)\n      v &= ~BFLAG;\n    }\n    const int j = njF + (v >> rsh), rb = v & (rbs - 1);\n    float nwn0 = 0.f, nwn1 = 0.f;\n    int vn = -1;\n    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];\n    if (PB && tid < 128 && vn >= 0) {\n      const int rbq = (vn & ~BFLAG) & (rbs - 1);\n      nwn0 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n#if OKC_NW == 16\n    // a warp covers half of an 8-warp K slice: its 4 chunks\' scale bytes are 8 of the slice lane\'s 16\n    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);\n    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);\n#else\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n#endif\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();  // slot sl fully consumed; partials visible\n    if (tid == 0) {\n      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float2 achita_pair_2 = achita_marlin_out2(s0, s1, gsc);\n          const float gt = achita_pair_2.x, up = achita_pair_2.y;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float2 achita_pair_3 = achita_marlin_out2(s0, s1, gsc);\n        const float v0f = achita_pair_3.x * nwc0, v1f = achita_pair_3.y * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n    ++nproc;\n  }\n  seq = seq0 + nproc;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      if (nU > 0) {\n        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);\n        for (int jj = jf; jj <= jl; ++jj) {\n          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n          atomicAdd(p.doneA + jj, b - a);\n        }\n      }\n      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units\n        const int k = S.dync[jj];\n        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#else  // static skinny_ring: fixed range, no run-time pool\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  if (nU <= 0) return;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  // cross-phase prefill: with no fat jobs, phase A\'s last ring refills already fetch this CTA\'s first B units\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  for (int i = pre; i < OKC_NS; ++i) {\n    const int sl = (seq0 + i) % OKC_NS;\n    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }\n  }\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    float nwn0 = 0.f, nwn1 = 0.f;\n    if (PB && tid < 128 && i + 1 < nU) {\n      const int rbq = (v + 1) % rbs;\n      nwn0 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) {\n#pragma unroll\n      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();  // slot sl fully consumed; partials visible\n    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i + OKC_NS - nU < nB) {\n      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);\n      ++preB;\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float2 achita_pair_4 = achita_marlin_out2(s0, s1, gsc);\n          const float gt = achita_pair_4.x, up = achita_pair_4.y;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float2 achita_pair_5 = achita_marlin_out2(s0, s1, gsc);\n        const float v0f = achita_pair_5.x * nwc0, v1f = achita_pair_5.y * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n  }\n  seq = seq0 + nU;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n      for (int jj = jf; jj <= jl; ++jj) {\n        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n        atomicAdd(p.doneA + jj, b - a);\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#endif\n\n// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens\n// ---------------------------------------------------------------- fat unit through the smem ring\n// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x\n// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile\n// for the stage is staged by cp.async into two buffers that live in slots 3..4.\nconstexpr int NSF = OKC_NSF;\ntemplate <bool PB>\n__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {\n  const int nub = PB ? UB_B : UB_A;\n  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;\n  if (threadIdx.x == 0) {\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n  }\n}\n\n// ---------------------------------------------------------------- fat units: one continuous stream per CTA\n// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps\n// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit\'s weights).\n// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem\n__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);\n}\n__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }\ntemplate <int N>\n__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory"); }\ntemplate <int NTT> struct WG;\ntemplate <> struct WG<1> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<2> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<3> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<4> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<5> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<6> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<7> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<8> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\n\n// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from\n// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {\n  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads\n  // k block cc at byte offset 32 st\n  const int g = lane >> 2;\n  const uint32_t xb = smem_u32(xs);\n  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired\n  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const int sb = cc & 1;\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);\n    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact\n      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2\n      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)\n      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);\n      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);\n#define FMG0(r) bmul2((r), sg0)\n#define FMG80(r) bmul2((r), sg80)\n#define FMG1(r) bmul2((r), sg1)\n#define FMG81(r) bmul2((r), sg81)\n      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;\n      dq4((&w0.x)[st], r0, r1, r2, r3);\n      dq4((&w1.x)[st], q0, q1, q2, q3);\n      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);\n      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);\n      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);\n      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);\n#undef FMG0\n#undef FMG80\n#undef FMG1\n#undef FMG81\n    }\n    wg_fence();\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);\n      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);\n      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);\n    }\n    wg_commit();\n    if (cc >= 1) {\n      wg_wait<1>();  // the previous chunk\'s group retired: its register set may be rewritten\n#pragma unroll\n      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];\n    }\n  }\n  wg_wait<0>();\n#pragma unroll\n  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];\n  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f\n}\n\n// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {\n  const int g = lane >> 2, t = lane & 3;\n  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);\n#if OKC_NW == 8\n  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);\n#endif\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n#if OKC_NW == 8\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n#endif\n#if OKC_PRMTSC\n    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);\n#if OKC_NW == 8\n    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);\n#endif\n#else\n    const int sh = (cc & 1) * 16;\n    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;\n    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;\n    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];\n    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];\n#endif\n    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;\n#pragma unroll\n    for (int sp = 0; sp < 2; ++sp) {\n      uint32_t a0[2][4], a1[2][4];\n      (void)a1;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int s2 = 2 * sp + h;\n        uint32_t r0, r1, r2, r3;\n        dq4((&w0.x)[s2], r0, r1, r2, r3);\n        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);\n#if OKC_NW == 8\n        dq4((&w1.x)[s2], r0, r1, r2, r3);\n        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);\n#endif\n      }\n#pragma unroll\n      for (int nt = 0; nt < NTT; ++nt) {\n        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);\n        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);\n#endif\n        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);\n#endif\n      }\n    }\n  }\n}\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);\n  const int g = lane >> 2, t = lane & 3;\n  int* ctr = p.ctr + (PB ? 1 : 0);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  uint64_t* fb = S.bars + 5;\n  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);\n  constexpr int NKS = HD / KS;  // 16\n  int u = blockIdx.x;\n  (void)ctr;\n  if (u >= nF) return;\n  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)\n  (void)ubs;\n  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };\n  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete\n    if (PB) {\n      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }\n      CTA_SYNC();\n    }\n  };\n  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };\n  auto stage_x = [&](int j, int ks, int xsl) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int ntl = (cnt + 7) >> 3;\n    const int nrow = ntl * 8;\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n    const int per = nrow * (KS / 8);\n    if (is_nat(j)) {\n      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)\n      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);\n      for (int i = tid; i < per; i += NT) {\n        const int r = i >> 5, ch = i & 31;\n        const bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;\n        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);\n      }\n    } else {\n      for (int i = tid; i < per; i += NT) {\n        int r = i / (KS / 8), c = i - r * (KS / 8);\n        bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;\n        cp_async16(xs + r * XP_FT + c * 8, src, v);\n      }\n    }\n    cp_commit();\n  };\n  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)\n  const __nv_bfloat16* xsrc[8];\n  uint32_t xdst[8];\n  uint32_t xval = 0;\n  int xn = 0;\n  auto prep_x = [&](int j) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);\n    xval = 0;\n    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;\n#pragma unroll\n    for (int k = 0; k < 8; ++k) {\n      const int i = tid + k * NT;\n      const int r = i >> 5, c = i & 31;\n      const bool v = i < per && r < cnt;\n      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;\n      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;\n      xdst[k] = r * XP_FT + c * 8;\n      if (v) xval |= 1u << k;\n    }\n  };\n  auto stage_xp = [&](int ks, int xsl) {\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n#pragma unroll\n    for (int k = 0; k < 8; ++k)\n      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);\n    cp_commit();\n  };\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int f = S.fseq;  // ring position of the current unit\'s k-stage 0\n  int xq = 0;      // x stage parity counter\n  int j, ub;\n  job_of(u, j, ub);\n#pragma unroll\n  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));\n  wait_A(j);\n  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);\n  while (true) {\n    const int un = u + gridDim.x;\n    int jn = 0, ubn = 0;\n    if (un < nF) job_of(un, jn, ubn);\n    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];\n    const int ntile = (cnt + 7) >> 3;\n    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)\n    const bool nat = is_nat(j);\n    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;\n    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];\n#pragma unroll\n    for (int i = 0; i < FT_T / 8; ++i)\n#pragma unroll\n      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }\n    for (int ks = 0; ks < NKS; ++ks) {\n      if (OKC_XPRE) {\n        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }\n        else cp_commit();\n      } else {\n        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }\n        else cp_commit();\n      }\n      cp_wait<1>();\n      const int q = f + ks, sl = q % NSF;\n      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));\n      if (nat) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");  // cp.async x writes -> wgmma reads\n      CTA_SYNC();  // x stage visible to every warp\n      const unsigned char* slot = ring + sl * UNIT_BYTES;\n      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);\n      if (nat) {\n        switch (ntc) {\n          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n        }\n      } else {\n        switch (ntc) {\n          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;\n          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;\n        }\n      }\n      CTA_SYNC();  // slot sl and the x stage consumed\n      ++xq;\n      // refill: the stage NSF ahead, in this unit or the next one\n      const int kn = ks + NSF;\n      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);\n      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));\n    }\n    // epilogue of unit u (registers -> global)\n    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;\n    if (!PB) {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);\n#if OKC_NW == 8\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);\n#endif\n        }\n      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) atomicAdd(p.doneA + j, 1);\n    } else {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          float sq0[2], sq1[2];\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);\n#if OKC_NW == 8\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);\n          sq0[0] += sq1[0]; sq0[1] += sq1[1];\n#else\n          (void)sq1;\n#endif\n          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);\n        }\n    }\n    f += NKS;\n    if (un >= nF) break;\n    u = un; j = jn; ub = ubn;\n  }\n  cp_wait<0>();\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  if (tid == 0) S.fseq = f;\n  CTA_SYNC();\n}\n\ntemplate <bool PB>\n__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  fat_stream<PB, false>(p, S, xbuf, nF, ubs);\n}\n\n\n\n// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)\n__device__ void early_issue(const Params& p, Smem& S) {\n#if FAST1_ON || OKC_WGS\n  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring\n#endif\n  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region\n    if (threadIdx.x == 0) S.preissued = 0;\n    return;\n  }\n  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);\n  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in\n  // skinny_ring issues the rest after routing, so every CTA\'s first unit is served ahead of the second ones\n  const int n = min(OKC_EARLYN, v1 - v0);\n  if (threadIdx.x == NT - 32) {\n    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }\n    S.preissued = n > 0 ? n : 0;\n  }\n}\n\n// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost\n#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper\n// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit\'s\n// epilogue, so no CTA-wide barrier sits in the per-unit loop.\n// the last WSBLK units keep their partial buffers until the compute warps\' own epilogues: WSBLK > WSNR deadlocks the\n// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)\nstatic_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");\n#define OKC_WSRF 0\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\ntemplate <bool PB>\n__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  const int seq0 = seq;\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs;\n    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    if (j != curj) {\n      curj = j;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n    }\n    const int q = seq0 + i;\n    const int sl = q % OKC_NS;\n    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB: the compute warps keep the base loop\'s CTA barrier per unit (slot consumed + partials visible), thread 0\n    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop\n    const int bb = q % NRB;\n    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n    }\n  }\n  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nU - 1 - warp;\n    const int q = seq0 + ul, bb = q % NRB;\n    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = __uint_as_float(lnw0 << 16), nw1 = __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float2 achita_pair_6 = achita_marlin_out2((h ? sm.y : sm.x), (h ? sm.w : sm.z), gsc);\n      const float v0f = achita_pair_6.x * nw0, v1f = achita_pair_6.y * nw1;\n      float sq = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n    }\n  }\n  seq = seq0 + nU;\n}\n\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {\n  const int sl = q % OKC_NS;\n  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);\n  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);\n}\n\n__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n  const int lane = threadIdx.x & 31;\n  WS_JOIN();  // routing tables and this CTA\'s ranges are ready\n  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n  for (int q = 0; q < Nh; ++q) {\n    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q\'s slot once every compute warp released it\n      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);\n    }\n    const bool PB = q >= nA;\n    const int v = PB ? b0 + (q - nA) : a0 + q;\n    const int rbs = PB ? RB_B : RB_A;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    const int bb = q % NRB;\n    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    if (!PB) {\n      const float gsc = p.gsA[ce] * TWO62;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        if (tk < cnt) {\n          const float2 achita_pair_7 = achita_marlin_out2((h ? sm.y : sm.x), (h ? sm.w : sm.z), gsc);\n          const float gt = achita_pair_7.x, up = achita_pair_7.y;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      }\n      if (q == nA - 1) {  // this CTA\'s phase A is done: credit its units to the jobs\' counters\n        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n          for (int jj = jf; jj <= jl; ++jj) {\n            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);\n            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {\n      const float gsc = p.gsB[ce] * TWO62;\n      const int n0 = rb * 16 + gg;\n      const float nw0 = __bfloat162float(p.normw[n0]), nw1 = __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float2 achita_pair_8 = achita_marlin_out2((h ? sm.y : sm.x), (h ? sm.w : sm.z), gsc);\n        const float v0f = achita_pair_8.x * nw0, v1f = achita_pair_8.y * nw1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\ntemplate <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {\n  const float rG = __fdividef(1.f, (float)G);\n  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;\n  auto fcost = [](long long k) -> long long {\n    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B\'s skinny units are slower, so its ratio is lower)\n    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);\n    return k > 0 ? (long long)fc * k : 0;\n  };\n  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level\n  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;\n  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;\n  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };\n  const long long WG = W(G);\n  if (nF > 0 && WG > 0) {\n    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))\n    const float sc = __fdividef((float)nS, (float)WG);\n    auto split = [&](int x) -> int {\n      if (x >= G) return nS;\n      const int v = __float2int_rz((float)W(x) * sc);\n      return v < nS ? v : nS;\n    };\n    v0 = split(c); v1 = split(c + 1);\n  } else {\n    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);\n  }\n}\n\n#if OKC_WGS\n// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)\n// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,\n// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales\n// [grp][warp][g][h][j]; the stage\'s x slice (the job\'s token rows, rounded up to 8, x 256 K as wgmma B core matrices\n// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16\n// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).\n// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a\n// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.\n#define OKC_WGSNX 3                             // x ring slots\nconstexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)\nconstexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage\'s last wgmma retired)\nconstexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job\nconstexpr int WGS_PIECE = 9216;                 // one warpgroup\'s weights in a stage\nconstexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K\nconstexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices\nstatic_assert(BX_E + WGS_NX <= 24, "WGS barriers");\nconstexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)\nconstexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]\nconstexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs\n__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {\n  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;\n}\n__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {\n  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;\n}\n// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at\n// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each\n#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)\n#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\nconstexpr unsigned long long wgs_pack8(const int (&c)[8]) {\n  unsigned long long r = 0;\n  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);\n  return r;\n}\nconstexpr int WGS_CA[8] = {OKC_WGSA};\nconstexpr int WGS_CB[8] = {OKC_WGSB};\nconstexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);\nstatic_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");\n__device__ __forceinline__ int wgs_cost(int pb, int cnt) {\n  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);\n}\n#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)\n#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time\n#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather\n#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n#define OKC_WGSFCI 3\n#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)\n__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {\n  (void)tag;\n  mbar_wait(bar, parity);\n}\ntemplate <int NTT> struct WGSMMA;\ntemplate <> struct WGSMMA<1> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<2> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<3> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<4> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<5> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<6> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<7> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<8> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\n// two producer warps: the weights warp streams every stage\'s 36 KB through the weight ring (TMA, lane 0), the x warp\n// gathers every stage\'s x slice into the x ring (cp.async) and, in phase B, first waits for the job\'s phase-A units;\n// so the weights run ahead by the whole ring even while x waits\n// Phase A runs a CTA\'s trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit\'s\n// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase\n// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the\n// order (its first stages wait for phase A\'s units in order).\n__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {\n  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range\'s last stage\n  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;\n}\n__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)\n__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((threadIdx.x & 31) != 0) return;\n  int ws = 0, n = 0;\n  uint32_t wph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job\n    const unsigned char* base = pb ? p.wBs : p.wAs;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);\n      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n      uint64_t* bar = S.bars + BW_F + ws;\n      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n}\n// x producer: one bulk copy per stage (lane 0): the job\'s x slice for the stage is contiguous in xA (phase A, written\n// by every CTA\'s gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B\nconstexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound\n__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {\n  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the\n  // stage ranges while the consumers gather phase A\'s x\n  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");\n  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n  __syncwarp();\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((threadIdx.x & 31) != 0) return;\n  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;\n  int xs = 0, n = 0;\n  uint32_t xph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;\n    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;\n    int curj = -1, npad = 8;\n    const __nv_bfloat16* jb = xsrc;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl;\n      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);\n      if (j != curj) {\n        curj = j;\n        npad = ((S.jC[j] + 7) >> 3) << 3;\n        jb = xsrc + (size_t)S.jrow[j] * HD;\n        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered\n      }\n      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them\n      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }\n      uint64_t* bar = S.bars + BX_F + xs;\n      const uint32_t bytes = (uint32_t)npad * 512u;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"\n                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }\n    }\n  }\n}\n// phase A\'s x gather (all consumer threads of every CTA, once, at the start): this CTA\'s share of the jobs\' padded rows\n// copied into xA\'s per-stage wgmma layout, then counted into xready (padding rows are counted, not written)\n__device__ void wgs_gather(const Params& p, Smem& S) {\n  const int tid = threadIdx.x, nj = S.njF + S.njS;\n  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);\n  int j = 0;\n  // all of a batch\'s row loads first, then the stores: one x row at a time was one dependent HBM round trip per row\n  // (the next row\'s load could not move above the previous row\'s store: they may alias), 1.3-7 us at bs256\n  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");\n  constexpr int GB = OKC_GATHB;\n  for (int qb = q0; qb < q1; qb += GB) {\n    uint4 v[GB];\n    uint4* d[GB];\n#pragma unroll\n    for (int k = 0; k < GB; ++k) {\n      const int q = qb + k;\n      d[k] = nullptr;\n      if (q < q1) {\n        while (S.jrow[j + 1] <= q) ++j;\n        const int r = q - S.jrow[j], cnt = S.jC[j];\n        if (r < cnt) {\n          const int ntt = (cnt + 7) >> 3;\n          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));\n          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);\n        }\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < GB; ++k)\n      if (d[k]) *d[k] = v[k];\n  }\n  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job (release: the CTA\'s stores are ordered before by the barrier)\n    int jj = 0;\n    for (int q = q0; q < q1;) {\n      while (S.jrow[jj + 1] <= q) ++jj;\n      const int e = min(q1, S.jrow[jj + 1]);\n      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");\n      q = e;\n    }\n  }\n}\n// Work left from the previous segment, done after the next segment\'s first wgmma group (so the stores and atomics it\n// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split\n// unit\'s share and, if that completes the unit, read the summed partials back and run its epilogue.\nstruct WgsPend { int kind, u, slot, nst; };\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {\n  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];\n  if (!PB) {\n    // SwiGLU straight into xB (phase B\'s per-stage wgmma B layout) instead of row-major inter\n    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;\n    const float gsc = p.gsA[e] * TWO62;\n    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = nt * 8 + 2 * t + h;\n      if (tk < cnt) {\n        const float2 achita_pair_9 = achita_marlin_out2(d4[h], d4[2 + h], gsc);\n        const float gt = achita_pair_9.x, up = achita_pair_9.y;\n        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),\n                     __float2bfloat16(probe_q(v)));\n      }\n    }\n  } else {\n    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)\n    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;\n    const float gsc = p.gsB[e] * TWO62;\n    const float w0 = __bfloat162float(__ldg(p.normw + n0)), w1 = __bfloat162float(__ldg(p.normw + n0 + 8));\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = tok0 + h;\n      if (tk < cnt) {\n        const float2 achita_pair_wgs = achita_marlin_out2(d4[h], d4[2 + h], gsc);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), achita_pair_wgs.x * w0);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), achita_pair_wgs.y * w1);\n      }\n    }\n  }\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {\n  if (pd.kind == 0) return;\n  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  if (pd.kind == 1) {\n    // release (cumulative over the warpgroup\'s epilogue stores ordered by the barrier); pairs with the producer\'s\n    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase\n  if constexpr (DF) {\n  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the\n  // unit\'s last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA\'s end). The others just\n  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts\n  if (pd.kind == 2) {\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  pd.kind = 0;\n  } else {\n  if (wt == 0) {\n    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");\n    S.wglast[wg] = old + pd.nst == 16;\n  }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  const bool last = S.wglast[wg];\n  pd.kind = 0;\n  if (!last) return;\n  }\n  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;\n  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;\n  // two n-tiles\' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load\n  // waits behind the previous tile\'s stores (they may alias), ntt serialized L2 round trips under full HBM load\n  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {\n    float4 v[2];\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) {\n        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};\n        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);\n      }\n  }\n  if (wt == 0) *fc = 0;\n  if (!PB) {\n    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n  }\n}\n// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)\ntemplate <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n                                        int& seq, WgsPend& pd) {\n  const int tid = threadIdx.x, lane = tid & 31;\n  const int warp = WARP_IDX(tid);\n  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)\n  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);\n  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)\n  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas\n  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time\n  constexpr int GS = OKC_WGSGS;\n  uint32_t af[2][GS][4];\n  int prev = -1;  // x slot the last committed group may still read\n  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot\'s offset (the address field cannot\n  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)\n  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);\n  const unsigned char* pc0 = ring + wg * WGS_PIECE;\n  auto stage = [&](auto first) {\n    constexpr bool FIRST = decltype(first)::value;\n    const int ws = seq % WGS_NS, xs = seq % WGS_NX;\n    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));\n    // codes and scales one 4-step group at a time (fewer live registers)\n    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n      const uint4 cwg = cwn;\n      const uint2 scg = scn;\n      if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);\n      } else if (lane == 0) {\n        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free\n      }\n#pragma unroll\n      for (int h = 0; h < 4 / GS; ++h) {\n        const int ci = lg * (4 / GS) + h;  // commit group index in the stage\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          const int js = h * GS + jj;      // k16 step within the 4-step load group\n          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;\n          uint32_t r0, r1, r2, r3;\n          dq4(q, r0, r1, r2, r3);\n          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows\' scales in\n          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer\n          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);\n        }\n        wg_fence();\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          if (FIRST && ci == 0 && jj == 0)\n            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\n          else\n            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n        }\n        wg_commit();\n        wg_wait<1>();\n        if (ci == 0) {\n          if (prev >= 0) {  // the previous stage\'s last group just retired: its x slot is free\n            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n            prev = -1;\n          }\n        }\n        // the previous segment\'s deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG\'s\n        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)\n        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n      }\n    }\n    prev = xs;\n    ++seq;\n  };\n  stage(std::true_type{});\n#pragma unroll 1\n  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});\n  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }\n  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)\n    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;\n#pragma unroll\n    for (int nt = 0; nt < NTT; ++nt)\n      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),\n                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");\n    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,\n    // phase B the one holding stage 0\n    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};\n    return;\n  }\n  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n#pragma unroll\n  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {\n  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)\n  constexpr int UBL = PB ? 4 : 5;\n  WgsPend pd{0, 0, 0, 0};\n#pragma unroll 1\n  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {\n    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)\n      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);\n      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;\n    }\n    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));\n    // a unit split between CTAs: its fixup slot is the CTA holding its first stage\n    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));\n    const int ntt = (S.jC[u >> UBL] + 7) >> 3;\n    switch (ntt) {\n      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n    }\n    gs += s1 - s0;\n  }\n  wgs_flush<PB>(p, S, pd);\n}\n// cost-weighted stage ranges: per-stage cost by the job\'s n-tiles, a prefix per job and phase (S.wcp, S.wcpB)\n// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run\n// from a cold instruction cache at the start of every launch\n__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {\n  uint32_t acc = 0, accB = 0;\n  for (int j = 0; j < nj; ++j) {\n    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;\n    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;\n    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;\n  }\n  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;\n  S.wtot = acc; S.wtotB = accB;\n}\n// stage at cost position x (phase pb), the stage whose cost interval holds x\n__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n  const int* pre = pb ? S.wcpB : S.wcp;\n  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x\n  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }\n  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);\n  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);\n}\n__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);\n  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);\n}\n// this CTA\'s stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same\n__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\n  const int nj = S.njF + S.njS;\n  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);\n  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}\n// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)\n__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  const int j = s >> (9 - pb);\n  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);\n  // largest c with floor(tot * c / G) < xe\n  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);\n  if (c > G - 1) c = G - 1;\n  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;\n  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;\n  return c;\n}\n#endif\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  const int njF = S.njF, njS = S.njS;\n  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;\n  const int nS = njS * rbs, nF = njF * ubs;\n  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost\n#if OKC_TINY\n  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units\n#else\n  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }\n#endif\n  const int G = gridDim.x, c = blockIdx.x;\n  int v0, v1;\n  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool\n  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);\n  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A\'s last ring\n  // refills already fetch those units (same split function as phase B uses)\n  int cB0 = 0, cBn = 0;\n  bool cpf = false;\n  if (!PB) {\n    const int nFB = njF * UB_B;\n    if (c >= nFB) {\n      int b1;\n      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);\n      cBn = b1 - cB0;\n      cpf = OKC_CPFF || njF == 0;\n    }\n  }\n#if OKC_WS\n  (void)cpf; (void)cBn;\n  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n#else\n  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,\n                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);\n#endif\n}\n\n// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register\n// cap: spills)\n__device__ __forceinline__ float* ss_cur() {\n#if OKC_WGS\n  unsigned long long v;\n  asm volatile("ld.volatile.shared.u64 %0, [%1];\\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));\n  return reinterpret_cast<float*>(v);\n#else\n  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist\n#endif\n}\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n  Smem& S = *reinterpret_cast<Smem*>(smem_raw);\n  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);\n  const int tid = threadIdx.x;\n  DBG(0);\n  // programmatic dependent launch: everything below reads the previous kernel\'s outputs (routing, x) or writes\n  // scratch, so wait for it here; without the launch attribute this returns at once\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if OKC_WGS\n  if (tid >= NT) {\n    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    return;\n  }\n#endif\n#if FAST1_ON\n  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA\'s first units follow\n  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path\n  if (tid >= NT - 32) {\n    const int ln = tid & 31;\n#if OKC_SPLIT\n    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)\n    S.fcnt[ln] = 0;\n    __syncwarp();\n    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {\n      const int i = i0 + ln;\n      int e = -1;\n      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n      const unsigned peers = __match_any_sync(0xffffffffu, e);\n      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n    }\n    __syncwarp();\n    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }\n    const int njob = __shfl_sync(0xffffffffu, jincl, 31);\n    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);\n    const float rG1 = __fdividef(1.f, (float)gridDim.x);\n    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n    const int n1 = min(OKC_FAST1, v11 - v01);\n    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];\n#pragma unroll\n    for (int i = 0; i < OKC_FAST1; ++i) {\n      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)\n      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;\n    }\n#elif OKC_F1EARLY\n    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the\n    // barriers while they are in flight (the inits used to wait behind the cold loads)\n    uint32_t m = 0;\n    const int E8 = p.M * 8;\n    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n    if (ln == 0) {\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    }\n    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)\n      const int e = __ldg(p.topk_ids + i) - p.lo;\n      if (e >= 0 && e < p.nloc) m |= 1u << e;\n    }\n    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n    if (ln == 0) {\n#if OKC_SPLIT || !OKC_F1EARLY\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#else\n      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)\n      const float rG = __fdividef(1.f, (float)gridDim.x);\n      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n#if OKC_SPLIT\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n        if (i >= n) break;\n        const int v = v0 + i;\n        const int e = fex[i];\n#else\n      for (int i = 0; i < n; ++i) {\n        const int v = v0 + i;\n        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);\n#endif\n        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;\n        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;\n        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");\n        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n      }\n      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)\n    asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));\n  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)\n    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;\n    uint32_t b;\n    if (ex == 15 && mt == 7) {\n      b = 0xFFFFu;\n    } else {\n      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);\n      v *= 0x1p64f;\n      b = __float_as_uint(sgn ? -v : v) >> 16;\n    }\n    S.lut[i] = b | (b << 16);\n  }\n  if (tid == 0) {\n#if !FAST1_ON\n#if OKC_WGS\n    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),\n    // partial buffer free (the epilogue warpgroup)\n    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n#else\n    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n#endif\n    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    S.preissued = 0;\n#endif\n    S.seq = 0;\n    S.fseq = 0;\n    S.preissuedB = 0;\n    S.xbuf_ptr = xbuf;\n    // the generation can only change at this launch\'s grid barrier, which every CTA reaches after this point\n    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)\n    S.gen = gen;\n    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n  }\n#if SKDYN\n  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;\n#endif\n  route(p, S);\n#if OKC_WGS && OKC_WGSPR\n  __threadfence_block();\n  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges\n#endif\n  DBG(1);\n  {  // the previous launch\'s ss copy and barrier counter are the next launch\'s: zero them now (this CTA\'s share)\n    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n  }\n#if OKC_WS\n  if (tid == 0) {  // this CTA\'s unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n    const int G = gridDim.x, c = blockIdx.x;\n    const float rG = __fdividef(1.f, (float)G);\n    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n#endif\n#if OKC_WGS\n  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)\n    const int nj = S.njF + S.njS;\n    int carry = 0;\n    for (int j0 = 0; j0 < nj; j0 += 32) {\n      const int j = j0 + tid;\n      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;\n      int incl = v;\n#pragma unroll\n      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }\n      if (j < nj) S.jrow[j] = carry + incl - v;\n      carry += __shfl_sync(0xffffffffu, incl, 31);\n    }\n    if (tid == 0) S.jrow[nj] = carry;\n  }\n  CTA_SYNC();\n  // phase A\'s x into xA (this CTA\'s share of the rows), before the producers start: its x reads do not queue behind\n  // every CTA\'s first weight stages\n  wgs_gather(p, S);\n  DBG(7);\n  WS_JOIN();\n  DBG(6);\n  {\n    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n    int seq = 0;\n    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n  }\n#else\n  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n#endif\n#if OKC_WS\n  WS_JOIN();  // the helper\'s last epilogue stores are visible\n#endif\n  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still\n  // waits for this whole grid to finish and its writes to be visible)\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n  // grid barrier on this launch\'s counter (two, by generation: the other one was zeroed by this launch for the next);\n  // the last arriver bumps the generation, which no CTA of this launch reads again\n#if OKC_FINPF && OKC_FINCOL && !OKC_WGS\n  // the CTA\'s first final item: its routing (pos) and route weights are inputs, so they load before the barrier\n  constexpr int FSEGP = HD / (NT * 4);\n  const int itp = blockIdx.x;\n  const bool hasp = itp < p.M * FSEGP;\n  const int tkp = hasp ? itp / FSEGP : 0;\n  int qvp[8];\n  float wvp[8];\n#pragma unroll\n  for (int s = 0; s < 8; ++s) {\n    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;\n    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;\n  }\n#endif\n  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);\n    __threadfence();\n    const int old = atomicAdd(bc, 1);\n    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);\n    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n  }\n  CTA_SYNC();\n  DBG(4);\n  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA\'s share)\n  {\n    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;\n    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters\n    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n#if OKC_WGS\n    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;\n    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;\n#endif\n  }\n  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_WGSFINSS\n  // item = one token\'s whole row (thread = 8 columns): pass 1 sums each local route\'s y^2 over the row (warp\n  // shuffles, then the 16 warps\' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)\n  {\n    static_assert(HD == NT * 8, "8 columns per thread");\n    float* red = reinterpret_cast<float*>(xbuf);\n    const int lane = tid & 31, warp = WARP_IDX(tid);\n    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n      int qv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n      float ps[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        ps[sl] = 0.f;\n        if (qv[sl] >= 0) {\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;\n        }\n      }\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {  // uniform over the CTA (one token)\n#pragma unroll\n          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);\n        }\n      if (lane == 0) {\n#pragma unroll\n        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];\n      }\n      CTA_SYNC();\n      if (tid < 8) {\n        float tsum = 0.f;\n        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];\n        red[NW * 8 + tid] = tsum;\n      }\n      CTA_SYNC();\n      float acc[8];\n#pragma unroll\n      for (int i = 0; i < 8; ++i) acc[i] = 0.f;\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {\n          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;\n          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;\n        }\n      uint4 o;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);\n#pragma unroll\n      for (int i = 0; i < 4; ++i) {\n        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n        ow[i] = *reinterpret_cast<uint32_t*>(&v);\n      }\n      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n      CTA_SYNC();  // red is reused by the next token\n    }\n  }\n#elif OKC_FINCOL\n  // column-parallel: item = (token, 1024-column segment); thread = 4 columns\n  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)\n  int it0 = blockIdx.x;\n  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math\n    const int col = (itp % FSEG) * (NT * 4) + tid * 4;\n    float ssv[8];\n    float4 yv[8];\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;\n      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);\n    }\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      if (qvp[s] >= 0) {\n        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);\n        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n  }\n  it0 += gridDim.x;\n  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {\n    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));\n        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;\n  }\n#else\n  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n    float acc[16];\n#pragma unroll\n    for (int i = 0; i < 16; ++i) acc[i] = 0.f;\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;\n#pragma unroll\n        for (int i = 0; i < 4; ++i) {\n          float4 v = __ldcg(src + i);\n          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;\n        }\n      }\n    }\n    uint4 o[2];\n    uint32_t* ow = reinterpret_cast<uint32_t*>(o);\n#pragma unroll\n    for (int i = 0; i < 8; ++i) {\n      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n      ow[i] = *reinterpret_cast<uint32_t*>(&v);\n    }\n    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);\n    dst[0] = o[0]; dst[1] = o[1];\n  }\n#endif\n  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered\n}\n\n}  // namespace okc\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\nstatic size_t okc_smem_bytes() {\n  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;\n#if OKC_WGS\n  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;\n#endif\n  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);\n  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n  return base + (xa > xb ? xa : xb);\n}\n\nstatic int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {\n  okc::Params p;\n  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();\n  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();\n  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();\n  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();\n  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }\n  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)\n#if OKC_WGS\n  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");\n  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,\n              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");\n  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();\n#else\n  p.wAs = nullptr; p.wBs = nullptr;\n#endif\n  p.zero = 0;\n  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();\n  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;\n#if OKC_WGS && OKC_RLOAD\n  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");\n#endif\n  int* ib = (int*)scratch[0].data_ptr();\n  p.ctr = ib; p.doneA = ib + 8;\n  p.ss = (float*)scratch[1].data_ptr();\n  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");\n  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);\n  p.ybuf = p.ss + okc::EMAX;\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n#if OKC_WGS\n  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,\n              "OKC_WGS needs xA / xB after inter (scratch[2])");\n  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;\n  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;\n  p.xready = ib + 8 + 512;\n  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;\n#else\n  p.xA = p.xB = nullptr;\n  p.xready = nullptr;\n  p.doneAs = nullptr;\n#endif\n  p.out = (__nv_bfloat16*)out.data_ptr();\n  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n  static bool attr = false;\n  size_t sm = okc_smem_bytes();\n  if (!attr) {\n    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    (void)e0; (void)e1;\n    attr = true;\n  }\n  cudaStream_t st = at::cuda::getCurrentCUDAStream();\n  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which\n  // leaves the skinny loop\'s code and registers as tuned\n  if (g_okc_pdl) {\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3((unsigned)grid);\n    cfg.blockDim = dim3(okc::NTL);\n    cfg.dynamicSmemBytes = sm;\n    cfg.stream = st;\n    cudaLaunchAttribute at[1];\n    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[0].val.programmaticStreamSerializationAllowed = 1;\n    cfg.attrs = at;\n    cfg.numAttrs = 1;\n    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);\n    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);\n    return;\n  }\n  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);\n  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);\n}\n\n\n'
_native_moe_CPP_SRC = '\n#include <torch/extension.h>\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);\nvoid okc_set_pdl(int64_t on);\n'
_native_moe__EXTS = {}

def _native_moe__ext(defines=None):
 defines = tuple(_native_moe_MAIN_DEFINES if defines is None else defines)
 if _native_moe_DQ7 and '-DOKC_DQ7=1' not in defines:
  defines = defines + ('-DOKC_DQ7=1',)
 if defines not in _native_moe__EXTS:
  src = _native_moe_CUDA_SRC
  defs = [d for d in defines if not d.startswith('@')]
  if torch.cuda.get_device_capability()[0] < 9:
   defs = [d for d in defs if not d.startswith('-DOKC_NS=') and (not d.startswith('-DOKC_NSF='))]
  for d in defines:
   if d.startswith('@'):
    src = _native_moe_ALT_SRCS[d[1:]]
  (major, minor) = torch.cuda.get_device_capability()
  arch = f'{major}{minor}' + ('a' if major == 9 else '')
  tag = '_'.join((d.replace('=', '').replace('-D', '').replace('@', 'src').lower() for d in defines)) or 'main'
  _native_moe__EXTS[defines] = load_inline(name=f'okc_moe_w4a16_v216x_pair1dq8_{tag}_sm{arch}', cpp_sources=_native_moe_CPP_SRC, cuda_sources=src, functions=['okc_run', 'okc_set_pdl'], extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo', *([] if any((d.startswith('-DOKC_NS=') for d in defs)) else [f'-DOKC_NS={(_native_moe_MAIN_NS if major >= 9 else 1)}']), *([] if any((d.startswith('-DOKC_NSF=') for d in defs)) else [f'-DOKC_NSF={(3 if major >= 9 else 1)}']), *defs], verbose=False)
  _native_moe__EXTS[defines].okc_set_pdl(1 if _native_moe_PDL else 0)
 return _native_moe__EXTS[defines]

def _native_moe__scales_prmt_exact(*sfs):
 for sf in sfs:
  b = sf.contiguous().view(torch.uint8)
  if bool(((b < 8) | (b > 126)).any()):
   return False
 return True

def _native_moe__stored_pos():
 pos = {}
 for reg in range(4):
  for up in range(2):
   for (bit, ob) in enumerate((6, 7, 8, 15)):
    output_bit = ob + 16 * up
    pos[reg, up, bit] = (output_bit - (0, 10, 13)[reg]) % 32 if reg < 3 else (4, 0, 1, 3)[bit] + 16 * up
 assert sorted(pos.values()) == list(range(32))
 return pos
_native_moe__POS = _native_moe__stored_pos()

def _native_moe__split_nibbles(codes, rows):
 c = codes.index_select(0, rows)
 nib = torch.stack((c & 15, c >> 4), -1).reshape(c.shape[0], -1)
 return (nib, c.shape[0])

def _native_moe__scatter_code_bits(word, code, reg, up):
 for b in range(4):
  word |= (code >> b & 1) << _native_moe__POS[reg, up, b]
 return word

def _native_moe__pack_codes(codes, rows):
 (nib, R) = _native_moe__split_nibbles(codes, rows)
 nb = R // 16
 n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)
 n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)
 word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
 for (Rg, (h, j0)) in enumerate(_native_moe__REG_HJ):
  for up in range(2):
   code = n[..., h, j0 + up]
   word = _native_moe__scatter_code_bits(word, code, Rg, up)
 word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
 return word.contiguous().view(-1)

def _native_moe__pack_scales(sf, rows):
 s = sf.view(torch.uint8).index_select(0, rows)
 R = s.shape[0]
 nb = R // 16
 s = s.view(nb, 2, 8, 8, 8, 4)
 s = s.permute(0, 3, 2, 5, 4, 1)
 return s.contiguous().view(-1)

def _native_moe__fat_order(codes, scales):
 nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
 c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
 c = c.view(torch.uint8).view(nub, 16, 32768)
 s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
 return torch.cat([c, s], dim=2).contiguous().view(-1)

def _native_moe__unit_order(codes, scales):
 nblk = codes.numel() // (8 * 8 * 32 * 4)
 c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
 s = scales.contiguous().view(nblk, 4096)
 return torch.cat([c, s], dim=1).contiguous().view(-1)

def _native_moe__pack_codes_nat(codes, rows):
 (nib, R) = _native_moe__split_nibbles(codes, rows)
 nb = R // 16
 n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)
 n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
 word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
 for (Rg, (h, q)) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
  for up in range(2):
   code = n[..., h, q, up]
   word = _native_moe__scatter_code_bits(word, code, Rg, up)
 word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
 return word.contiguous().view(-1)

def _native_moe__wgs_copy(codes, sf, rows):
 wn = _native_moe__pack_codes_nat(codes, rows)
 R = rows.numel()
 (nb, nub) = (R // 16, R // 256)
 w = wn.view(nb, 8, 8, 8, 4, 4)
 w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
 w = w.view(nub, 4, 4, 16, 4, 4, 32)
 w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()
 cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
 s = sf.view(torch.uint8).index_select(0, rows)
 s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
 s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)
 return torch.cat([cb, s], dim=3).contiguous().view(-1)

def _native_moe__main_defines(sfx):
 if not sfx and '-DOKC_WGS=1' in tuple(_native_moe_MAIN_DEFINES):
  return _native_moe_MAIN_FALLBACK_DEFINES
 return _native_moe_MAIN_DEFINES

def _native_moe__wgs_on():
 return any(('-DOKC_WGS=1' in tuple(d) for d in (_native_moe_MAIN_DEFINES, _native_moe_TINY_DEFINES, _native_moe_TINY2_DEFINES, _native_moe_MID_DEFINES)))

def _native_moe_prepare(weights):
 dev = weights['w13'].device
 (lo, cnt) = (_native_moe_LOCAL_START, _native_moe_LOCAL_EXPERTS)
 I = weights['w13'].shape[1] // 2
 H = weights['w2'].shape[1]
 assert I == _native_moe_HIDDEN and H == _native_moe_HIDDEN
 assert weights['post_norm_w'].dtype == torch.bfloat16
 assert weights['w13_gs'].dtype == torch.float32 and weights['w2_gs'].dtype == torch.float32
 rb = torch.arange(I // 8, device=dev)
 g = torch.arange(8, device=dev)
 rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
 rowsB = torch.arange(H, device=dev)
 (wA, sA, wB, sB) = ([], [], [], [])
 for e in range(lo, lo + cnt):
  wA.append(_native_moe__pack_codes(weights['w13'][e], rowsA))
  sA.append(_native_moe__pack_scales(weights['w13_sf'][e], rowsA))
  wB.append(_native_moe__pack_codes(weights['w2'][e], rowsB))
  sB.append(_native_moe__pack_scales(weights['w2_sf'][e], rowsB))
 lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float() * 2.0 ** 64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 65535
 lut = (lut | lut << 16).to(torch.int32)
 state = {'w': [torch.cat(wA), torch.cat(sA), weights['w13_gs'][lo:lo + cnt].contiguous(), torch.cat(wB), torch.cat(sB), weights['w2_gs'][lo:lo + cnt].contiguous(), lut.contiguous(), (weights['post_norm_w'].float() + 1.0).to(torch.bfloat16).contiguous(), torch.cat([_native_moe__fat_order(a, b) for (a, b) in zip(wA, sA)]), torch.cat([_native_moe__fat_order(a, b) for (a, b) in zip(wB, sB)]), torch.cat([_native_moe__unit_order(a, b) for (a, b) in zip(wA, sA)]), torch.cat([_native_moe__unit_order(a, b) for (a, b) in zip(wB, sB)])], 'lo': lo, 'cnt': cnt, 'scratch': _native_moe__make_scratch(dev), 'grid': torch.cuda.get_device_properties(dev).multi_processor_count}
 if _native_moe__wgs_on():
  state['w'].append(torch.cat([_native_moe__wgs_copy(weights['w13'][e], weights['w13_sf'][e], rowsA) for e in range(lo, lo + cnt)]))
  state['w'].append(torch.cat([_native_moe__wgs_copy(weights['w2'][e], weights['w2_sf'][e], rowsB) for e in range(lo, lo + cnt)]))
 prmt = _native_moe_PRMT_SCALES and _native_moe__scales_prmt_exact(weights['w13_sf'][lo:lo + cnt], weights['w2_sf'][lo:lo + cnt])
 state['sfx'] = ('-DOKC_PRMTSC=1',) if prmt else ()
 mods = [tuple(_native_moe__main_defines(state['sfx'])) + state['sfx']]
 for d in (_native_moe_TINY_DEFINES, _native_moe_TINY2_DEFINES, _native_moe_MID_DEFINES):
  if d and (state['sfx'] or '-DOKC_WGS=1' not in tuple(d)):
   mods.append(tuple(d) + state['sfx'])
 for d in mods:
  _native_moe__ext(d)
 torch.cuda.synchronize()
 return state

def _native_moe__make_scratch(dev):
 return [torch.zeros(8 + 1024 + _native_moe_EMAX, dtype=torch.int32, device=dev), torch.zeros(_native_moe_EMAX + _native_moe_EMAX * _native_moe_HIDDEN + _native_moe_WGS_FIXF, dtype=torch.float32, device=dev), torch.zeros((_native_moe_EMAX + 2 * _native_moe_WGS_XROWS) * _native_moe_HIDDEN, dtype=torch.bfloat16, device=dev)]

def _native_moe_kernel_fn(data, state):
 cfg = data['config']
 x = data['hidden_states'].contiguous()
 M = x.shape[0]
 assert cfg['local_start'] == state['lo'] and cfg['local_count'] == state['cnt']
 ids = data['topk_ids'].to(torch.int32).contiguous()
 if ids.data_ptr() & 15:
  ids = ids.clone()
 tw = data['topk_weights'].to(torch.float32).contiguous()
 out = torch.empty((M, _native_moe_HIDDEN), dtype=torch.bfloat16, device=x.device)
 sfx = state.get('sfx', ())
 if _native_moe_TINY_DEFINES and M <= _native_moe_TINY_MAX_M:
  ext = _native_moe__ext(tuple(_native_moe_TINY_DEFINES) + sfx)
 elif _native_moe_TINY2_DEFINES and M <= _native_moe_TINY2_MAX_M and (sfx or '-DOKC_WGS=1' not in tuple(_native_moe_TINY2_DEFINES)):
  ext = _native_moe__ext(tuple(_native_moe_TINY2_DEFINES) + sfx)
 elif _native_moe_MID_DEFINES and M <= _native_moe_MID_MAX_M:
  ext = _native_moe__ext(tuple(_native_moe_MID_DEFINES) + sfx)
 else:
  ext = _native_moe__ext(tuple(_native_moe__main_defines(sfx)) + sfx)
 for m0 in range(0, M, _native_moe_CHUNK_TOKENS):
  m1 = min(M, m0 + _native_moe_CHUNK_TOKENS)
  ext.okc_run(state['w'], state['scratch'], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1], state['lo'], state['cnt'], state['grid'])
 return out
'Gated attention decode for Muse Spark Open: one fused CUDA kernel per call.\n\nAlgorithm\n---------\nFor each request, the kernel computes gated attention over the request\'s cached\nkeys/values plus the block\'s own new tokens:\n\n    O = rmsnorm(softmax(Q K^T * sm_scale) V) * sigmoid(gate)\n\nwith Q/K first RMS-normalized (Q additionally scaled by qk_scale_factor), RoPE\napplied on sliding-window layers, and K/V quantized to FP8 (E4M3) in the\ncache. QK^T runs on FP8 tensor-core MMAs (queries rounded to E4M3, as FA3\ndoes) and PV on FP16 MMAs (P in FP16, V exact in FP16) -- except q_len 8 on\nsm_90, where PV is an FP8 wgmma (P in E4M3, as SGLang\'s FA3 FP8 path) -- with\nan online softmax, so no full score matrix is ever materialized.\n\nDesign\n------\n* One CTA per (request, pair of KV heads, split-KV split). A "pair" is two KV\n  heads, i.e. 16 query heads: the kernel\'s unit of work.\n* Each CTA builds the block\'s new K/V (normalized, RoPE, FP8) and writes them\n  to the cache, then streams its range of the paged FP8 cache through shared\n  memory in 64-key tiles (cp.async, multi-stage pipeline).\n* Splits are merged with FP32 atomics after rescaling to a fixed reference\n  (QK-norm bounds the logits); the last CTA of each (request, KV pair)\n  normalizes, applies rmsnorm * sigmoid(gate), writes the output, then zeroes\n  the workspace for the next call.\n* Launch shapes: per-request split-KV, per-request clusters (DSMEM\n  reduction), a flat kernel (one CTA pass over all requests\' keys, for full\n  layers), and a cooperative finalize (every CTA finalizes a slice) for grids\n  known resident. The host picks one in _plan(); the kernel itself is built\n  once per launch mode (the MODE template parameter), so each build carries\n  only its own epilogue.\n* Fixed build: sm_90a only. There are no tuning switches left in the\n  CUDA source; the taken path is the only path.\n* CUDA-graph capturable: no host syncs in the timed path. The module-level\n  caches hold only device properties, derived constants (RoPE frequencies),\n  and scratch buffers (the workspace is zeroed by the kernel itself between\n  calls), never call inputs or results.\n\nEntry point: kernel_fn(data) -> output tensor. Shapes the CUDA path does not\ncover fall back to _fallback(), a plain PyTorch implementation of the same\nmath.\n\nCUDA layout: Cfg/Params/Smem (launch config, kernel\narguments, shared-memory layout), small device helpers (cp.async, mbarrier,\nwgmma, conversions, cluster/DSMEM), the per-segment main loop (segment()),\nthe finalizes (sum_copies, fin_nc1, flat_finalize), attn_kernel, then the\nhost launchers (launch, run) at the end.\n'
from typing import TypedDict
import torch
_native_attn_HEAD_DIM = 64
_native_attn_Q_HEADS_PER_PAIR = 16
_native_attn_SUPPORTED_Q_LENS = (1, 8)
_native_attn_MAX_BATCH = 512
_native_attn_MAX_REQUEST_PAIRS = 4096
_native_attn_TILE_KEYS = 64
_native_attn_INDEX_LIMIT = 1 << 31
_native_attn_WORKSPACE_MIN_ROWS = 1 << 15
_native_attn_LOG2E = 1.4426950408889634
_native_attn_COOP_BIT = 1
_native_attn_PDL_BIT = 16
_native_attn_PLAN_HEADROOM_NUM = 21
_native_attn_PLAN_HEADROOM_DEN = 22

class _native_attn_Config(TypedDict):
 batch: int
 q_len: int
 num_q_heads: int
 num_kv_heads: int
 head_dim: int
 page_size: int
 window: int
 rope_theta: float
 qk_scale_factor: float
 sm_scale: float
 kv_scale: float

class _native_attn_Inputs(TypedDict):
 qkv: torch.Tensor
 positions: torch.Tensor
 cache_loc: torch.Tensor
 cache_seqlens: torch.Tensor
 cu_seqlens_q: torch.Tensor
 scheduler_metadata: torch.Tensor | None
 k_cache: torch.Tensor
 v_cache: torch.Tensor
 k_scale: torch.Tensor
 v_scale: torch.Tensor
 page_table: torch.Tensor
 seq_lens: torch.Tensor
 config: _native_attn_Config
_native_attn_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <type_traits>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <cuda.h>\n#include <stdint.h>\n\nnamespace {\n\n__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop\'s code)\n#define SPIN_T0 unsigned _spin_n = 0   /* (an iteration count: a %globaltimer read per poll slowed the coop barriers ~1%) */\n#define SPIN_CHECK do { if (++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */\ntemplate <int L, bool F>\nstruct Cfg {\n  static constexpr int NT = (L == 1) ? 128 : 256;   // consumer threads per CTA\n  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the\n  // consumers never stall in cp.async issue; the stages are handed over with mbarriers\n  static constexpr bool WS = F;\n  static constexpr int NP = WS ? 128 : 0;   // producer threads\n  static constexpr int NTX = NT + NP;                     // threads per CTA\n  static constexpr int MINB = 2;   // CTAs per SM the registers must allow\n  // setmaxnreg split of the CTA\'s registers (launch: 65536 / (MINB * NTX), rounded down to 8)\n  static constexpr int BASEREGS = (65536 / (2 * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (2 * NTX)) / 8 * 8;\n  static constexpr int PREGS = (L == 1) ? 56 : 32;\n  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248\n                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;\n  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,\n  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request\n  static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;\n  // producer passes: NP/8 rows each must tile the 64-row tile exactly (else rows go missing and the consumers wait\n  // forever on the stage barrier)\n  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),\n                "producer threads must cover a tile in whole passes");\n};\nconstexpr int TILE = 64;\nconstexpr int STAGES = 4;\nconstexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64\nconstexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)\nconstexpr int IDX_SLOTS = 2 * STAGES;\nconstexpr int MAXB = 512;               // flat partition: requests per call\nconstexpr int MAXSEG = 64;              // flat partition: segments per CTA\n\nstruct Params {\n  const __nv_bfloat16* qkv;\n  const int* seq_lens;\n  const int64_t* cache_loc;\n  uint8_t* k_cache;\n  uint8_t* v_cache;\n  const int* page_table;\n  const float* k_scale;\n  const float* v_scale;\n  const float* inv_freq;\n  __nv_bfloat16* out;\n  float* ws_acc;\n  float* ws_l;\n  int* ws_cnt;\n  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;\n  long long* dbg;      // optional per-CTA phase timestamps (globaltimer), null in normal runs\n  int cs, ncl;          // cluster mode: cs CTAs per cluster (DSMEM reduction), ncl clusters per request\n  int ld_qkv, row_bytes;\n  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)\n  int coop;             // per-request grids that are all resident: every CTA finalizes a slice of the rows\n  int lp;               // log2(page size)\n  int lrb;              // log2(row bytes of a cache row: Hkv * 64)\n  int off32;            // both caches below 4 GB (32-bit cache offsets); else the 64-bit path\n  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n};\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return (uint32_t)__cvta_generic_to_shared(p);\n}\n__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes));\n}\n__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {\n  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills\n  asm volatile("{\\n .reg .pred q;\\n setp.ne.b32 q, %3, 0;\\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\\n}\\n"\n               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));\n}\n__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n" ::"r"(dst), "l"(src));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n\n// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----\n__device__ __forceinline__ void mbar_fence_init() {\n  asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n// unbounded wait for a phase (producer / consumer hand-off)\n__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {\n  asm volatile("{\\n.reg .pred q;\\nWAIT_%=:\\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\\n@!q bra WAIT_%=;\\n}\\n"\n               ::"r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n// the mbarrier receives one arrival when all of this thread\'s earlier cp.async copies have landed\n__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {\n  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier\n__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"\n               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n// 16 zero bytes into shared memory through the same cp.async path (no global read)\n__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\\n" ::"r"(dst), "l"(any_src));\n}\n// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer\n// barrier over one warpgroup (named barriers 3 and 4)\n__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\\n" ::"r"(id), "r"(n) : "memory"); }\n__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\\n" ::"r"(id), "r"(n) : "memory"); }\n__device__ __forceinline__ void wgsync(int wg) {\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n}\ntemplate <bool WS, int NT>\n__device__ __forceinline__ void csync() {\n  if constexpr (WS) asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory");\n  else __syncthreads();\n}\n\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));\n  return d;\n}\n// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP\'s operand (.H1), no SHF\n__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 t;\\n\\tmov.b32 {_, t}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, t;\\n\\t}" : "=r"(d) : "r"(x));\n  return d;\n}\n// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)\n__device__ __forceinline__ uint32_t cvt2(uint32_t x) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 t;\\n\\tmov.b32 {t, _}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, t;\\n\\t}" : "=r"(d) : "r"(x));\n  return d;\n}\n__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {\n  __half2 h = __floats2half2_rn(lo, hi);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n// 2^lo, 2^hi as a packed f16 pair: the inputs rounded to f16 (|x| <= 8 here: error <= 2^-8), one MUFU op\n__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {\n  return pack_h2(ex2(lo), ex2(hi));\n}\n// Pinned SGLang materializes BF16 Q/K and BF16 attention output.\n__device__ __forceinline__ float bf16_round(float x) {\n  return __bfloat162float(__float2bfloat16_rn(x));\n}\n__device__ __forceinline__ float attn_out_bf16(float x) {\n  return bf16_round(x);\n}\n__device__ __forceinline__ uint8_t to_e4m3(float x) {\n  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);\n}\n\n__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                        uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\n      "{%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n\n__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }\n__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }\n\n// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.\n__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,\n                                             uint32_t* b1) {\n  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;\n  const int k0 = kb + 2 * tig;\n  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);\n  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);\n  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);\n  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);\n  uint32_t x;\n  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);\n  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);\n  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);\n  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);\n  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);\n  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);\n  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);\n  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);\n}\n\n// FP8 PV uses the same key permutation in both operands. For each K16\n// half, lane tig supplies physical keys 2*tig + {0,1,8,9}. This keeps\n// the old conflict-free V loads and packs P directly, without shuffles.\n__device__ __forceinline__ void load_v_frags_f8(const uint8_t* V, int kb, int hk, int g, int tig,\n                                              uint32_t* b0, uint32_t* b1) {\n  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    const int k0 = kb + 16 * h + 2 * tig;\n    const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);\n    const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);\n    const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);\n    const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);\n    uint32_t* dst = h ? b1 : b0;\n#pragma unroll\n    for (int z = 0; z < 2; ++z) {\n      const uint32_t a = z ? v0.y : v0.x, b = z ? v1.y : v1.x;\n      const uint32_t c = z ? v2.y : v2.x, d = z ? v3.y : v3.x;\n      const uint32_t e01 = prmt(a, b, 0x5140), e23 = prmt(c, d, 0x5140);\n      const uint32_t o01 = prmt(a, b, 0x7362), o23 = prmt(c, d, 0x7362);\n      dst[4 * z + 0] = prmt(e01, e23, 0x5410);\n      dst[4 * z + 1] = prmt(e01, e23, 0x7632);\n      dst[4 * z + 2] = prmt(o01, o23, 0x5410);\n      dst[4 * z + 3] = prmt(o01, o23, 0x7632);\n    }\n  }\n}\n\n__device__ __forceinline__ void mma_f8pv(float* c, uint32_t a0, uint32_t a2,\n                                       uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, "\n      "{%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(0u), "r"(a2), "r"(b0), "r"(b1));\n}\n\n// the merge / barrier fences as fence.acq_rel.gpu (MEMBAR.ALL.GPU) instead of __threadfence\'s\n// fence.sc.gpu (MEMBAR.SC.GPU + ERRBAR + CCTL.IVALL): the message-passing pattern needs release / acquire only\n__device__ __forceinline__ void gpu_fence() {\n  __threadfence();\n}\n__device__ __forceinline__ void red_f32(float* a, float x) {\n  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");\n}\n__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {\n  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");\n}\n\n\n__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\\n"\n               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));\n}\n// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the\n// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)\n__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {\n  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)\n               : "memory");\n}\n\n\n// ---- per-warp phase profile of the main loops (diagnostic builds only: -DOKC_PROF=1) ----\n#define PROF_INIT\n#define PROF_MARK(k)\n#define PROF_TILE\n#define PROF_DUMP\n\n// ---- sm_90 warpgroup MMA helpers ----\n__device__ __forceinline__ void fence_proxy_async() {\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n}\n__device__ __forceinline__ void wgmma_fence() {\n  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");\n}\n__device__ __forceinline__ void wgmma_commit() {\n  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");\n}\ntemplate <int N>\n__device__ __forceinline__ void wgmma_wait() {\n  asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory");\n}\n__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }\n// K-major, 128-byte swizzled shared-memory matrix descriptor (8-row groups 1024 bytes apart)\n__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {\n  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n// D[64x64] (+)= A[64x32] (e4m3, registers) * B[64x32]^T (e4m3, shared memory descriptor)\n__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                                     uint64_t desc, int scale_d) {\n  asm volatile(\n      "{\\n.reg .pred p;\\nsetp.ne.b32 p, %37, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "\n      "{%32, %33, %34, %35}, %36, p, 1, 1;\\n}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));\n}\n// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)\n__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {\n  asm volatile(\n      "{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "\n      "%32, %33, p, 1, 1;\\n}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n// the same with scale-d = 0 and write-only accumulators: the previous tile\'s (masked) scores do not feed this\n// wgmma, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline (warning C7515, which\n// serializes every wgmma of the function)\n// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {\n  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);\n}\n// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators\n__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {\n  asm volatile(\n      "{\\n"\n      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "\n      "{%36, %37, %38, %39}, %40, 1, 1, 1;\\n}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n}\n// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):\n// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code\n__device__ __forceinline__ int fdiv_i(int x, int d) {\n  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)\n  const float rd = __fdividef(1.f, __int2float_rn(d));\n  int q = __float2int_rz(__int2float_rn(x) * rd);\n  const int r = x - q * d;\n  q += (r >= d) - (r < 0);\n  return q;\n}\n// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,\n// below the BF16 output\'s rounding) or exp + fast reciprocal\n__device__ __forceinline__ float sigm(float x) {\n  return __fdividef(1.f, 1.f + __expf(-x));\n}\n// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;\n// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.\n__device__ __forceinline__ void pdl_wait() {\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n}\n__device__ __forceinline__ void pdl_launch_dependents() {\n  // (no early trigger at the end of the tile loop -- the dependent kernel launches when this grid\n  // exits; for decode_layer\'s chain A/B, no effect on standalone launches)\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n}\n// 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies\n// in the once-per-launch address math\ntypedef int ix_t;\n__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {\n  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);\n  return base + row * p.row_bytes + col_bytes;\n}\n// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the\n// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)\ntemplate <int N, int NTH, class F>\n__device__ __forceinline__ void tfor(int tid, F&& f) {\n#pragma unroll\n  for (int e0 = 0; e0 < N; e0 += NTH) {\n    const int e = e0 + tid;\n    if ((N % NTH) == 0 || e < N) f(e);\n  }\n}\n// four floats -> four e4m3 bytes (x0 in the low byte), saturating\n__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {\n  uint16_t lo, hi;\n  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));\n  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));\n  return (uint32_t)lo | ((uint32_t)hi << 16);\n}\n// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)\n__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {\n  __half2_raw a, b;\n  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);\n  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);\n  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);\n  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);\n  return (lo & 0xffffu) | (hi << 16);\n}\n\n\n__device__ __forceinline__ float quad_max(float v) {\n  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));\n  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));\n  return v;\n}\n__device__ __forceinline__ float quad_sum(float v) {\n  v += __shfl_xor_sync(0xffffffffu, v, 1);\n  v += __shfl_xor_sync(0xffffffffu, v, 2);\n  return v;\n}\n__device__ __forceinline__ float oct_sum(float v) {\n  v += __shfl_xor_sync(0xffffffffu, v, 1);\n  v += __shfl_xor_sync(0xffffffffu, v, 2);\n  v += __shfl_xor_sync(0xffffffffu, v, 4);\n  return v;\n}\n\n// ---- thread-block clusters (sm_90: DSMEM; elsewhere emulated through global memory for testing) ----\n__device__ __forceinline__ int cl_rank(const Params& p) {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return (int)r;\n}\n__device__ __forceinline__ int cl_gid(const Params& p) {\n  return (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x / p.cs) + blockIdx.x / p.cs;\n}\n__device__ __forceinline__ void cl_sync(const Params& p, int phase) {\n  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");\n}\n// float4 at byte offset `off` of peer `rank`\'s partial buffer (`local` = this CTA\'s copy)\n__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {\n  uint32_t a = smem_u32(local + off_floats), r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));\n  float4 v;\n  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));\n  return v;\n}\n__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {\n  uint32_t a = smem_u32(local + off_floats), r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));\n  float v;\n  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));\n  return v;\n}\n\ntemplate <int L, bool F>\nstruct Smem {\n  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];\n  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];\n  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)\n  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)\n  int pgc;            // page-index fills issued in this CTA\'s earlier segments\n  int dbgn;           // debug builds: progress-mark count\n  int vidx;           // flat weighted partition: this CTA\'s slot (2 * SM + arrival rank), -1 = uniform\n  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)\n  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)\n  uint64_t empty[8];  // producer mode: stage consumed (one arrival per consumer warp)\n  uint64_t vtf[2];    // V^T slot written by the producers (one arrival per producer thread)\n  uint64_t vte[2];    // V^T slot read by both heads\' PV wgmmas (one arrival per consumer warp)\n  uint64_t qdone;     // consumers have read the q staging (S.kvq) and staged S.kn / S.vn\n  int claim[2];       // Cfg::F2: next unclaimed tile per KV head\n  int nxt[4][2];      // Cfg::F2: each warpgroup\'s next claimed tile (double buffered)\n  int idx[2 * Cfg<L, F>::STAGES * TILE];\n  // raw q rows of the block (bf16, prologue only) | L==8: the current tile\'s K and V in f16 (2 x 2 x 64 x 64)\n  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile\n  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to\n  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes\n  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : 18432];\n  // L == 8 with FP8 PV and the block\'s e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)\n  alignas(1024) uint8_t q8s[(L == 8) ? 8192 : 16];\n  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }\n  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token\n  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA\'s output slice (<= L*16/4 rows)\n  uint8_t kn[L * ROWB];        // new keys of the block, [j][2 heads x 64] e4m3\n  uint8_t vn[L * ROWB];\n  float cs[L * 32];\n  float sn[L * 32];\n  float ifreq[32];          // RoPE inverse frequencies (staged with cp.async)\n  float scl[4];             // k_scale, v_scale\n  long long cloc[L];        // cache rows of the block\'s tokens\n  int flag;\n  int sense;          // the cooperative barrier\'s flag, read early (prologue) with the constants\n  int cum[F ? MAXB + 1 : 1];\n  int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];\n  int nseg;\n  int fcs, fce;             // this CTA\'s key range (warp 0 -> everyone)\n  // flat dynamic chunks : the producer\'s segment list of its current chunk, and a queue that hands the\n  // segments (request, first key, end key, copy; request -1 = done) to the consumers\n  int pseg[F ? MAXSEG : 1][3];\n  int pnseg, pchunk;\n  int dq[F ? 16 : 1][4];\n  uint64_t dqf[F ? 16 : 1];\n  uint64_t dqe[F ? 16 : 1];\n};\n\n// Cluster mode, several clusters per request (Params::ncl > 1): add this CTA\'s reduced slice into the global\n// workspace; the last cluster to arrive sums the copies back into RB/RL and finalizes. Out of line: no scored shape\n// uses it, and keeping it out of the one-cluster epilogue keeps that code contiguous (cold instruction fetch).\ntemplate <int L, typename SM>\n__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,\n                                           const int row0, const int bp, float* RB, float* RL) {\n  constexpr int NT = Cfg<L, false>::NT;\n  const int tid = threadIdx.x;\n  bool fin = true;\n    const int clu = fdiv_i(blockIdx.x, CS);\n    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;\n    for (int e = tid; e < RPC * 16; e += NT) {\n      const int row = row0 + (e >> 4);\n      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4;\n      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);\n      red_v4(acc, a.x, a.y, a.z, a.w);\n    }\n    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);\n    __threadfence();\n    __syncthreads();\n    if (tid == 0) {\n      // every cluster of the request arrives (empty ones with zero partials)\n      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);\n      S.flag = (old == p.ncl - 1);\n    }\n    __syncthreads();\n    fin = S.flag;\n    if (fin) {\n      __threadfence();\n      for (int e = tid; e < RPC * 16; e += NT) {\n        const int row = row0 + (e >> 4);\n        constexpr int NCM = 4;\n        float4 v[NCM];\n#pragma unroll\n        for (int c = 0; c < NCM; ++c)\n          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4));\n        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n        for (int c = 0; c < NCM; ++c) {\n          if (c < p.nc) {\n            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;\n            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);\n          }\n        }\n        *reinterpret_cast<float4*>(&RB[e * 4]) = a;\n      }\n      for (int e = tid; e < RPC; e += NT) {\n        float a = 0.f;\n        for (int c = 0; c < p.nc; ++c) {\n          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;\n          a += __ldcg(lw);\n          *lw = 0.f;\n        }\n        RL[e] = a;\n      }\n      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;\n      __syncthreads();\n    }\n  return fin;\n}\n\n// Cluster mode epilogue. Every CTA of the cluster has written its rescaled partial (PB: [L*16][64] o * 2^m,\n// PL: [L*16] l * 2^m) into its own shared memory. CTA `rank` sums rows [rank*RPC, (rank+1)*RPC) over the\n// cluster through DSMEM, then either finalizes them (one cluster per request) or accumulates them into the\n// global workspace, where the last cluster to arrive for this slice finalizes.\ntemplate <int L, typename SM>\n__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,\n                                               const int n_keys, const int chunk) {\n  constexpr int NT = Cfg<L, false>::NT;\n  constexpr int STAGES = Cfg<L, false>::STAGES;\n  constexpr int IDX_SLOTS = 2 * STAGES;\n  const int tid = threadIdx.x;\n  const int CS = p.cs, rank = cl_rank(p);\n  const int RPC = fdiv_i(L * 16, CS);\n  const int pairs = p.Hkv >> 1;\n  const int bp = b * pairs + pair;\n  // S.k and S.v are adjacent: PB [L*16*64] | PL [L*16, padded to 1024] | RB [RPC*64] | RL [RPC]\n  float* PB = reinterpret_cast<float*>(S.k);\n  float* PL = PB + L * 16 * 64;\n  float* RB = PL + 1024;\n  float* RL = RB + RPC * 64;\n  constexpr int REGION = L * 16 * 64 + 1024;   // emulation: per-CTA scratch floats (PB, then PL at +L*16*64)\n  cl_sync(p, 1);\n  \n  const int row0 = rank * RPC;\n  if constexpr (L == 1) {   // compact reduction (this code runs once per launch from a cold instruction cache, so it is kept short):\n      // (row, slot) items -- 16 float4 slots of o and one for l -- times the CS peers; CS consecutive lanes load\n      // one item from every peer and sum it with shuffles. Two items per thread per pass keep loads in flight.\n    const int TOT0 = RPC * 17 * CS;\n    const int lcs = 31 - __clz(CS);\n#pragma unroll 1\n    int TOT = TOT0;\n       // dry pass: same code, no work\n    \n    for (int k0 = 0; k0 < TOT; k0 += 2 * NT) {   // warp-uniform trip count\n      float4 v[2];\n#pragma unroll\n      for (int u = 0; u < 2; ++u) {\n        const int k = k0 + u * NT + tid;\n        const int item = k >> lcs, q = k & (CS - 1);\n        const int rl = item / 17, slot = item - rl * 17;\n        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);\n        if (k < TOT) {\n          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);\n          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);\n        }\n      }\n#pragma unroll\n      for (int u = 0; u < 2; ++u) {\n        for (int off = 1; off < CS; off <<= 1) {\n          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);\n          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);\n          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);\n          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);\n        }\n        const int k = k0 + u * NT + tid;\n        const int item = k >> lcs, q = k & (CS - 1);\n        const int rl = item / 17, slot = item - rl * 17;\n        if (k < TOT && q == 0) {\n          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];\n          else RL[rl] = v[u].x;\n        }\n      }\n    }\n      } else\n  {\n  // compact: peers in groups of 4 (one loop body, warm after its first pass: this code runs once per launch)\n  for (int e = tid; e < RPC * 16; e += NT) {\n    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;\n    const bool lrow = (e & 15) == 0;\n    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);\n    float la = 0.f;\n#pragma unroll 1\n    for (int q0 = 0; q0 < CS; q0 += 4) {\n      float4 v[4];\n      float lv[4];\n#pragma unroll\n      for (int u = 0; u < 4; ++u) {\n        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);\n        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;\n      }\n#pragma unroll\n      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }\n    }\n    *reinterpret_cast<float4*>(&RB[e * 4]) = a;\n    if (lrow) RL[e >> 4] = la;\n  }\n  \n  }\n  \n  cl_sync(p, 2);   // peers are done reading this CTA\'s partial\n\n  bool fin = true;\n  if (p.ncl > 1) {\n    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);\n  }\n  if (fin) {\n    // rmsnorm * sigmoid(gate) for the slice: 16 threads per row, 4 dims each\n    \n    const float vscale = S.scl[1];\n    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {   // warp-uniform trip count (shuffles below)\n      const int e = eb + (tid & 31);\n      const bool act = e < RPC * 16;\n      const int rl = act ? (e >> 4) : 0, c4 = e & 15;\n      const int row = row0 + rl;\n      const int j = row >> 4, hh = row & 15;\n      const float lsum = act ? RL[rl] : 0.f;\n      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;\n      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);\n      float y[4] = {attn_out_bf16(a.x * inv), attn_out_bf16(a.y * inv), attn_out_bf16(a.z * inv), attn_out_bf16(a.w * inv)};\n      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];\n      ss += __shfl_xor_sync(0xffffffffu, ss, 1);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 2);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 4);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 8);\n      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);\n      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);\n      uint2 ov;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);\n#pragma unroll\n      for (int k = 0; k < 2; ++k) {\n        const float2 gt = __bfloat1622float2(g2[k]);\n        const float s0 = sigm(gt.x), s1 = sigm(gt.y);\n        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);\n        ow[k] = *reinterpret_cast<uint32_t*>(&r2);\n      }\n      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;\n    }\n    \n  }\n}\n\n// Gate rows of this CTA\'s cluster slice -> S.gst (cp.async, in the caller\'s commit group)\ntemplate <int L, typename SM>\n__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {\n  constexpr int NT = Cfg<L, false>::NT;\n  constexpr int STAGES = Cfg<L, false>::STAGES;\n  constexpr int IDX_SLOTS = 2 * STAGES;\n  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;\n  for (int e = threadIdx.x; e < RPC * 8; e += NT) {\n    const int rl = e >> 3, ch = e & 7;\n    const int row = row0 + rl, j = row >> 4, hh = row & 15;\n    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),\n               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);\n  }\n}\n\n// One (request b, KV pair) segment [start, end) of the request\'s key positions.\n// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in\n// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a\n// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).\ntemplate <int L>\n__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,\n                                           const bool act, float (&y)[8], float& lsum) {\n  if (p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout\'s census)\n    if (act) {\n      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;\n      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));\n      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);\n      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);\n      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;\n      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;\n      lsum += lc;\n      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);\n      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);\n      if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;\n    }\n    return;\n  }\n  if (p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy\'s loads in one round trip\n    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)\n      float4 xa[8], xb[8];\n      float lc[8];\n#pragma unroll\n      for (int c = 0; c < 8; ++c) {\n        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;\n        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));\n        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);\n        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);\n      }\n#pragma unroll\n      for (int c = 0; c < 8; ++c) {\n        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;\n        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;\n        lsum += lc[c];\n        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;\n        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);\n        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);\n        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;\n      }\n    }\n    return;\n  }\n#pragma unroll 1\n  for (int c0 = 0; c0 < p.nc; c0 += 4) {\n    float4 xa[4], xb[4];\n    float lc[4];\n#pragma unroll\n    for (int u = 0; u < 4; ++u) {\n      const int c = c0 + u;\n      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);\n      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);\n      lc[u] = 0.f;\n      if (act && c < p.nc) {\n        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;\n        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));\n        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);\n        lc[u] = __ldcg(p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);\n      }\n    }\n#pragma unroll\n    for (int u = 0; u < 4; ++u) {\n      const int c = c0 + u;\n      if (act && c < p.nc) {\n        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;\n        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;\n        lsum += lc[u];\n        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;\n        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);\n        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);\n        if (part == 0) p.ws_l[((bp * p.nc + c) * L + j) * 16 + hh] = 0.f;\n      }\n    }\n  }\n}\n\n// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,\n// reset the workspace. Not inlined: this runs once per launch and is otherwise fetched cold.\ntemplate <int L, int NT>\n__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,\n                                     const int dry) {\n  const int tid = threadIdx.x;\n  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a\n  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)\n  constexpr int PASSES = L * 16 / (NT / 8);\n  constexpr int GP = PASSES >= 2 ? 2 : 1;\n  const int part = tid & 7;\n#pragma unroll 1\n  for (int p0 = 0; p0 < PASSES; p0 += GP) {\n    float4 xa[GP], xb[GP];\n    float lc[GP];\n    uint4 graw[GP];\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n      const int j = rr >> 4, hh = rr & 15;\n      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;\n      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));\n      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);\n      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);\n      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n      const int j = rr >> 4, hh = rr & 15;\n      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;\n      if (!dry) {\n        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);\n        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);\n        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;\n      }\n      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;\n      float y[8] = {attn_out_bf16(xa[u].x * inv), attn_out_bf16(xa[u].y * inv), attn_out_bf16(xa[u].z * inv), attn_out_bf16(xa[u].w * inv),\n                    attn_out_bf16(xb[u].x * inv), attn_out_bf16(xb[u].y * inv), attn_out_bf16(xb[u].z * inv), attn_out_bf16(xb[u].w * inv)};\n      float ss = 0.f;\n#pragma unroll\n      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];\n      ss = oct_sum(ss);\n      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);\n      uint4 ov;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const float2 gt = __bfloat1622float2(g2[e]);\n        const float s0 = sigm(gt.x), s1 = sigm(gt.y);\n        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);\n        ow[e] = *reinterpret_cast<uint32_t*>(&r2);\n      }\n      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n    }\n  }\n}\n\n// the q_len 8 flat loop\'s rare new-row patch as a call, so ptxas cannot hoist its bounds math into the\n// common path of every tile\ntemplate <int L, int NT, bool PTR, class SM>\n__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {\n  SM& S = *Sp;\n  const int t0 = start + i * TILE;\n  const int lo = max(t0, seq), hi = min(t0 + TILE, end);\n  if (lo < hi) {\n    for (int e = tid; e < (hi - lo) * 8; e += NT) {\n      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;\n      const int sw = ((c ^ (r & 7)) << 4);\n      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);\n      *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);\n    }\n    if (L == 8) fence_proxy_async();\n  }\n  return lo < hi;\n}\n\ntemplate <int L, bool FLAT, int ROLE, int MODE = 0>\n__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,\n                                        const int start_, const int end_, const int seq_, const int n_keys_,\n                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,\n                                        const bool skip_init_ = false) {\n  // (returns false only in the builds, when the speculated length missed: nothing global was written yet and\n  // the caller re-runs the segment with the real length)\n  // per-segment values as lane-0 broadcasts : provably warp-uniform, so ptxas keeps the address math\n  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)\n#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)\n  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);\n  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);\n#undef UNI\n  // tb: tiles this CTA\'s pipeline has already run (flat producer mode: the stage ring and its mbarrier phases\n  // continue across segments, so the producer runs ahead into the next segment instead of draining)\n  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own\n  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,\n  // 3 cooperative; 0 = decided at run time (flat kernels)\n  // MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,\n  // the in-loop copies and the first-tile tests fold at compile time (scout\'s once-path census, s1 / vs1)\n  constexpr bool ONE = !FLAT && (MODE & 4) != 0;\n  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);\n  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;\n  constexpr int NT = Cfg<L, FLAT>::NT;\n  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile\'s V^T for the PV wgmma\n  constexpr int IDX_SLOTS = 2 * STAGES;\n  constexpr bool WS = Cfg<L, FLAT>::WS;\n  // (the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses\n  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS\'s canonical_warp_idx_sync idiom)\n  const int tid = threadIdx.x, lane = tid & 31, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);\n  const int g = lane >> 2, tig = lane & 3;\n  const int split = first ? 0 : 1;\n  const int pairs = p.Hkv >> 1;\n  const int bp = b * pairs + pair;\n  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)\n\n  if constexpr (WS) {\n    constexpr int NP = Cfg<L, FLAT>::NP;\n    constexpr int PR = NP / 8;            // rows per producer pass\n    constexpr int PK = TILE / PR;         // passes per tile\n    if (!skip_init_) {   // (the first segment\'s barriers were set up at kernel start)\n      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)\n        for (int st = 0; st < STAGES; ++st) {\n          mbar_init_n(&S.full[st], NP);\n          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile\n        }\n        \n        mbar_fence_init();\n      }\n      __syncthreads();\n    }\n    if constexpr (ROLE == 1) {\n      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,\n      // zeros past the range; the block\'s own new keys are patched in by the consumers\n      const int q = tid - NT;\n      const int ch = q & 7, r0 = q >> 3;\n      const int* ptr = p.page_table + (ix_t)b * p.max_pages;\n      const int cache_end = min(end, seq);\n      int pg[PK];\n      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread\'s rows");\n      const bool pfast = p.off32 && p.lp == 0;\n      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);\n      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);\n      // page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)\n      constexpr bool TMAPG_ON = (L == 1);\n      // page indices by TMA: one 64-entry box per tile, PD tiles ahead, PS slots, producer barrier per tile\n      constexpr int PS = 4, PD = 3;\n      int pgb = 0;   // fills issued by earlier segments (slot / phase base)\n      if constexpr (TMAPG_ON) {\n          \n          if (q == 0) {\n            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);\n            mbar_fence_init();\n          }\n          asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");\n        \n      }\n      // the 16-byte aligned window holding the tile\'s page indices; reads stay inside the table\'s allocation\n      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)\n      const long long pt_total = (long long)p.B * p.max_pages;\n      const long long pt_lim = (pt_total + 3) & -4;\n      auto fetch_pages = [&](int x) {\n        if (q == 0 && x < ntiles) {\n          const int slot = (pgb + x) % PS;\n          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);\n          const long long w0 = e0 & -4;\n          const long long w1 = min(pt_lim, (e0 + 64 + 3) & -4);\n          const int bytes = (int)(w1 - w0) * 4;\n          mbar_expect_tx(&S.pbar[slot], bytes);\n          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);\n        }\n      };\n      if constexpr (TMAPG_ON) {\n        for (int x = 0; x < PD; ++x) fetch_pages(x);\n      }\n      auto load_pages_t = [&](int x, int* dst) {\n        const int slot = (pgb + x) % PS;\n        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);\n        const int fp = (start + x * TILE) >> p.lp;\n        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page\n#pragma unroll\n        for (int k = 0; k < PK; ++k) {\n          const int pos = start + x * TILE + r0 + PR * k;\n          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;\n        }\n      };\n      auto load_pages = [&](int x, int* dst) {\n#pragma unroll\n        for (int k = 0; k < PK; ++k) {\n          const int pos = start + x * TILE + r0 + PR * k;\n          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;\n        }\n      };\n      if constexpr (!TMAPG_ON) load_pages(0, pg);\n      \n      // tile y\'s V (both KV heads) -> V^T slot y & 1 (72 x 64 per head: 64 dims + 8 rows of e4m3 ones for the\n      // row sums, keys permuted to the P register order, K-major, 64B swizzle), the layout the PV wgmma reads\n      for (int x = 0; x < ntiles; ++x) {\n        const int st = (tb + x) % STAGES;\n        int pgn[PK];\n        if constexpr (TMAPG_ON) {\n          fetch_pages(x + PD);\n          load_pages_t(x, pg);\n        } else {\n          load_pages(x + 1, pgn);   // in flight while this tile\'s copies issue\n        }\n        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);\n        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);\n        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row\n        // branches (the producer\'s instructions take issue slots from the consumers on the same SMSPs)\n        if (pfast && start + x * TILE + TILE <= cache_end) {\n#pragma unroll\n          for (int k = 0; k < PK; ++k) {\n            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;\n            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);\n            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);\n          }\n        } else\n#pragma unroll\n        for (int k = 0; k < PK; ++k) {\n          const int r = r0 + PR * k;\n          const int pos = start + x * TILE + r;\n          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);\n          if (pos < cache_end) {\n            if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)\n              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);\n              cp_async16(kb + so, p.k_cache + off, 16);\n              cp_async16(vb + so, p.v_cache + off, 16);\n            } else\n            {\n              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;\n              cp_async16(kb + so, p.k_cache + off, 16);\n              cp_async16(vb + so, p.v_cache + off, 16);\n            }\n          } else if (pos >= end) {\n            cp_async16_zero(kb + so, p.k_cache);\n            cp_async16_zero(vb + so, p.v_cache);\n          }\n        }\n        cp_async_mbar_arrive(&S.full[st]);\n        \n        PROF_TILE;\n        if constexpr (TMAPG_ON) {\n          asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS\n        } else {\n#pragma unroll\n          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];\n        }\n      }\n      \n      \n      PROF_DUMP;\n      \n        cp_wait<0>();\n        __syncthreads();   // matches the consumers\' end-of-segment barrier\n      \n      return true;\n    }\n  }\n  if constexpr (ROLE != 1) {\n\n  const int prow = tid >> 1, phalf = tid & 1;\n  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;\n  const int part = tid & 7;\n\n  // ---- producer helpers ----\n  // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.\n  // New-token rows come from S.kn / S.vn; rows past the range are zeroed.\n  const int cchunk = tid & 7, crow0 = tid >> 3;\n  constexpr int RPP = NT / 8;   // rows per pass\n  auto issue_kv = [&](int x, bool with_new) {\n    const int st = x % STAGES;\n    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];\n    bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence\n    // every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add\n    // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)\n    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {   // (in-loop issues only: the\n      // prologue\'s cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)\n      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);\n      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);\n#pragma unroll\n      for (int k = 0; k < TILE / RPP; ++k) {\n        const int r = crow0 + RPP * k;\n        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);\n        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;\n        cp_async16(kst + so, p.k_cache + off, 16);\n        cp_async16(vst + so, p.v_cache + off, 16);\n      }\n      return;\n    }\n#pragma unroll\n    for (int k = 0; k < TILE / RPP; ++k) {\n      const int r = crow0 + RPP * k;\n      const int pos = start + x * TILE + r;\n      const int sw = ((cchunk ^ (r & 7)) << 4);\n      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;\n      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;\n      if (pos < end && pos < seq) {\n        if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)\n          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);\n          cp_async16(smem_u32(dk), p.k_cache + off, 16);\n          cp_async16(smem_u32(dv), p.v_cache + off, 16);\n        } else\n        {\n          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));\n          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;\n          cp_async16(smem_u32(dk), p.k_cache + off, 16);\n          cp_async16(smem_u32(dv), p.v_cache + off, 16);\n        }\n      } else if (pos < end) {\n        if (with_new) {\n          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);\n          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);\n          generic = true;\n        }\n      } else {   // past the range: masked out of S; V must be finite (P = 0 times V)\n        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);\n        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);\n      }\n    }\n    if (L == 8 && generic) fence_proxy_async();\n  };\n  auto issue_idx = [&](int y) {\n    if (phalf == 0 && prow < TILE && y < ntiles) {\n      const int pos = start + y * TILE + prow;\n      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));\n    }\n  };\n  // producer mode: wait for stage `st` (tile i) and write the block\'s own new keys / values into it\n  auto ws_patch = [&](int i, int st) {   // the block\'s own new rows (positions >= seq) into tile i\'s stage\n    const int t0 = start + i * TILE;\n    const int lo = max(t0, seq), hi = min(t0 + TILE, end);\n    if (lo < hi) {   // same for every consumer thread\n      for (int e = tid; e < (hi - lo) * 8; e += NT) {\n        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;\n        const int sw = ((c ^ (r & 7)) << 4);\n        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);\n        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);\n      }\n      if (L == 8) fence_proxy_async();\n    }\n    return lo < hi;\n  };\n  auto ws_acquire = [&](int i, int st) {\n    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);\n    const int t0 = start + i * TILE;\n    const int lo = max(t0, seq), hi = min(t0 + TILE, end);\n    if (lo < hi) {   // same for every consumer thread\n      for (int e = tid; e < (hi - lo) * 8; e += NT) {\n        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;\n        const int sw = ((c ^ (r & 7)) << 4);\n        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);\n        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);\n      }\n      if (L == 8) fence_proxy_async();\n    }\n    return lo < hi;\n  };\n\n  // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----\n  int tidL = tid, partL = part;   // (re-read after the page-index copies are issued)\n  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass\n  const bool has_new = end > seq;\n  constexpr int KPASS = (L * 4 + RPP - 1) / RPP; // L*4 rows (k, v of 2 heads), RPP rows per pass\n  constexpr bool QDIR = L == 8 && !WS;\n  constexpr bool IDXF = !WS;\n  {\n  if constexpr (!WS) {\n#pragma unroll\n    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);\n  }\n  cp_commit();\n  if (!WS && (MODE & 8)) {   // (speculative builds only) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index\n    // copies (or out of the speculative retry loop to the kernel top: scout\'s v200 census, segment head 21 -> 62/97)\n    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));\n    partL = tidL & 7;\n  }\n\n  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel\'s output; nothing above reads or writes it\n  // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row\n  tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)\n    const int rr = e >> 3, ch = e & 7;\n    const int j = rr >> 4, hh = rr & 15;\n    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);\n  });\n  if (has_new) {\n    tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)\n      const int rr = e >> 3, ch = e & 7;\n      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;\n      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;\n      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);\n    });\n  }\n  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);\n  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue\n  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);\n  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);\n  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);\n  else if (!FLAT && M_COOP && tidL == 10)   // the coop barrier\'s flag / epoch, early\n    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);\n  else if (tidL >= 16 && tidL < 16 + L) {\n    const int jj = tidL - 16;\n    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));\n  }\n  cp_commit();\n  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last\n    tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)\n      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;\n      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));\n    });\n  }\n\n  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----\n  // (per-request kernels): wait for the page-index group only (committed first, 256 B); the q / new\n  // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out\n  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  if constexpr (!FLAT && (MODE & 8) != 0) {\n    // this CTA\'s ranges and page indices came from the speculated length (see the kernel); the real\n    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been\n    // written: drain the copies and let the caller redo the segment with the real length (today\'s path)\n    if (seq_real_ != seq) {\n      cp_wait<0>();\n      __syncthreads();\n      return false;\n    }\n  }\n  if constexpr (!WS) {\n    constexpr int PRE = STAGES - 1;   // two streams: one cp.async group per tile, the loop issues pairs\n    bool one = false;\n    // a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one\n    // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the\n    // pipeline\'s group count expects (scout\'s census: ~168 instr, 11 taken branches here on s1, cold)\n    if (ONE && p.off32 && p.lp == 0) {\n      one = true;\n      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);\n      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);\n      const int cend = min(end, seq);\n#pragma unroll\n      for (int k = 0; k < TILE / RPP; ++k) {\n        const int r = crow0 + RPP * k;\n        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);\n        \n          // branch-free (scout\'s v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):\n          // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block\'s new\n          // rows are skipped (written by the patch pass after the new-k/v pass)\n          const int pr = start + r;\n          const bool cached = pr < cend;\n          const bool doit = cached || pr >= end;\n          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;   // (zero-filled index past the cached rows)\n          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);\n          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);\n            // (the block\'s new rows: written by the patch pass after the new-k/v pass, as with the general issue)\n      }\n#pragma unroll\n      for (int k = 0; k < STAGES - 1; ++k) cp_commit();\n    }\n    if (!one) {\n#pragma unroll\n    for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)\n      if (k < PRE && k < ntiles) issue_kv(k, false);\n      issue_idx(k + STAGES - 1);\n      if (k < PRE - 1 || k == STAGES - 2) cp_commit();\n    }\n    }\n  }\n  }\n\n  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n  }\n  // ---- 3. while they load: RoPE table, queries, the block\'s new keys / values ----\n  if (p.rope) {\n    tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)\n      const int j = e >> 5, i = e & 31;\n      const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it\n      float sv, cv;\n      // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for\n      // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against\n      // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf\n      const float kq = rintf(ang * 0.15915493667125702f);\n      float rr = fmaf(-kq, 6.25f, ang);\n      rr = fmaf(-kq, 0.03318530693650246f, rr);\n      rr = fmaf(-kq, 2.430837753308879e-10f, rr);\n      __sincosf(rr, &sv, &cv);\n      S.cs[e] = cv;\n      S.sn[e] = sv;\n    });\n  }\n  csync<WS, NT>();\n\n  const float kscale = S.scl[0], vscale = S.scl[1];\n  const float sl2 = p.scale_log2 * kscale;\n\n  if constexpr (QDIR) {\n    // (q_len 8, QDIRECT): four threads per query row, 16 dims each (half the passes of the 8-thread\n    // layout, two shuffle levels instead of three, one 16-byte e4m3 store); same math per element\n    constexpr int RP4 = NT / 4;                    // rows per pass\n    constexpr int QP4 = L * 16 / RP4;\n    const int q4 = tidL & 3;\n#pragma unroll\n    for (int ps = 0; ps < QP4; ++ps) {\n      const int rr = ps * RP4 + (tidL >> 2);\n      const int j = rr >> 4, hh = rr & 15;\n      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);\n      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);\n      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);\n      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);\n      float x[16];\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const float2 fa = __bfloat1622float2(ha[e]);\n        const float2 fb = __bfloat1622float2(hb[e]);\n        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;\n        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;\n      }\n      float ss = 0.f;\n#pragma unroll\n      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];\n      ss += __shfl_xor_sync(0xffffffffu, ss, 1);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 2);\n      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n#pragma unroll\n      for (int e = 0; e < 16; ++e) x[e] = bf16_round(x[e] * r);\n      if (p.rope) {\n#pragma unroll\n        for (int e = 0; e < 8; ++e) {\n          const int i = q4 * 8 + e;\n          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];\n          const float a0 = x[2 * e], a1 = x[2 * e + 1];\n          x[2 * e] = bf16_round(a0 * c - a1 * sn);\n          x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);\n        }\n      }\n      const int hk4 = hh >> 3, gg4 = hh & 7;\n      const int r4 = j * 8 + gg4, kb = q4 * 16;\n      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =\n          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),\n                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));\n    }\n  } else\n#pragma unroll\n  for (int ps = 0; ps < QPASS; ++ps) {\n    const int rr = ps * RPP + (tidL >> 3);\n    const int j = rr >> 4, hh = rr & 15;\n    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);\n    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);\n    float x[8];\n#pragma unroll\n    for (int e = 0; e < 4; ++e) {\n      const float2 f = __bfloat1622float2(h2[e]);\n      x[2 * e] = f.x;\n      x[2 * e + 1] = f.y;\n    }\n    float ss = 0.f;\n#pragma unroll\n    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];\n    ss = oct_sum(ss);\n    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n#pragma unroll\n    for (int e = 0; e < 8; ++e) x[e] = bf16_round(x[e] * r);\n    if (p.rope) {\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const int i = partL * 4 + e;\n        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];\n        const float a0 = x[2 * e], a1 = x[2 * e + 1];\n        x[2 * e] = bf16_round(a0 * c - a1 * sn);\n        x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);\n      }\n    }\n    // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage\n    uint4 qw;\n    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);\n    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the\n    // kv scale would push it into e4m3\'s subnormals -- and scale the scores instead)\n    const float qsc = (L == 8) ? 1.f : sl2;\n    const int hk = hh >> 3, gg = hh & 7;\n    if constexpr (QDIR) {\n      // straight to the e4m3 A tile of the SS QK^T wgmma (64-byte rows, 16-byte chunks swizzled by\n      // (row >> 1) & 3), one fp32 -> e4m3 rounding, no f16 staging / re-read (scout\'s census: ~90 instr / warp)\n      const int r = j * 8 + gg, kb = partL * 8;\n      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =\n          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));\n      continue;\n    }\n#pragma unroll\n    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);\n    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)\n    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)\n      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;\n    } else {\n      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;\n      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;\n    }\n  }\n  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop\'s first barrier\n\n  if (has_new) {\n#pragma unroll\n    for (int ps = 0; ps < KPASS; ++ps) {\n      const int rr = ps * RPP + (tidL >> 3);\n      const bool live = rr < L * 4;\n      const int kind = rr / (2 * L);\n      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;\n      float x[8];\n      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);\n      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const float2 f = __bfloat1622float2(h2[e]);\n        x[2 * e] = f.x;\n        x[2 * e + 1] = f.y;\n      }\n      float ss = 0.f;\n#pragma unroll\n      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];\n      ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)\n      float sc;\n      if (kind == 0) {\n        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n#pragma unroll\n        for (int e = 0; e < 8; ++e) x[e] = bf16_round(x[e] * r);\n        if (p.rope && live) {\n#pragma unroll\n          for (int e = 0; e < 4; ++e) {\n            const int i = partL * 4 + e;\n            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];\n            const float a0 = x[2 * e], a1 = x[2 * e + 1];\n            x[2 * e] = bf16_round(a0 * c - a1 * sn);\n            x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);\n          }\n        }\n        sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;\n      } else {                          // exact for the power-of-two scales)\n        sc = __fdividef(1.f, vscale);\n      }\n      if (live) {\n        uint32_t w0 = 0, w1 = 0;\n#pragma unroll\n        for (int e = 0; e < 4; ++e) {\n          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);\n          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);\n        }\n        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;\n        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);\n        const int pos = seq + jj;\n        if (pos >= start && pos < end) {\n          const int64_t row = S.cloc[jj];\n          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));\n          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);\n        }\n      }\n    }\n  }\n  csync<WS, NT>();\n  if (!WS && has_new) {\n    // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)\n    for (int x = 0; x < (STAGES - 1) && x < ntiles; ++x) {\n      const int st = x % STAGES;\n#pragma unroll\n      for (int k = 0; k < TILE / RPP; ++k) {\n        const int r = crow0 + RPP * k;\n        const int pos = start + x * TILE + r;\n        if (pos >= seq && pos < end) {\n          const int sw = ((cchunk ^ (r & 7)) << 4);\n          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);\n          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);\n        }\n      }\n    }\n    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma\n  }\n\n  \n  // ---- consumer state ----\n  constexpr int MT = 1;                    // m-tiles per warp\n  constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread\n  // (lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)\n  const int hk = __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0);      // L==8: warpgroup = KV head\n  const int wsub = __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0);    // L==1: key half; L==8: m-tile (tokens 2*wsub, 2*wsub+1)\n  const int wgi = hk;          // named barrier of this warpgroup\n  float o[MT][8][4];\n#pragma unroll\n  for (int mt = 0; mt < MT; ++mt)\n#pragma unroll\n    for (int t = 0; t < 8; ++t)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;\n  float m_r[NROW], l_r[NROW];\n#pragma unroll\n  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }\n  // tiles with no window / causal / range edge skip the per-element mask\n  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);\n  const int hi_all = __shfl_sync(0xffffffffu, min(end - 1, seq), 0);\n\n  if constexpr (L == 1) {\n  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims\n  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.\n  uint32_t qa[4][2];\n  {\n    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;\n    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);\n    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);\n    // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)\n    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;\n    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;\n  }\n\n  for (int i = 0; i < ntiles; ++i) {\n    if constexpr (WS) {\n      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();\n    } else {\n      cp_wait<STAGES - 2>();\n      __syncthreads();\n      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n      issue_idx(i + 2 * STAGES - 2);\n      cp_commit();\n    }\n\n    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;\n    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;\n    const int t0 = start + i * TILE;\n    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);\n    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile\n    float s[4][4];\n    uint32_t kw[4][4];\n#pragma unroll\n    for (int nt = 0; nt < 4; ++nt) {\n      const int key = 32 * wsub + 8 * nt + g;\n      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));\n      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;\n    }\n#pragma unroll\n    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)\n      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));\n        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));\n      }\n      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;\n    }\n    if (!interior) {\n#pragma unroll\n      for (int nt = 0; nt < 4; ++nt)\n#pragma unroll\n        for (int e = 0; e < 2; ++e) {\n          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;\n          if (pos >= end) s[nt][e] = -INFINITY;\n        }\n    }\n    float mx = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));\n    mx = quad_max(mx);\n    const float m_new = fmaxf(m_r[0], mx);\n    if (m_new != -INFINITY) {\n      const float alpha = ex2(m_r[0] - m_new);\n      m_r[0] = m_new;\n      float ls = 0.f;\n#pragma unroll\n      for (int nt = 0; nt < 4; ++nt)\n#pragma unroll\n        for (int e = 0; e < 2; ++e) {\n          // FA3 Max_offset=8: exact power-of-two scale extends the FP8 tail.\n          const float pv = ex2(s[nt][e] - m_new) * 256.f;\n          s[nt][e] = pv;\n          ls += pv;\n        }\n      if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)\n        l_r[0] = ls;\n      } else {\n        l_r[0] = l_r[0] * alpha + ls;\n#pragma unroll\n        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }\n      }\n      uint32_t b0[8], b1[8];\n      load_v_frags_f8(V, 32 * wsub, hk, g, tig, b0, b1);\n      const uint32_t a0 = e4m3x4f(s[0][0], s[0][1], s[1][0], s[1][1]);\n      const uint32_t a2 = e4m3x4f(s[2][0], s[2][1], s[3][0], s[3][1]);\n#pragma unroll\n      for (int t = 0; t < 8; ++t) mma_f8pv(o[0][t], a0, a2, b0[t], b1[t]);\n    }\n    if constexpr (WS) {   // this warp is done with the stage\n      __syncwarp();\n      mbar_arrive(&S.empty[(tb + i) % STAGES]);\n    }\n    PROF_TILE;\n  }\n  PROF_DUMP;\n  pdl_launch_dependents();\n  } else {\n  // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =\n  // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a\n  // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.\n  uint8_t* V16 = S.kvq;\n  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims\n  if constexpr (!QDIR) {\n    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g\n    const int r0 = 16 * wsub + g, r1 = r0 + 8;\n    // producer mode: in place in S.kvq, row j*16 + hk*8 + g\n    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;\n    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;\n#pragma unroll\n    for (int ks = 0; ks < 2; ++ks) {\n#pragma unroll\n      for (int hf = 0; hf < 2; ++hf) {        // dims 32*ks + 16*hf + 4*tig .. +3\n        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;\n        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);\n        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);\n        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);\n        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);\n      }\n    }\n       // the A tile for the SS QK^T wgmma\n#pragma unroll\n      for (int ks = 0; ks < 2; ++ks)\n#pragma unroll\n        for (int hf = 0; hf < 2; ++hf) {\n          const int kb = 32 * ks + 16 * hf + 4 * tig;\n          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =\n              qa8[ks][2 * hf];\n          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =\n              qa8[ks][2 * hf + 1];\n        }\n      fence_proxy_async();   // read by the wgmma after the loop\'s first barrier\n    \n    \n  }\n  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8\n  const float ssc = sl2;   // raw score -> log2 units\n  // the lazy-max test on raw scores against (offset + 8) / ssc (ssc > 0), updated with the offset\n  float thr[2] = {0.f, 0.f};\n  float lacc[4] = {0.f, 0.f, 0.f, 0.f};\n  bool first = true;\n     // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T\n    float* oz = &o[0][0][0];   // wgmma\'s pipeline stage (warning C7515: every wgmma of the kernel serialized)\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);\n#pragma unroll\n    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n  \n  const int ckey = tid >> 2, cq = tid & 3;   // V conversion: key, two 16-byte chunks\n  // FP8 PV: the per-CTA kernels (the flat kernel\'s register budget serializes its wgmmas: measured slower)\n  // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments\n  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;   // (warp: uniform)\n  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;\n  const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)\n  // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk\n  // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two\n  // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)\n  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;\n  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);\n  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);\n  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);\n  // the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,\n  // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same\n  // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)\n  const int stj = lane & 7, stm = lane >> 3;\n  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);\n  // (per-request kernels: stmatrix measured vs16 -2.5%, vs1/vs32 -1.5%, vf1 -1%; the flat kernel: +0..1.5%)\n  constexpr bool STSM_ON = !FLAT;\n  // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot\n  // hand their registers to anything the next tile writes before that\n  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};\n  // per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and\n  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without\n  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once\n  // per-request (non-producer) q8 kernels with the split PV and the consumers\' own V^T\n  constexpr bool LSYNCN = !WS && !FLAT;\n  constexpr bool LSYNC = WS;\n  int st_c = tb % STAGES;\n  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;\n  // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the\n  // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word\n  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);\n  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);\n  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;\n  const uint32_t vsrc0 = smem_u32(S.v);\n  uint8_t* vtb_c = S.kvq + hk * 4608;\n  const uintptr_t vtb_x = uintptr_t(S.kvq + hk * 4608) ^ uintptr_t(S.kvq + (2 + hk) * 4608);\n  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));\n  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));\n  const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block\'s new rows\n  // interior tiles (no mask) are i in [i_lo, i_lo + i_span], found once (shuffled: kept uniform)\n  uint32_t i_lo = 0, i_span = 0;\n  \n\n  // (producer mode): the common tiles -- no mask, no new rows (every key before the block\'s first new\n  // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the\n  // exact mask test and the new-row patch run only on the rare tiles\n  uint32_t r_lo = 0x7fffffffu, r_span = 0u;\n  if constexpr (WS) {\n    const int hi2 = min(end - 1, seq - 1);\n    const int a = (lo_all - start + TILE - 1) / TILE;\n    const int z = hi2 - (TILE - 1) - start;\n    const int bz = z >= 0 ? z / TILE : -1;\n    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);\n    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);\n  }\n  // (flat producer mode): the two KV heads\' warpgroups take turns in the exponential phase (named barriers\n  // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment\n  // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)\n  \n  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup\'s claimed tile, and how many it has taken\n  \n  const int nit = ntiles;\n  constexpr int UNR8 = WS ? 2 : 1;   // (pragma arguments are not macro-expanded;\n  // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)\n#pragma unroll (UNR8)\n  for (int jj = 0; jj < nit; ++jj) {\n    const int i = jj;   // S2: tiles 2jj and 2jj + 1; F2: claimed\n    \n      if constexpr (WS) {\n      // tile i landed (+ new rows); previous tile\'s PV of this warpgroup done (its V16 half is free). The two\n      // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.\n      bool patched;\n      \n        mbar_wait_full(&S.full[st_c], ph_c);\n        patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);\n        \n       \n      if (patched || i == 0) csync<WS, NT>();\n      // (FP8 PV: V^T is double-buffered and each warp\'s wgmma wait already covers the PV that read this tile\'s\n      // buffer, so the tile needs no warpgroup barrier before its transpose)\n    } else {\n      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)\n      __syncthreads();       // tile i landed; previous tile\'s PV done (V16 free)\n    }\n      \n    \n\n    const int st = st_c;\n    \n    float s[32];\n    wgmma_fence();\n    \n      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));\n      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;\n      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);\n      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);\n      \n       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);\n      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);\n     \n    wgmma_commit();\n    if constexpr (!WS) {   // the copies for tile i + STAGES - 1\n      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n      issue_idx(i + 2 * STAGES - 2);\n      cp_commit();\n    }\n    if constexpr (FLAT) {\n      // the previous tile\'s PV (committed before this QK^T) is done: its P registers are free for the transpose\n      // below (the flat kernel\'s 104-register consumers spilled there) and o / lacc are settled\n      wgmma_wait<1>();\n#pragma unroll\n      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));\n      float* od0 = &o[0][0][0];\n#pragma unroll\n      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n    }\n    uint8_t* V16t = V16 + 0;   // PV reads it asynchronously: double buffer\n    // this head\'s V^T for this tile (PV reads it asynchronously: double buffer per stream)\n    uint8_t* vtb = vtb_c;\n    \n      if (jj == 0) {   // rows 64..71 of this head\'s two V^T buffers: e4m3 ones (0x38), so the PV\n                                     // wgmma also produces the row sums (after the first barrier: the q staging shares this space)\n        const int wt = tid & 127;\n        *reinterpret_cast<uint2*>(S.kvq + ((0 + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =\n            make_uint2(0x38383838u, 0x38383838u);\n      }\n    \n         // meanwhile: this head\'s V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for\n        // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread\n      const uint32_t vsrc = vsrc0 + st * STAGE_BYTES;\n#pragma unroll\n      for (int u = 0; u < 2; ++u) {\n        uint32_t r0, r1, r2, r3;\n        ldsm_x4_t(vsrc + (u ? (vld_off0 ^ 32u) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)\n        // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1\n        if constexpr (STSM_ON) {\n          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),\n                  prmt(r2, r3, vsel1));\n        } else {\n#pragma unroll\n        for (int g2 = 0; g2 < 2; ++g2) {\n          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;\n          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);\n          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0\n          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;   // row 16(2u+g2) + 2vw + 1 - vr0\n        }\n        }\n      }\n      if (!LSYNC && !LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma\n     \n      if (LSYNC || LSYNCN) {   // (/ N: fence + barrier just before the PV)\n    } else wgsync(wgi);   // this head\'s V16 / V^T ready\n                              // V16 ready\n    wgmma_wait<0>();\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(s[e]);\n    \n#pragma unroll\n      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));\n    \n    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed\n      mbar_arrive(&S.empty[st]);\n    }\n    \n\n    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);\n    const int t0 = start + i * TILE;\n    const bool interior = common_t || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));\n    if (!interior) {\n#pragma unroll\n      for (int hf = 0; hf < 2; ++hf) {\n        const int pj = seq + 2 * wsub + hf;\n        const int vlo = p.window ? max(start, pj - p.window + 1) : start;\n        const int vhi = min(end - 1, pj);\n#pragma unroll\n        for (int nt = 0; nt < 8; ++nt)\n#pragma unroll\n          for (int e = 0; e < 2; ++e) {\n            const int pos = t0 + 8 * nt + 2 * tig + e;\n            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;\n          }\n      }\n    }\n    // after a segment\'s first tile the offset test runs on the packed P instead of on 32 row-max steps: an\n    // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte\n    // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)\n    constexpr bool PCHK = WS;\n    const bool chk = PCHK && !first;\n    float mlo = -INFINITY, mhi = -INFINITY;\n    if (!chk) {\n#pragma unroll\n    for (int nt = 0; nt < 8; ++nt) {\n      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));\n      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));\n    }\n    }\n    // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread\'s\n    // partial max; only the (rare) offset move and the first tile reduce over the row\'s quad\n    if (!chk) {\n      mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)\n      mhi *= ssc;\n    }\n    float dlo, dhi;\n    if (first) {\n       mlo = quad_max(mlo); mhi = quad_max(mhi); \n      \n      dlo = (mlo == -INFINITY) ? 0.f : mlo;\n      dhi = (mhi == -INFINITY) ? 0.f : mhi;\n      cr[0] = dlo; cr[1] = dhi;\n      \n      first = false;\n    } else {\n      dlo = cr[0]; dhi = cr[1];\n      if (!PCHK && __any_sync(0xffffffffu, (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {\n        // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state\n         mlo = quad_max(mlo); mhi = quad_max(mhi);    // (warp-uniform branch: full-mask shuffles)\n        \n        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);\n        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);\n        // (the previous tile\'s PV wgmma completed at this tile\'s wgmma wait)\n#pragma unroll\n        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }\n        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)\n        cr[0] = nlo; cr[1] = nhi;\n        dlo = nlo; dhi = nhi;\n        \n      }\n    }\n       // my turn for the exponentials\n    \n       // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of\n        // the row sum) as an asynchronous wgmma; the next tile\'s wgmma wait covers it before P, o or V^T are reused\n      constexpr bool PK2 = PCHK;\n      if constexpr (PK2) {\n        auto pack2 = [&]() {   // both halves of P from the scores (32 independent chains)\n#pragma unroll\n          for (int kk = 0; kk < 2; ++kk) {\n            const float* sv = s + 16 * kk;\n            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));\n            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));\n            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));\n            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));\n          }\n        };\n        pack2();\n        if (chk) {\n          // an e4m3 byte above 0x78 (P > 256: a score past its row\'s offset + 8) sets bit 7 when 0x07 is added\n          constexpr uint32_t PB = 0x07070707u;\n          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |\n                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;\n          if (__any_sync(0xffffffffu, ov != 0u)) {\n            // rare: offsets up to the tile\'s row maxima, state rescaled (no PV of this tile issued yet; the previous\n            // tile\'s finished at its end), both halves repacked (P <= 1 afterwards: one pass)\n            float hlo = -INFINITY, hhi = -INFINITY;\n#pragma unroll\n            for (int nt = 0; nt < 8; ++nt) {\n              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));\n              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));\n            }\n            hlo = quad_max(hlo) * ssc;\n            hhi = quad_max(hhi) * ssc;\n            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);\n            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);\n#pragma unroll\n            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }\n            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;\n            cr[0] = nlo; cr[1] = nhi;\n            dlo = nlo; dhi = nhi;\n            pack2();\n          }\n        }\n        float* odk = &o[0][0][0];\n#pragma unroll\n        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);\n#pragma unroll\n        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n#pragma unroll\n        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));\n        if constexpr (LSYNC) {   // every warp\'s V^T for this tile written (and visible to the async proxy)\n          fence_proxy_async();\n          wgsync(wgi);\n        }\n        wgmma_fence();\n        \n          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);\n          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);\n         \n        wgmma_commit();\n      }\n#pragma unroll\n      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {\n        const float* sv = s + 16 * kk;\n#define EXQ(x, q) ex2(x)\n        bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores\n                           // would otherwise retrigger the check forever: the SASS oracle\'s value-perturbing probe hung)\n#pragma unroll 1\n        for (;;) {   // (repeated once after a rare offset move; otherwise a single pass)\n        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));\n        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));\n        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));\n        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));\n        if constexpr (!PCHK) break;\n        else {\n          if (!rchk) break;\n          {\n            // (test hook, a hit at P > 1 instead of P > 256: exercises the offset move on most tiles)\n            constexpr uint32_t PB = 0x07070707u;\n            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;\n            if (!__any_sync(0xffffffffu, ov != 0u)) break;\n            {\n              // rare: a score of this half passed its row\'s offset + 8: offsets up to the half\'s row maxima,\n              // state rescaled (half 0\'s PV wgmma finished first), this half repacked\n              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));\n              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));\n              hlo = quad_max(hlo) * ssc;\n              hhi = quad_max(hhi) * ssc;\n              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);\n              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);\n              if ((kk == 1)) {   // (a PV wgmma may still accumulate into o)\n                wgmma_wait<0>();\n                float* odw = &o[0][0][0];\n#pragma unroll\n                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);\n#pragma unroll\n                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n              }\n#pragma unroll\n              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }\n              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;\n              cr[0] = nlo; cr[1] = nhi;\n              dlo = nlo; dhi = nhi;\n              rchk = false;\n            }\n          }\n        }\n        }   // (pack / check loop)\n#undef EXQ\n           // each half of P goes to the tensor cores as soon as it is packed\n          float* od = &o[0][0][0];\n          if constexpr (LSYNCN) {   // every warp\'s V^T for this tile written (and visible to the async proxy)\n            if (kk == 0) {\n              fence_proxy_async();\n              wgsync(wgi);\n            }\n          }\n          if (kk == 0) {\n#pragma unroll\n            for (int e = 0; e < 32; ++e) reg_fence(od[e]);\n#pragma unroll\n            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n          }\n#pragma unroll\n          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));\n          wgmma_fence();\n          wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);\n          \n          wgmma_commit();\n        \n      }\n      \n      \n      float* od = &o[0][0][0];\n      \n           // done before the next tile (no wgmma in flight across the loop edge)\n          wgmma_wait<0>();\n#pragma unroll\n          for (int e = 0; e < 32; ++e) reg_fence(od[e]);\n#pragma unroll\n          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n#pragma unroll\n          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));\n        \n          // (!PVSPLIT)\n     \n    PROF_TILE;\n    \n    \n      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }\n      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);\n      vdesc_c ^= vdesc_x;\n    \n  }\n  PROF_DUMP;\n  pdl_launch_dependents();\n  \n  wgmma_wait<0>();\n  \n#pragma unroll\n    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));\n  \n  {\n    float* od = &o[0][0][0];\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(od[e]);\n#pragma unroll\n    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n  }\n  \n  m_r[0] = first ? -INFINITY : cr[0];\n  m_r[1] = first ? -INFINITY : cr[1];\n  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums\n  l_r[1] = 0.25f * lacc[2];\n  \n  }\n\n  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n  bool do_red = true;\n     // stream 1\'s state was merged into stream 0\n  if (L == 1) {\n    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory\n    // (outside cluster mode both halves add their own rescaled state to the workspace instead)\n    cp_wait<0>();\n    csync<WS, NT>();\n    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;\n    if (wsub == 1) {\n      xs[0] = m_r[0];\n      xs[1] = l_r[0];\n#pragma unroll\n      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }\n    }\n    csync<WS, NT>();\n    if (wsub == 0) {\n      const float m1 = xs[0], l1 = xs[1];\n      const float mm = fmaxf(m_r[0], m1);\n      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);\n      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);\n      m_r[0] = mm;\n      l_r[0] = l_r[0] * f0 + l1 * f1;\n#pragma unroll\n      for (int t = 0; t < 8; ++t) {\n        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;\n        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;\n      }\n    } else {\n      do_red = false;\n    }\n  }\n  if (M_CL) {\n    cp_wait<0>();\n    csync<WS, NT>();\n    float* PB = reinterpret_cast<float*>(S.k);\n    float* PL = PB + L * 16 * 64;\n    // two tile streams (Cfg::S2): stream 0 writes its rows, then stream 1 (same rows) adds its own\n#pragma unroll 1\n    for (int ph = 0; ph < 1; ++ph) {\n    if (ph == 1) csync<WS, NT>();\n    const bool mine = true;\n#pragma unroll\n    for (int r = 0; r < NROW; ++r) {\n      const int mt = r >> 1, hf = r & 1;\n      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;\n      const int row = j * 16 + hk * 8 + g;\n      const float lsum = quad_sum(l_r[r]);\n      if (do_red && mine) {\n        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));\n        if constexpr (L == 1) {   // permuted dims: n-tile t, column c <-> dim 16*tig + 8*c + t\n#pragma unroll\n          for (int c = 0; c < 2; ++c)\n#pragma unroll\n            for (int t4 = 0; t4 < 8; t4 += 4)\n              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =\n                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,\n                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);\n        } else {                  // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c\n#pragma unroll\n          for (int t = 0; t < 8; ++t) {\n            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);\n            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);\n            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);\n          }\n        }\n        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;\n      }\n    }\n    }\n    csync<WS, NT>();\n    cluster_finish<L>(p, S, b, pair, n_keys, chunk);\n    return true;\n  }\n  if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)\n#pragma unroll\n  for (int r = 0; r < NROW; ++r) {\n    const int mt = r >> 1, hf = r & 1;\n    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;\n    const int hh = hk * 8 + g;\n    const float lsum = quad_sum(l_r[r]);\n    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));\n    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;\n    if constexpr (L == 1) {\n#pragma unroll\n      for (int c = 0; c < 2; ++c)\n#pragma unroll\n        for (int t4 = 0; t4 < 8; t4 += 4)\n          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,\n                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);\n    } else {\n      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so\n      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)\n      const bool odd = tig & 1;\n#pragma unroll\n      for (int t = 0; t < 8; t += 2) {\n        // mine: (t, 2tig..2tig+1) and (t+1, same); partner\'s: same t\'s at 2(tig^1)..\n        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;\n        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;\n        // even lane sends its t+1 pair, odd lane sends its t pair\n        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;\n        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);\n        // even lane: t (dims 2tig, 2tig+1, then the partner\'s); odd lane: t+1 (the partner\'s, then mine) -- selects,\n        // one atomic, no branch\n        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);\n      }\n    }\n    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);\n  }\n  }\n  // (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier\n  // releases every thread\'s reds (cumulativity through bar.sync, as CUTLASS\'s semaphore release) and acquires the\n  // others\' for the last CTA\'s reads\n  const bool mode1 = !((FLAT && (L == 8)) && M_COOP) && !(!FLAT && M_COOP);\n  // the same for the cooperative grids (the barrier\'s arrive is an acq_rel atomic after a CTA barrier)\n  if (!(mode1 || !mode1)) gpu_fence();\n  csync<WS, NT>();\n  if ((FLAT && (L == 8)) && M_COOP) {\n    // flat cooperative mode: every request is finalized after all CTAs have merged all their segments\n  } else if (!FLAT && M_COOP) {\n    // ---- all splits of this (request, kv pair) are resident: wait for every split\'s merge, then each\n    // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)\n    // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier\'s chain)\n    const int split_idx = (int)blockIdx.x;\n    int nact = fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys\n    int R = fdiv_i(L * 16 + nact - 1, nact);          // rows per CTA\n    // keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where\n    // they are first used: scout\'s v200 census found them after the poll); volatile asm keeps program order\n    asm volatile("" : "+r"(nact), "+r"(R));\n      if (tid == 0) {\n      // two key counters alternating per launch (parity = this slot\'s epoch, read in the prologue): the last split\n      // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a\n      // departure counter (no atomic round trip at the very end of the kernel)\n      const int ep = S.sense;\n      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;\n      const int mine = end - start;\n      int old;\n      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");\n      \n      if (old + mine == n_keys) {\n        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;\n        p.ws_cnt[24576 + bp] = ep + 1;\n      } else {\n        int v;\n        SPIN_T0;\n        for (;;) {\n          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(c) : "memory");\n          if (v >= n_keys) break;\n          SPIN_CHECK;\n          __nanosleep(64);\n        }\n      }\n    } else if (tid == 0) {\n      atomicAdd(p.ws_cnt + bp, end - start);\n      int v;\n      SPIN_T0;\n      for (int spin = 0;; ++spin) {\n        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");\n        if (v >= n_keys) break;\n        SPIN_CHECK;\n        \n        __nanosleep(64);\n      }\n    }\n    csync<WS, NT>();\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);\n    const int part = tid & 7;\n    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)\n      const int rr = rbase + (tid >> 3);\n      const bool act = rr < r_hi;\n      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;\n      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n                             : make_uint4(0, 0, 0, 0);\n      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n      float lsum = 0.f;\n      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);\n      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;\n      float ss = 0.f;\n#pragma unroll\n      for (int e = 0; e < 8; ++e) { y[e] = attn_out_bf16(y[e] * inv); ss += y[e] * y[e]; }\n      ss += __shfl_xor_sync(0xffffffffu, ss, 1);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 2);\n      ss += __shfl_xor_sync(0xffffffffu, ss, 4);\n      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);\n      uint4 ov;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const float2 gt = __bfloat1622float2(g2[e]);\n        const float s0 = sigm(gt.x), s1 = sigm(gt.y);\n        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);\n        ow[e] = *reinterpret_cast<uint32_t*>(&r2);\n      }\n      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n    }\n    \n  } else {\n  if (tid == 0) {\n    const int mine = end - start;\n    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");\n    \n    S.flag = (old + mine == n_keys);\n  }\n  csync<WS, NT>();\n  if (S.flag) {\n  \n\n  // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----\n  if (p.nc == 1) {\n    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n  } else\n  {\n    const int part = tid & 7;\n    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {\n      const int j = rr >> 4, hh = rr & 15;\n      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n      float lsum = 0.f;\n      // gate row in flight with the copies (one L2 round trip per row instead of two)\n      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);\n      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;\n      float ss = 0.f;\n#pragma unroll\n      for (int e = 0; e < 8; ++e) { y[e] = attn_out_bf16(y[e] * inv); ss += y[e] * y[e]; }\n      ss = oct_sum(ss);\n      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);\n      uint4 ov;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const float2 gt = __bfloat1622float2(g2[e]);\n        const float s0 = sigm(gt.x), s1 = sigm(gt.y);\n        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);\n        ow[e] = *reinterpret_cast<uint32_t*>(&r2);\n      }\n      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n    }\n  }\n  if (tid == 0) p.ws_cnt[bp] = 0;\n  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;\n  }\n  }\n     // consumers\' shared memory is reused by the next segment\n   __syncthreads();                   // shared memory is reused by the next segment\n  }\n  return true;\n}\n\n// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),\n// each CTA finalizes its own slice of all requests\' rows (sum the copies, normalize, rmsnorm * gate, store,\n// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.\ntemplate <int L, bool WS, int NT>\n__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {\n  const int tid = threadIdx.x;\n  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)\n  const int G = gridDim.x * gridDim.y * gridDim.z;\n  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);\n  int* cnt = p.ws_cnt + 30000;\n     // (thread 0\'s acq_rel arrive releases the CTA\'s reds)\n  csync<WS, NT>();\n  if (tid == 0) {\n    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA\n    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel\n    cp_wait<0>();\n    const int ep = *reinterpret_cast<const volatile int*>(sense_s);\n    int* ac = p.ws_cnt + 30020 + (ep & 1);\n    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\\n" : "=r"(old) : "l"(ac) : "memory");\n    \n    if (old + 1 == G) {\n      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;\n      p.ws_cnt[30022] = ep + 1;\n      \n    } else {\n      int v;\n      SPIN_T0;\n      for (;;) {\n        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(ac) : "memory");\n        if (v >= G) break;\n        SPIN_CHECK;\n        __nanosleep(64);\n      }\n    }\n  } else   if (tid == 0) {\n    atomicAdd(cnt, 1);\n    int v;\n    SPIN_T0;\n    for (;;) {\n      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(cnt) : "memory");\n      if (v >= G) break;\n      SPIN_CHECK;\n      __nanosleep(64);\n    }\n  }\n  csync<WS, NT>();\n  const int pairs = p.Hkv >> 1;\n  const int rows = p.B * pairs * L * 16;\n  const int R = fdiv_i(rows + G - 1, G);\n  const int r_lo = c * R, r_hi = min(rows, r_lo + R);\n  const float vscale = __ldg(p.v_scale);\n  const int part = tid & 7;\n  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)\n    const int r = rbase + (tid >> 3);\n    const bool act = r < r_hi;\n    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;\n    const int j = rr >> 4, hh = rr & 15;\n    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;\n    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n                           : make_uint4(0, 0, 0, 0);\n    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};\n    float lsum = 0.f;\n    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);\n    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;\n    float ss = 0.f;\n#pragma unroll\n    for (int e = 0; e < 8; ++e) { y[e] = attn_out_bf16(y[e] * inv); ss += y[e] * y[e]; }\n    ss += __shfl_xor_sync(0xffffffffu, ss, 1);\n    ss += __shfl_xor_sync(0xffffffffu, ss, 2);\n    ss += __shfl_xor_sync(0xffffffffu, ss, 4);\n    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);\n    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);\n    uint4 ov;\n    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);\n#pragma unroll\n    for (int e = 0; e < 4; ++e) {\n      const float2 gt = __bfloat1622float2(g2[e]);\n      const float s0 = sigm(gt.x), s1 = sigm(gt.y);\n      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);\n      ow[e] = *reinterpret_cast<uint32_t*>(&r2);\n    }\n    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;\n  }\n  \n}\n\ntemplate <int L, bool FLAT, int MODE = 0>\n__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel_fp8pv(const __grid_constant__ Params p) {\n  constexpr int NT = Cfg<L, FLAT>::NT;\n  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n  constexpr int IDX_SLOTS = 2 * STAGES;\n  extern __shared__ __align__(1024) uint8_t smem_raw[];\n  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)\n  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));\n  const int pair = blockIdx.z;\n  if (!FLAT) {\n    // grid (n_splits, B): request b, split s of its window / context\n    const int split = blockIdx.x, b = blockIdx.y;\n    // (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang\'s FA3 backend builds\n    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/\n    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row\n    // is exactly seq_len + q_len entries wide. The page table\'s width is a shape: speculate seq = width - q_len and\n    // issue the window\'s page indices in the same trip as cache_seqlens; the real length is checked (segment) before\n    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one\n    const int seq_ld = __ldg(p.seq_lens + b);\n    constexpr bool SPEC = (MODE & 8) != 0;\n    int seq = SPEC ? p.max_pages - L : seq_ld;\n    for (;;) {\n    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;\n    const int n_keys = seq + L - lo;\n    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);\n    chunk = (chunk + 15) & -16;\n    // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:\n    // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).\n    // Not for clusters (an empty cluster CTA still runs the cluster epilogue: measured slower)\n    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {\n      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;\n      // ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI instead\n      // of the ~20-instruction fdiv_i on the once-path (scout\'s census)\n      chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;\n      \n    }\n    int start = lo + split * chunk;\n    int end = min(seq + L, start + chunk);\n    // splits beyond the SM count share an SM with an earlier split, which gets the warp schedulers\'\n    // priority: the first-wave splits take w0/1024 of a second-wave split\'s tiles (only when every split still gets\n    // at least one tile, so all splits have keys and the cooperative finalize counts all of them)\n    \n    if (start >= end) {\n      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }   // (an empty split only on the real length)\n      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {   // still part of its cluster: contributes an empty partial, finalizes its slice\n        pdl_wait();     // (the gate rows are qkv)\n        cluster_stage_gate<L>(p, S, b, pair);\n        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);\n        cp_commit();\n        float* PB = reinterpret_cast<float*>(S.k);\n        float* PL = PB + L * 16 * 64;\n        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;\n        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;\n        cp_wait<0>();\n        __syncthreads();\n        cluster_finish<L>(p, S, b, pair, n_keys, chunk);\n      }\n      return;\n    }\n    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,\n                                  chunk, 0, SPEC ? seq_ld : seq))\n      return;\n    seq = seq_ld;   // (miss: today\'s path with the real length)\n    }\n  }\n  if constexpr (FLAT && (L == 8)) {\n    if (p.coop && threadIdx.x == 0) {   // the grid barrier\'s epoch, long before it is needed\n      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);\n      cp_commit();\n    }\n  }\n  // the per-SM registration\'s atomics go out from warp 1 while warp 0 builds the prefix sums\n  // the same uneven per-SM shares for the non-cooperative (q_len 1) flat grids. No grid barrier ends those,\n  // so the registration state is generation-tagged: a monotonic launch counter picks one of two state copies, and\n  // the launch\'s last registrant clears the other copy (the previous launch\'s) for the next launch\n  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;\n  unsigned w0_smid = 0;\n  int w0_rank = 0;\n  \n  \n  // grid (G): CTA c takes keys [c*T/G, (c+1)*T/G) of all requests\' keys laid end to end (equal work)\n  // warp 0 computes the key range and the segment list right after its prefix sums (one request per lane)\n  constexpr bool fpart = L == 1;   // (compile time: no run-time branch in the main flow)\n  if constexpr (L == 1 && Cfg<L, FLAT>::WS) {\n    if (threadIdx.x == 32) {   // the first segment\'s stage barriers (segment skips its init for that segment)\n      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {\n        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);\n        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);\n      }\n      mbar_fence_init();\n    }\n  }\n  if (threadIdx.x < 32) {\n    int run = 0;\n    int incl_l = 0, n_l = 0, seq_l = 0;\n    for (int b0 = 0; b0 < p.B; b0 += 32) {\n      const int b = b0 + (int)threadIdx.x;\n      int n = 0;\n      if (b < p.B) {\n        const int seq = __ldg(p.seq_lens + b);\n        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;\n        n = seq + L - lo;\n        seq_l = seq;\n      }\n      int incl = n;\n#pragma unroll\n      for (int o = 1; o < 32; o <<= 1) {\n        const int t = __shfl_up_sync(0xffffffffu, incl, o);\n        if ((int)threadIdx.x >= o) incl += t;\n      }\n      if (b < p.B) S.cum[b + 1] = run + incl;\n      run += __shfl_sync(0xffffffffu, incl, 31);\n      incl_l = incl;\n      n_l = n;\n    }\n    if (threadIdx.x == 0) S.cum[0] = 0;\n    if (fpart && p.B > 32) {   // many requests: the generic partition by lane 0 (after the whole prefix sum)\n      __syncwarp();\n      if (threadIdx.x == 0) {\n        constexpr int SEGKF = (L == 8) ? 600 : 400;\n        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);\n        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);\n        if (SEGKF > 0) {\n          auto vmapg = [&](int v) -> int {\n            int lo_b = 0, hi_b = p.B - 1;\n            while (lo_b < hi_b) {\n              const int mid = (lo_b + hi_b + 1) >> 1;\n              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;\n            }\n            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);\n          };\n          csg = vmapg(csg);\n          ceg = vmapg(ceg);\n        }\n        int b = 0, n = 0;\n        while (S.cum[b + 1] <= csg) ++b;\n        for (; b < p.B && S.cum[b] < ceg && n < MAXB; ++b) {\n          S.seg[n][0] = b;\n          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];\n          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];\n          ++n;\n        }\n        S.nseg = n;\n        S.fcs = csg;\n        S.fce = ceg;\n      }\n    } else if (fpart) {   // (one iteration above: lane b holds cum[b + 1] = incl_l, cum[b] = incl_l - n_l; lanes >= B: n = 0)\n      const unsigned FULL = 0xffffffffu;\n      const int lane_f = (int)threadIdx.x;\n      constexpr int SEGKF = (L == 8) ? 600 : 400;\n      const int excl = incl_l - n_l;\n      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);\n      const int cf = blockIdx.x, Gf = gridDim.x;\n      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);\n      if (SEGKF > 0) {   // virtual -> real: the last request whose virtual start (cum[b] + b * SEGK) is <= v\n        const int vstart = excl + lane_f * SEGKF;\n        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);\n        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);\n        const int bs = 31 - __clz(ms), be = 31 - __clz(me);   // (lane 0 always qualifies: virtual start 0)\n        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);\n        csf = min(csf - bs * SEGKF, is);\n        cef = min(cef - be * SEGKF, ie);\n      }\n      // segment list: the requests with cum[b + 1] > cs and cum[b] < ce (a contiguous run of lanes)\n      const bool in = lane_f < p.B && incl_l > csf && excl < cef;\n      const unsigned mi = __ballot_sync(FULL, in);\n      const int rank = __popc(mi & ((1u << lane_f) - 1u));\n      if (in && rank < MAXSEG) {\n        S.seg[rank][0] = lane_f;\n        S.seg[rank][1] = max(csf, excl) - excl;\n        S.seg[rank][2] = min(cef, incl_l) - excl;\n      }\n      if (lane_f == 0) {\n        S.nseg = min(__popc(mi), MAXSEG);\n        S.fcs = csf;\n        S.fce = cef;\n      }\n      if (mi != 0u && csf < cef) {   // the first segment\'s first STAGES tiles of page indices toward L2\n        const int b0 = __ffs(mi) - 1;\n        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);\n        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;\n        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;\n        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);\n        if (cend > s0) {\n          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);\n          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);\n          const long long l0 = e0 >> 5, l1 = e1 >> 5;   // 128-byte lines of int32 page indices\n          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));\n        }\n      }\n    }\n  }\n  __syncthreads();\n  const int64_t T0 = S.cum[p.B];\n  // every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,\n  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each\n  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)\n  constexpr int SEGK = (L == 8) ? 600 : 400;\n  const int64_t T = T0 + (int64_t)(p.B - 1) * SEGK;\n  const int c = blockIdx.x, G = gridDim.x;\n  // (experiment: the first-dispatched half of the grid takes the second half of the keys)\n  const int cpart = c;\n  int cs, ce;\n  if constexpr (fpart) {   // (computed by warp 0)\n    cs = S.fcs;\n    ce = S.fce;\n  } else {\n    cs = (int)((int64_t)cpart * T / G);\n    ce = (int)((int64_t)(cpart + 1) * T / G);\n  }   // (a float-estimate + correction\n  // version without the 64-bit division call measured slower: q_len 1 b16 full +3 us, likely code layout)\n  \n  int vc = c;\n  \n  \n  if (!fpart) {   // virtual -> real key positions\n    auto vmap = [&](int v) -> int {\n      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v\n      while (lo_b < hi_b) {\n        const int mid = (lo_b + hi_b + 1) >> 1;\n        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;\n      }\n      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);\n    };\n    cs = vmap(cs);\n    ce = vmap(ce);\n  }\n  if (cs >= ce) {\n    if constexpr (L == 8) {\n      if (p.coop) {   // still part of the grid-wide finalize\n        if constexpr (Cfg<L, FLAT>::WS) {\n          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);\n        } else {\n          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);\n        }\n      }\n    }\n    return;\n  }\n  \n  \n  // segment list in shared memory (keeps the per-segment loop state out of registers)\n  if (threadIdx.x == 0 && !fpart) {\n    int b = 0, n = 0;\n    while (S.cum[b + 1] <= cs) ++b;\n    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {\n      S.seg[n][0] = b;\n      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];\n      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];\n      ++n;\n    }\n    S.nseg = n;\n  }\n  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)\n  const int nseg = S.nseg;\n  auto run_segments = [&](auto role) {\n    constexpr int R = decltype(role)::value;\n    int tb = 0;\n#pragma unroll 1\n    for (int k = 0; k < nseg; ++k) {\n      const int b = S.seg[k][0];\n      const int seq = __ldg(p.seq_lens + b);\n      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;\n      const int n_keys = seq + L - lo;\n      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;\n      \n      if (s0 < s1) {\n        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,\n                            0, 0,\n                            L == 1 && Cfg<L, FLAT>::WS && k == 0);\n        tb += (s1 - s0 + TILE - 1) / TILE;\n      }\n      \n    }\n       // the producer leaves with its copies landed\n  };\n  if constexpr (Cfg<L, FLAT>::WS) {\n    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers\n    if (threadIdx.x >= Cfg<L, FLAT>::NT) {\n      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::PREGS));\n      run_segments(std::integral_constant<int, 1>());\n    } else {\n      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(Cfg<L, FLAT>::CREGS));\n      run_segments(std::integral_constant<int, 2>());\n      if constexpr (L == 8) {\n        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);\n      }\n    }\n  } else {\n    run_segments(std::integral_constant<int, 0>());\n    if constexpr (FLAT && (L == 8)) {\n      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);\n    }\n  }\n}\n\n}  // namespace\n\ntemplate <int L>\nvoid launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool pdl) {\n  Params p = p0;\n  static bool init = false;\n  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;\n  const int sm = p.flat ? sm_f : sm_n;\n  static int occ_c = 0, occ_f = 0, nsm = 0;\n  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;   // the coop ONE / SPEC / SPEC-ONE instances\n  typedef void (*KernFn)(const Params);\n  // per-request kernels, one build per launch mode (1 plain split-KV, 2 clusters, 3 cooperative)\n  static const KernFn kn[4] = {attn_kernel_fp8pv<L, false, 2>, attn_kernel_fp8pv<L, false, 1>,\n                               attn_kernel_fp8pv<L, false, 2>, attn_kernel_fp8pv<L, false, 3>};\n  static const KernFn kn1[4] = {0, attn_kernel_fp8pv<L, false, 5>, 0, attn_kernel_fp8pv<L, false, 7>};\n  static const KernFn ksp[4] = {0, attn_kernel_fp8pv<L, false, 9>, 0, attn_kernel_fp8pv<L, false, 11>};\n  static const KernFn ksp1[4] = {0, attn_kernel_fp8pv<L, false, 13>, 0, attn_kernel_fp8pv<L, false, 15>};\n  if (!init) {\n    for (int m = 1; m < 4; ++m)\n      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n    for (int m = 1; m < 4; m += 2) {\n      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n    }\n    cudaFuncSetAttribute(attn_kernel_fp8pv<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);\n    cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);\n    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);\n    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);\n    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);\n    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel_fp8pv<L, true>, Cfg<L, true>::NTX, sm_f);\n    int dev;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);\n    init = true;\n  }\n  // the cooperative finalize waits for every split of a request: only when the whole grid is resident\n  Params q = p;\n  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;\n  const bool cl = cs > 1;\n  KernFn kern = p.flat ? (KernFn)attn_kernel_fp8pv<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];\n  if (!p.flat && !cl) {\n    // every split holds at most one tile: whole tiles per split (TCHUNK) and at least as many splits as the longest\n    // request\'s tiles (sliding windows: window + L - 1 keys)\n    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)\n                                    : (long long)p.max_pages * p.P;\n    if ((long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];\n  }\n  // one request, page size 1: the speculative-length builds (; the kernel verifies and falls back)\n  if (!p.flat && !cl && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {\n    const bool one = kern != kn[q.coop ? 3 : 1];\n    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];\n  }\n  if (q.coop && !p.flat && !cl) {\n    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever\n    int occ_k = occ_c;\n    KernFn plain = kn[1];\n    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }\n    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }\n    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }\n    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {\n      q.coop = 0;\n      kern = plain;\n    }\n  }\n  if (cl || pdl) {\n    // cluster dims and / or programmatic stream serialization (PDL: this grid may start while the preceding\n    // kernel finishes; the kernel waits with griddepcontrol.wait before reading qkv)\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = grid;\n    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);\n    cfg.dynamicSmemBytes = sm;   // (one cluster CTA per SM)\n    cfg.stream = stream;\n    cudaLaunchAttribute attr[2];\n    int na = 0;\n    if (cl) {\n      attr[na].id = cudaLaunchAttributeClusterDimension;\n      attr[na].val.clusterDim.x = cs;\n      attr[na].val.clusterDim.y = 1;\n      attr[na].val.clusterDim.z = 1;\n      ++na;\n    }\n    if (pdl) {\n      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n      attr[na].val.programmaticStreamSerializationAllowed = 1;\n      ++na;\n    }\n    cfg.attrs = attr;\n    cfg.numAttrs = na;\n    cudaLaunchKernelEx(&cfg, kern, q);\n  } else {\n    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);\n  }\n}\n\ntypedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,\n                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,\n                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);\n// [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle\n\nvoid run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,\n         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,\n         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,\n         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,\n         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,\n         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,\n         int64_t tma) {\n  Params p;\n  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());\n  p.seq_lens = seq_lens.data_ptr<int>();\n  p.cache_loc = cache_loc.data_ptr<int64_t>();\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());\n  p.page_table = page_table.data_ptr<int>();\n  p.k_scale = k_scale.data_ptr<float>();\n  p.v_scale = v_scale.data_ptr<float>();\n  p.inv_freq = inv_freq.data_ptr<float>();\n  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n  p.ws_acc = ws_acc.data_ptr<float>();\n  p.ws_l = ws_l.data_ptr<float>();\n  p.ws_cnt = ws_cnt.data_ptr<int>();\n  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;\n  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");\n  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");\n  p.lp = 0;\n  while ((1 << p.lp) < P) ++p.lp;\n  p.max_pages = (int)page_table.size(1);\n  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;\n  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;\n  TORCH_CHECK(n_splits <= 1024, "n_splits <= 1024");\n  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;\n  p.cs = (int)cs; p.ncl = (int)ncl;\n  p.ld_qkv = (int)qkv.size(-1);\n  p.row_bytes = (int)(Hkv * 64);\n  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");\n  p.lrb = 0;\n  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;\n  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;   // larger caches: 64-bit offsets\n  TORCH_CHECK(ws_acc.numel() < (1LL << 31), "workspace too large");\n  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");\n  TORCH_CHECK(qkv.numel() < (1LL << 31), "qkv too large");\n  TORCH_CHECK((out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "out / page table too large");\n  p.scale_log2 = (float)scale_log2;\n  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));\n  cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n  p.coop = (int)((tma >> 1) & 1);\n  const bool pdl = ((tma >> 16) & 1);   // programmatic dependent launch (off by default)\n  if (L == 1) launch<1>(p, grid, stream, (int)cs, pdl);\n  else launch<8>(p, grid, stream, (int)cs, pdl);\n}\n'
_native_attn_CPP_SRC = '\nvoid run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,\n         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,\n         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,\n         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,\n         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,\n         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,\n         int64_t tma);\n'
_native_attn__MOD = None
_native_attn__INV_FREQ = {}
_native_attn__WS = {}
_native_attn__NUM_SMS = {}

def _native_attn__cached(cache, key, make):
 value = cache.get(key)
 if value is None:
  value = make()
  cache[key] = value
 return value
_native_attn__TUNED = {(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1), (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)}

def _native_attn__build(tag='', flags=()):
 from torch.utils.cpp_extension import load_inline
 return load_inline(name=f'okc_attn_v298f8pv{tag}_90a', cpp_sources=_native_attn_CPP_SRC, cuda_sources=_native_attn_CUDA_SRC, functions=['run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr', *flags], verbose=False)

def _native_attn__module():
 global _native_attn__MOD
 if _native_attn__MOD is None:
  _native_attn__MOD = _native_attn__build()
 return _native_attn__MOD

def _native_attn__inv_freq(theta, device, n_pairs=_native_attn_HEAD_DIM // 2):

 def make():
  if theta:
   t = 1.0 / theta ** (torch.arange(0, 2 * n_pairs, 2, device=device).float() / (2 * n_pairs))
  else:
   t = torch.zeros(n_pairs, device=device)
  return t.float().contiguous()
 return _native_attn__cached(_native_attn__INV_FREQ, (float(theta), str(device), n_pairs), make)

def _native_attn__workspace(n_rows, device, signature=None):
 key = (str(device), int(n_rows), signature)
 ws = _native_attn__WS.get(key)
 if ws is None or ws[0].numel() < n_rows * _native_attn_HEAD_DIM:
  n = max(n_rows, _native_attn_WORKSPACE_MIN_ROWS)
  ws = (torch.zeros(n * _native_attn_HEAD_DIM, dtype=torch.float32, device=device), torch.zeros(n, dtype=torch.float32, device=device), torch.zeros(n, dtype=torch.int32, device=device))
  _native_attn__WS[key] = ws
 return ws

def _native_attn__num_sms(device):
 return _native_attn__cached(_native_attn__NUM_SMS, str(device), lambda : torch.cuda.get_device_properties(device).multi_processor_count)

def _native_attn__splits(tiles, quota):
 return max(1, min(tiles, quota))

def _native_attn__copies(n, cap):
 return max(1, min(n, cap))

def _native_attn__headroom(target):
 return target * _native_attn_PLAN_HEADROOM_NUM // _native_attn_PLAN_HEADROOM_DEN

def _native_attn__plan(B, pairs, L, window, max_pages, P, device):
 sms = _native_attn__num_sms(device)
 target = 2 * sms
 if not window and 1 < B <= _native_attn_MAX_BATCH:
  return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
 keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
 tiles = max(1, (keys_max + _native_attn_TILE_KEYS - 1) // _native_attn_TILE_KEYS)
 reqs = B * pairs
 if not window:
  if L == 8:
   n = _native_attn__splits(tiles, _native_attn__headroom(target) // reqs)
   return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_native_attn__copies(n, 8))
  n = _native_attn__splits(tiles, target // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_native_attn__copies(n, 4), coop=0)
 if L == 1 and reqs < 24:
  n = _native_attn__splits(tiles, target // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
 if L == 8 and reqs < 16:
  n = _native_attn__splits(tiles, target // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
 if L == 8 and reqs >= 16:
  n = _native_attn__splits(tiles, (target if reqs >= 24 else sms) // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
 if reqs >= 24:
  n = _native_attn__splits(tiles, target // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
 cs = 16
 limit = 2 * sms if L == 1 else sms
 if L == 8 and target == sms:
  limit = sms * 15 // 16
 while cs > 4 and reqs * cs > limit:
  cs //= 2
 ncl = 1
 while cs * ncl > tiles and cs > 4:
  cs //= 2
 if cs * ncl > tiles or cs < 4:
  n = _native_attn__splits(tiles, target // reqs)
  return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_native_attn__copies(n, 4))
 return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=_native_attn__copies(ncl, 4))

def _native_attn__launch(data, dbg=None, _mod=None, pdl=None, **override):
 cfg = data['config']
 (B, L) = (cfg['batch'], cfg['q_len'])
 (hq, hkv, d, P) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'], cfg['page_size'])
 qkv = data['qkv']
 device = qkv.device
 pairs = hkv // 2
 window = cfg['window']
 page_table = data['page_table']
 out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
 plan = _native_attn__plan(B, pairs, L, window, page_table.shape[1], P, device)
 tuned = _native_attn__TUNED.get((B, L, bool(window), pairs))
 if tuned and (not override):
  override = tuned
 if override:
  plan = dict(plan, **override)
  if 'cs' in override and override['cs'] > 1:
   plan['flat'] = 0
   if 'ncl' not in override:
    plan['ncl'] = max(1, plan['n_splits'] // override['cs'])
   plan['n_splits'] = plan['cs'] * plan['ncl']
   if 'nc' not in override:
    plan['nc'] = _native_attn__copies(plan['ncl'], 4)
  elif 'n_splits' in override and 'nc' not in override:
   plan['nc'] = 1 if plan['flat'] else _native_attn__copies(plan['n_splits'], 4)
 (n_splits, flat, cs, ncl, nc) = (plan['n_splits'], plan['flat'], plan['cs'], plan['ncl'], plan['nc'])
 if flat:
  coop = int(plan.get('coop', int(L == 8 and n_splits * pairs <= 2 * _native_attn__num_sms(device))))
 else:
  coop = int(plan.get('coop', 0 if cs > 1 or n_splits < 64 else int(n_splits * B * pairs <= 2 * _native_attn__num_sms(device))))
 use_pdl = bool(pdl)
 tma = int(plan.get('tma', 0)) | coop << _native_attn_COOP_BIT | int(use_pdl) << _native_attn_PDL_BIT
 if flat or cs <= 1:
  (cs, ncl) = (0, 1)
 (ws_acc, ws_l, ws_cnt) = _native_attn__workspace(B * pairs * nc * L * _native_attn_Q_HEADS_PER_PAIR, device, (B, pairs, L, window, P, n_splits, flat, cs, nc, coop))
 scale_log2 = cfg['sm_scale'] * cfg['qk_scale_factor'] * _native_attn_LOG2E
 (_mod or _native_attn__module()).run(qkv, data['seq_lens'], data['cache_loc'], data['k_cache'], data['v_cache'], page_table, data['k_scale'], data['v_scale'], _native_attn__inv_freq(cfg['rope_theta'], device), out, ws_acc, ws_l, ws_cnt, B, L, hq, hkv, P, window, 1 if cfg['rope_theta'] else 0, n_splits, flat, nc, scale_log2, dbg, cs, ncl, tma)
 return out

def _native_attn__supported(cfg):
 P = cfg['page_size']
 return cfg['head_dim'] == _native_attn_HEAD_DIM and cfg['num_q_heads'] == 8 * cfg['num_kv_heads'] and (cfg['num_kv_heads'] % 2 == 0) and (cfg['q_len'] in _native_attn_SUPPORTED_Q_LENS) and (cfg['batch'] <= _native_attn_MAX_BATCH) and (P > 0) and (P & P - 1 == 0) and (cfg['batch'] * (cfg['num_kv_heads'] // 2) <= _native_attn_MAX_REQUEST_PAIRS)

def _native_attn__rms(x, eps=1e-05):
 x = x.float()
 return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def _native_attn__rope(x, pos, theta):
 d = x.shape[-1]
 inv_freq = _native_attn__inv_freq(theta, x.device, d // 2)
 ang = pos.float()[..., None] * inv_freq
 (cos, sin) = (ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2))
 (even, odd) = (x[..., 0::2], x[..., 1::2])
 return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

def _native_attn__fallback(data):
    cfg = data['config']
    (batch, length) = (cfg['batch'], cfg['q_len'])
    (hq, hkv, d, page) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'], cfg['page_size'])
    (q_in, k_in, v_in, gate) = data['qkv'].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data['positions'].view(batch, length)
    q = _native_attn__rms(q_in.reshape(batch, length, hq, d)).to(torch.bfloat16).float()
    k = _native_attn__rms(k_in.reshape(batch, length, hkv, d)).to(torch.bfloat16).float()
    if cfg['rope_theta']:
        (q, k) = (_native_attn__rope(q, pos, cfg['rope_theta']), _native_attn__rope(k, pos, cfg['rope_theta']))
    (q, k) = (q.to(torch.bfloat16), k.to(torch.bfloat16))
    table = data['page_table'].long()
    slots = data['cache_loc'].view(batch, length)
    k_cache = data['k_cache'].view(-1, hkv, d)
    v_cache = data['v_cache'].view(-1, hkv, d)
    (k_scale, v_scale) = (data['k_scale'], data['v_scale'])
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg['window']:
            visible &= t[None, :] > pos[b][:, None] - cfg['window']
        att = torch.einsum('lkgd,skd->lkgs', q[b].view(length, hkv, hq // hkv, d).float(), keys) * (cfg['sm_scale'] * cfg['qk_scale_factor'])
        att = att.masked_fill(~visible[:, None, None, :], float('-inf')).softmax(-1)
        out[b] = torch.einsum('lkgs,skd->lkgd', att, vals).reshape(length, hq, d)
    out = _native_attn__rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)

def _native_attn__fits(data):
    cfg = data['config']
    out_elems = cfg['batch'] * cfg['q_len'] * cfg['num_q_heads'] * cfg['head_dim']
    return (data['qkv'].numel() < _native_attn_INDEX_LIMIT and data['page_table'].numel() < _native_attn_INDEX_LIMIT and (out_elems < _native_attn_INDEX_LIMIT)) and cfg['batch'] * (data['page_table'].shape[1] * cfg['page_size'] + cfg['q_len'] + 600) < _native_attn_INDEX_LIMIT and (cfg['q_len'] != 8 or cfg['batch'] <= 64)

def _native_attn_kernel_fn(data: _native_attn_Inputs) -> torch.Tensor:
 if not _native_attn__supported(data['config']) or not _native_attn__fits(data):
  return _native_attn__fallback(data)
 return _native_attn__launch(data)
import base64 as _zsrc_b64, zlib as _zsrc_zlib

def _zsrc(payload: str) -> str:
 return _zsrc_zlib.decompress(_zsrc_b64.b64decode(payload)).decode()
"Dense projections: small_m (12-bit lossless or BF16 register-direct GEMV) for the shapes it claims, then the\nregister-direct swap-AB GEMV for small m, 12-bit lossless pooler weights at m = 1, and a TMA + wgmma router for\n64 < m <= 256. Shapes no kernel claims fall back to SGLang's own dispatch."
import hashlib
import zlib
from typing import TypedDict
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

class _AchitaDenseWeights(TypedDict):
 qkv_gate: torch.Tensor
 o: torch.Tensor
 pooler: torch.Tensor
 unpooler: torch.Tensor
 router: torch.Tensor

class _AchitaDenseConfig(TypedDict):
 m: int
 n: int
 k: int
 layer: str

class _AchitaDenseInputs(TypedDict):
 x: torch.Tensor
 config: _AchitaDenseConfig
SM_CUDA_TEMPLATE = _zsrc('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')
SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
SM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
SM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
SM_ESC_CAP = 32

def sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def sm_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
  cases.append(f'    case {i}: return sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def sm_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, SM_CPP + extra_cpp, cuda_sources=sm_cuda_source(instances) + extra_cuda, functions=['sm_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def sm_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def _to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def sm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = _to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def sm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(SM_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = _to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def sm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(SM_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def sm_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def sm_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [sm_units(hi), sm_units(lo)]
 else:
  mats = [sm_units(w)]
 if fmt in (0, 2):
  data = torch.cat([sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = sm_c12_bytes(v)
  if not torch.equal(sm_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(sm_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def sm_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def sm_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = sm_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > SM_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = _to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.sm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
SM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
SM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
SM_MAX_M = 32
SM_TABLE_M.update({('router', 8): SM_TABLE_MT['router', 1], ('qkv_gate', 8): SM_TABLE_MT['qkv_gate', 1], ('pooler', 8): SM_TABLE_MT['pooler', 1], ('o', 8): SM_TABLE_MT['o', 1], ('unpooler', 1): SM_TABLE_MT['unpooler', 1], ('unpooler', 8): SM_TABLE_MT['unpooler', 1]})
SM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
SM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def sm_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def sm_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def sm_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def sm_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = sm_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = sm_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = sm_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = sm_verify_onehot(ext, inst_id[inst], sp, sm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def prepare_small_m(weights, table_m=None, table_mt=None, name='small_m_gemv'):
 table_m = dict(SM_TABLE_M if table_m is None else table_m)
 table_mt = dict(SM_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, SM_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if sm_bf16_twin(inst):
   instances.append(sm_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = sm_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and sm_bf16_twin(inst):
   plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, sm_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 return st

def run_small_m(x, layer, n, st):
 m = x.shape[0]
 if not 1 <= m <= SM_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', sm_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != SM_LAYER_K[layer]) or (not x.is_cuda):
  return None
 x = x.contiguous()
 if x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 if sm_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out
COMPRESSED_PAIRS = {('pooler', 1)}
M32_CUDA_TEMPLATE = _zsrc('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')
M32_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
M32_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
M32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
M32_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
M32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
M32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
M32_ESC_CAP = 32

def m32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def m32_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,))))
  cases.append(f'    case {i}: return m32_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return M32_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def m32_build(instances, name='m32_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, M32_CPP + extra_cpp, cuda_sources=m32_cuda_source(instances) + extra_cuda, functions=['m32_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def m32_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def m32_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def m32_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def m32_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = m32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def m32_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(M32_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = m32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def m32_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(M32_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def m32_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def m32_pack_format(w, fmt, layout=0):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [m32_units(hi), m32_units(lo)]
 else:
  mats = [m32_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([m32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = m32_c12_bytes(v)
  if not torch.equal(m32_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(m32_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def m32_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def m32_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = m32_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, M32_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > M32_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = m32_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def m32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.m32_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
M32_TABLE_M = {}
M32_MAX_M = 32
M32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def m32_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def m32_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def m32_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def m32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if m32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def m32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if m32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def m32_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def m32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * key[2]
 rows = m32_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
 pkey = (layer, rows, fmt, layout)
 if pkey not in packs:
  packs[pkey] = m32_pack_format(w[:rows].contiguous(), fmt, layout)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = m32_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
  lo = 1 if total_mt == 1 else 4 * total_mt + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
  streams[vkey] = m32_verify_onehot(ext, inst_id[inst], sp, m32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and m32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def prepare_m32(weights, table_m=None, table_mt=None, name='m32_gemv'):
 table_m = dict(M32_TABLE_M if table_m is None else table_m)
 table_mt = dict(M32_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = m32_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, M32_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if m32_bf16_twin(inst):
   instances.append(m32_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = m32_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and m32_bf16_twin(inst):
   plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, m32_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 st['outputs'] = {layer: torch.empty((M32_MAX_M, w.shape[0]), dtype=w.dtype, device=w.device) for (layer, w) in weights.items() if any((key[0] == layer for key in st['plans']))}
 return st

def run_m32(x, layer, n, st):
 if x.ndim != 2:
  return None
 m = x.shape[0]
 if not 9 <= m <= M32_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', m32_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != M32_LAYER_K[layer]) or (not x.is_cuda):
  return None
 if not x.is_contiguous() or x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=st['outputs'][layer].dtype, device=x.device)
 if m32_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out

def mma(n):
 regs = ','.join((f'%{i}' for i in range(n // 2)))
 outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
 return f'__device__ void mma{n}(float* a,uint64_t x,uint64_t y) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(y));\n    }}'
CUDA = _zsrc('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')
CPP = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nint launch_ss(const void*,const void*,void*,int,int,int,int,int,int,int,int,cudaStream_t);\nint launch(torch::Tensor x,torch::Tensor w,torch::Tensor y,int bn,int split,int stages,int hint,int plain) {\n  return launch_ss(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size(0),y.size(1),x.size(1),bn,split,stages,hint,plain,at::cuda::getCurrentCUDAStream());\n}\n'

def wide2_build():
 source = CUDA.replace('MMA_FUNCTIONS', '\n'.join((mma(n) for n in [64, 128, 256])))
 return load_inline('wide2_final_v1', CPP, cuda_sources=source, functions=['launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])
CANDIDATE_CONFIGS = {(128, 'qkv_gate'): (128, 3, 8, 1), (128, 'o'): (128, 1, 8, 1), (128, 'pooler'): (128, 2, 8, 1), (128, 'unpooler'): (128, 1, 9, 1), (256, 'qkv_gate'): (256, 3, 5, 1), (256, 'o'): (256, 1, 5, 1), (256, 'pooler'): (256, 2, 5, 1), (256, 'unpooler'): (256, 1, 5, 1)}

def prepare_wide2(weights) -> dict:
 device = weights['qkv_gate'].device
 props = torch.cuda.get_device_properties(device)
 if (props.major, props.minor) != (9, 0):
  return {'configs': {}}
 h200_keys = [(128, 'qkv_gate'), (128, 'pooler'), (256, 'pooler')]
 keys = h200_keys
 configs = {key: CANDIDATE_CONFIGS[key] for key in keys}
 return {'ext': wide2_build(), 'weights': {name: weight.contiguous() for (name, weight) in weights.items() if name != 'router'}, 'configs': configs}

def run_wide2(x, layer, n, st):
 m = x.shape[0]
 cfg = st['configs'].get((m, layer))
 if cfg is None or m > 256 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
  return None
 weight = st['weights'].get(layer)
 if weight is None or weight.shape != (n, x.shape[1]) or weight.dtype != torch.bfloat16:
  return None
 out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
 rc = st['ext'].launch(x, weight, out, *cfg, 0)
 if rc:
  raise RuntimeError(f'wide2 launch failed: {rc}')
 return out

def _mma(n):
 regs = ', '.join((f'%{i}' for i in range(n // 2)))
 outs = ', '.join((f'"+f"(d[{i}])' for i in range(n // 2)))
 return f'__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{\n    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));\n}}'
ROUTER_CUDA = _zsrc('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')
ROUTER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def build_router(configs):
 src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join((_mma(n) for n in sorted({c[0] for c in configs}))))
 for (tag, a, b) in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
  src = src.replace('@@CALL_' + tag + '@@', ' '.join((f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs}))))
 cases = '\n'.join((f"case {i}: return rr_launch_t<{','.join(map(str, c))}>(w,x,out,m,stream,query);" for (i, c) in enumerate(configs)))
 src = src.replace('@@CASES@@', cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
 name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
 return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch', 'rr_scalar', 'rr_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def pack_router(w, nr):
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)
 return torch.stack([p.view(256 // nr, nr, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)

def prepare_candidates(weights, configs):
 w = weights['router'].contiguous()
 return {'ext': build_router(configs), 'configs': configs, 'weight': w, 'packed': {nr: pack_router(w, nr) for nr in {64 if c[3] else c[0] for c in configs}}}

def run_candidate(x, st, index):
 out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
 stream = torch.cuda.current_stream().cuda_stream
 if index == -1:
  rc = st['ext'].rr_scalar(st['weight'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
 else:
  cfg = st['configs'][index]
  w = st['packed'][64 if cfg[3] else cfg[0]]
  rc = st['ext'].rr_launch(index, w.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
 if rc:
  raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
 return out
ROUTER_CONFIGS = [(32, 16, 4, 1, 2, 1), (32, 8, 4, 0, 2, 1), (64, 4, 8, 1, 2, 3)]

def prepare_router(weights) -> dict:
 w = weights.get('router')
 if w is None or w.shape != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
  return {'enabled': False}
 if torch.cuda.get_device_capability(w.device) != (9, 0):
  return {'enabled': False}
 st = prepare_candidates(weights, ROUTER_CONFIGS)
 st['small'] = {(16, 8): pack_small_router(w, 16, 8)}
 st['enabled'] = True
 for m in (1, 8, 16, 32, 128, 256):
  x = torch.zeros((m, 8192), dtype=torch.bfloat16, device=w.device)
  run_router(x, 'router', 256, st)
 torch.cuda.synchronize(w.device)
 return st

def run_router(x, layer, n, st) -> torch.Tensor | None:
 if not st.get('enabled') or layer != 'router' or n != 256:
  return None
 if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
  return None
 if x.device != st['weight'].device or x.data_ptr() % 16:
  return None
 m = x.shape[0]
 if not 1 <= m <= 256:
  return None
 if m == 1:
  return run_candidate(x, st, -1)
 if m <= 16:
  return run_small_candidate(x, st)
 return run_candidate(x, st, 0 if m <= 32 else 1 if m <= 128 else 2)
ROUTER_SCALAR = _zsrc('eNqdVm1v4zYM/p5focOAwo59PdtNi2tt+bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKL0matrtbPiQSRfIhH0pkJooXm5wpHqxLRe5sgj/3i+53vggncfyQVyuWxzHZVuuUCBHLhOVMxN+5KHluJFUpFcnyiqkpiWPBpRLrRIF+Y5PuMI7LbbzSKu7VC6XWnpD/+Lzme9dFWJjkCew1DG83Qgtv6Wf32vNBjju1Tqn6JjhL/0zb89YmomroKq+S7/0+ZyWnoHV24dqkYWKDmzC8RAcamrAkWd5Fy1lEn55H6Yw0zXK+iFBQA5BH2hb3oIkiVquK5CldnkUGhiHzStm4WHchEzRHYUQFBzkXG8FVnDCpggNKZ9PQaCwIeXprLtfaMyG/bAR7KBipS1HluRZlldAwijq+Cu58ZVkDDtCQGSoAqiBAjbh8G1RnApitpQ4Rn+Fbp34CPQBLAJbBfOFLRM5TQ9pAoyWn9wvz1JC4B6YrJnUBfFwEt59memHR+QKN+yzeyfkYeki6L9KWIs99GkOlWF+kNxy/QSdD3pBDJNDfc6ujNaQ1X5gQMCbQE3B8YP484lC4p/GNHD0ljzw6dHpSyWMdKOYZRn7ejgGcunF/3M3uwI0+9Ejm0Hiv6XVS49Ex7cx99cQ98IGPC66kE9GsYJmxhSeZOfA1yk913VF3h7o7e5Sf6nqj7iNEs/frvaJ7Meo2qDv6vdjrPu+fwwRplN+Y4Ck0pK5TyLqQ2CruF58uvOjdB/NOoTtfW2qMHFj7DC1jzMB6Gd87jySl7pWf+tDYqGuSrUUx9iyP06opY7krE8Nps/5jb+209wiB6f5IHbNLDuCwR0Z06+9JAPOuxUpD24GVbqRo9fRTLevljR+owCACzeqXIQwURTfOefZD7+p/5n/MwG6ppt7lFfbjnoDhLjxPJogyTkfEvJrFCobgsGrHVVV3o6AYJTDXONNb8k/Nxa4jQM8QHLP05dQNXPvS9Wwv7NnWRpR6A3FJnbKvdZn8qmBgrmrFJcnYWNHh9A+u9grGWcZs9G5+oKgwr5OES2kSaAi1KMlH1+kc9PuMnZd18Td/kFPXcRwLMzalSm9uinUZyPUjj1VogBaMWpbPYf/bDmDs6+trc7g6EEvxAesxYOihi0EEQQBE6yQdWwc81xTFiIGLMAyNo/8eZmMbr/3dMFvb6DWQ90KD93joF0j4C9rd70LATTHpYe5fnJuPn32o7b8VE84w')
ROUTER_SMALL = _zsrc('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')

def pack_small_router(w, s=16, wk=8):
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)

 def units(v):
  v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
  return v.permute(0, 1, 3, 2, 4)
 data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
 ku = 128 // s // wk
 return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def run_small_candidate(x, st, s=16, wk=8):
 out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
 rc = st['ext'].rr_small(st['small'][s, wk].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], s, wk, torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f'router small launch failed: {rc}')
 return out
V9_SELECTED_ROUTES = []

def qkv_mma(n):
 regs = ','.join((f'%{i}' for i in range(n // 2)))
 outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
 return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'
QKV_CUDA = _zsrc('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')

def build_qkv(configs, profile=False):
 ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
 source = QKV_CUDA.replace('@@MMA@@', '\n'.join((qkv_mma(n) for n in ns)))
 source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
 source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
 source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return q_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for (i, c) in enumerate(configs))))
 cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint q_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
 return load_inline('qkv_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['q_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def qkv_pack(w, groups, kt=1):
 rows = (2304 + groups - 1) // groups
 if kt == 2:
  return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
 if 2304 % groups == 0:
  return w.contiguous()
 out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
 for g in range(groups):
  (lo, hi) = (g * 2304 // groups, (g + 1) * 2304 // groups)
  out[g, :hi - lo].copy_(w[lo:hi])
 return out.reshape(-1, 8192)

def prepare_qkv_candidates(weights, configs, profile=False):
 w = weights['qkv_gate']
 return {'ext': build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): qkv_pack(w, g, kt) for (g, kt) in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

def run_qkv_candidate(x, st, index, stats=None, query=0):
 out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
 cfg = st['configs'][index]
 rc = st['ext'].q_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
 if query:
  return rc
 if rc:
  raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
 return out
QKV_SM_CUDA_TEMPLATE = _zsrc('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')
QKV_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
QKV_SM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
QKV_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
QKV_SM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
QKV_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
QKV_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
QKV_SM_ESC_CAP = 32

def qkv_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def qkv_sm_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,))))
  cases.append(f'    case {i}: return qkv_sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return QKV_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def qkv_sm_build(instances, name='qkv_sm_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, QKV_SM_CPP + extra_cpp, cuda_sources=qkv_sm_cuda_source(instances) + extra_cuda, functions=['qkv_sm_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def qkv_sm_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def qkv_sm_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def qkv_sm_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def qkv_sm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = qkv_sm_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def qkv_sm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = qkv_sm_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def qkv_sm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def qkv_sm_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def qkv_sm_pack_format(w, fmt, layout=0):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [qkv_sm_units(hi), qkv_sm_units(lo)]
 else:
  mats = [qkv_sm_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([qkv_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = qkv_sm_c12_bytes(v)
  if not torch.equal(qkv_sm_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(qkv_sm_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def qkv_sm_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def qkv_sm_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = qkv_sm_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, QKV_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > QKV_SM_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = qkv_sm_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.qkv_sm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
QKV_SM_TABLE_M = {}
QKV_SM_MAX_M = 32
QKV_SM_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def qkv_sm_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def qkv_sm_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def qkv_sm_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def qkv_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if qkv_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def qkv_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def qkv_sm_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def qkv_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * key[2]
 rows = qkv_sm_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
 pkey = (layer, rows, fmt, layout)
 if pkey not in packs:
  packs[pkey] = qkv_sm_pack_format(w[:rows].contiguous(), fmt, layout)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = qkv_sm_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
  lo = 1 if total_mt == 1 else 4 * total_mt + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
  streams[vkey] = qkv_sm_verify_onehot(ext, inst_id[inst], sp, qkv_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and qkv_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def prepare_qkv_sm(weights, table_m=None, table_mt=None, name='qkv_sm_gemv'):
 table_m = dict(QKV_SM_TABLE_M if table_m is None else table_m)
 table_mt = dict(QKV_SM_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = qkv_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, QKV_SM_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if qkv_sm_bf16_twin(inst):
   instances.append(qkv_sm_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = qkv_sm_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and qkv_sm_bf16_twin(inst):
   plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, qkv_sm_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 return st

def run_qkv_sm(x, layer, n, st):
 if x.ndim != 2:
  return None
 m = x.shape[0]
 if not 9 <= m <= QKV_SM_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', qkv_sm_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != QKV_SM_LAYER_K[layer]) or (not x.is_cuda):
  return None
 if not x.is_contiguous() or x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
 if qkv_sm_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out
QKV_LARGE_CONFIGS = [(64, 36, 128, 3, 8, 27, 2), (64, 36, 128, 3, 8, 25, 2), (64, 36, 128, 3, 4, 59, 2)]
QKV_SMALL_INST = (1, 2, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 8, 8, 2, 7, 1)
QKV_SMALL_GEOMETRY = (1, 8, 8, 2)

def prepare_qkv(weights) -> dict:
 w = weights.get('qkv_gate')
 if w is None or w.shape != (2304, 8192) or w.dtype != torch.bfloat16 or (not w.is_cuda):
  return {'is_enabled': False}
 if torch.cuda.get_device_capability(w.device) != (9, 0):
  return {'is_enabled': False}
 small = prepare_qkv_sm({'qkv_gate': w}, table_m={}, table_mt={('qkv_gate', 2): (QKV_SMALL_INST, QKV_SMALL_GEOMETRY)}, name='qkv_small_final')
 large = prepare_qkv_candidates(weights, QKV_LARGE_CONFIGS)
 warm = torch.zeros((128, 8192), dtype=torch.bfloat16, device=w.device)
 for index in range(len(QKV_LARGE_CONFIGS)):
  run_qkv_candidate(warm, large, index)
 torch.cuda.synchronize(w.device)
 return {'is_enabled': 'H100' in torch.cuda.get_device_name(w.device), 'device': w.device, 'small': small, 'large': large, 'large_index': 0}

def run_qkv(x, layer, n, st) -> torch.Tensor | None:
 if not st.get('is_enabled') or layer != 'qkv_gate' or n != 2304:
  return None
 if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or (not x.is_cuda):
  return None
 if x.device != st['device'] or not x.is_contiguous() or x.data_ptr() % 16:
  return None
 m = x.shape[0]
 if 9 <= m <= 16:
  return run_qkv_sm(x, layer, n, st['small'])
 if m == 128:
  return run_qkv_candidate(x, st['large'], st['large_index'])
 return None

def rt2_mma(n):
 regs = ', '.join((f'%{i}' for i in range(n // 2)))
 outs = ', '.join((f'"+f"(d[{i}])' for i in range(n // 2)))
 return f'__device__ __forceinline__ void rt2_mma_{n}(float* d, uint64_t a, uint64_t b) {{\n    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));\n}}'
rt2_ROUTER_CUDA = _zsrc('eNrdWltT4zgWfs+vcDNFynbUIQnpVE+MaUhDz3YtBBZnmt6lKJdsK2Bw7CArNxj2t++R5GtimL7MvCxdSWT56Og7Vx1JXfvFD91g5hFlz515uHm7Xyv32HQWMn9CKt4443av3B0zzw9ZuQ8HNxH12e0EOj0y9kOiUNaxL44U2/bI3HeJbUNzHFGX+GEA7227llDMgNtux2ZiRDwhE9WNwpgp88j3dGWqKU8KJWxGQ0VNSTXbducM2zckJNR3bRbZ8S2mxFOnmqE8p5w5B8HVD32WDVYcpEBTCTlnHE+ALMDMD4i6NXEwpT6hTT6gKVn2+y7DTafXVa62W9dI2W4bW0q/v0W3VEdD/CfU+lsAO6KrrcrZyXJK3O+bn//MSVOOtNlyA4uNfhjOAlcoY3oLaGrKGp4npUnJjdKcwtzK1FD40L6SoWR0Jbg1pxiMv2pi92HmU9IEp4gZvF8HPS2AVg7eTBWHYsETYK4JAXgKUtQ2pYhXoatuajCFlkJIFElJQHBMmjjwb0LiGco6nRAjxZ9ScUyVuiy57ARPc3ViqU7KkeU6NjhKMC+e4lwpycyz3Y6y3eI6gU/H2OpvmUINQhtYaoNqmpHEgFNl0uksvs0xeEgZBxFmXWWOCiCqDByD0FyTG6hSG0NPNJkGhBHwwn7fWTESN+fd5hhQS1s+AfDtDtreRdvdZ27gd9eZR3oAf7ylzpvLtLFKG49pYyFFdKrtnQgihQyCgpBSnFRQY02wwNsQKUH9BMpOIQu83Wuh9AynmQE1M6RmAjWRqgA1Mcu8ykex56kSYB0cI4HqCKdtLhum01wauLnijRU0HnnjERoL3lisuVuvm7ibR2K34G/r6ZHTaaqK663l7hj+tP39rvaHmr/rdff2djulrvbeXq9TmSvYBBdUHjOkFHMzm0hvv0fC51Hua5hWeZs7TbzNmQX3TUbCOKLNzqalboLIwcGrPpg43xWPmidpy+vUmKnzxQwsF2ypbCJd7D4NJulwgLHscgcHp6eHBwe1GiMwHWZkj4s1GArpLPk9kj+X4ufi+Ej8nn8+P96v2bbELVa6AM9C99Z2olnoxbatqpym3tE+dN71+u3Oey3X8T2hIQmSRQ+YUN+zxQMOGfD6+LtU1CmeKmyBQKnf9fdNbJdJxtCVaMakTSfSgGIELENUdA4vTOvyQ6/bB6Uoo1P+MBj2e11jg/JyYA4vdBAUKV8H5uhUNq2BeTnQO42vg80Rw09mr6tzxsMv5vDTThcpZ5dDc/hlx+LEZAmeEYIg0lmEkkWeBuW2W52upszCWKRtxQUKJZ5cXWeTiAmY75nslhLsffaWTRA5wCExobe+C060wHTKH/b33yGF4vDedILIvRek21aZE40Whbc7lj68QAqLCkNW+ug0H1OIi5iYWZkTg1/yUIlN3t+wRro1QAr4HluZvJv3gNKoY6qis5F7EX/Tb2mN9jut/t/2u1nFXJS4c5M6jXavDP7eMbl4ugoW2bE0/hLPWKSA/T3zqn6tciJfWr84Lmamv22NjEJ3LpfEGzOO15PiwIM1kNSFIshB1kDLe3l+8VAd/LqFVBX0ujO80LivNO6dhq+BYvUOctYHNDr65QBGLZEqyXpdBOpPCJ/5lz9Wub3NXGXAFDTG5fplSvHNBIPD0CgIBGsoTKXcZsvw90BKv9EQmTUtHoV8PojXhjzpj3PfTfnnpNJWnLYrcirnX86DYxK6BIqqaLlaX34ZXq86xPjNGbuphUqvICPtm+30lZKDog5H/hMwCjYEXqr1tq3pEJ96u5dArBYyzeECRVaGJYm+UtTn5PsbrKTU64q/J/xYGoz7sOpX6exNqrX/d7XJjy3qY5ntYlWTEVEh8LcWz3hJvJfK4ppSHRCpnnkgwmR8zUv1m0ZnqbNoXx6BRcM+FZa9bA+TJEWelTSINlX1d6AFGq63M+Xyv5JX5LrKf39ANd+yX3iB+xszyRZyC5OSySLOKJlRLMoKdt2rwXCnA+vZRlCkKruDkLjb41TGHVfY2r4ymjedZJ/RSqtdzvbuWpMG5FtASubAZW0OpV1bD7wNu7ywSGSGyhcHaaLMPmvLyKKwciBlaS5EojdeSwWvyw2SNnJJN2xTJl7cTCa4KeNPhPVLZn0FzgPAedjrGg9Fp82lFEW8tzSzOn7ZeNChGEfebd63SPuCYt/lQHZXepV1yRerg4OPhycn9j8ODoy0fcLbqZuTICY52bBINywQPr+oGyjDJz6DUjKaTV9XUfWatRFVki33kQqmSruSrZDie9i0vseAP+VPr6TBF9SR5zVIif5+i69oPA6h9JNhxjtEnWq2tD85LHrhiCjbDyXpkrOXGVPbzFXPiZOUYP19YHhOqEJRK2B5VW+vaE0sMK2kJkjepmtlWcoNGUtD/cI48S3ypG8kKfrnPDGpvQmEVVZ7z9OTgsfN/MrM+bbYSd2Z8x3RCEKTyQ0MNRlsXoy1EZiaCwrlRkMNwv19SCuOa97p7/ljfVfTO0V6udzM51fda/PpEbZIj80VfB7hs3h+NWxcCBsX8p67mfcECBPThurudPg6zXcLLjwVVumCgCBFdN+AjPbB6WPYHy1cE7YFogP3nUL6Uxndg00qbFivGNVhW91YuNcmgHevC4voRnVimbLQ+5Yc8F7mAG4d9U7sSsAz0ATfE1uaSGSCrn53jZJGo503O3lzN1lmq1waKgLhbFJrkq+uTDFlZnJ4pGvxxPgewHzwVQb42vwxxPkiXixU0jkZ3ywbbA+KWYM1ypVcbs653GsCUYOlpku8O55NSsBaSPyrTs/5vBRkpXuWQbmg6SEb8ELZEWF6JitqiTm4PqKaljmOMOYcwYhSUJfFLCWIv6AQ/A7bVekw9z1+BgeRD/qEyI9cHEA+4BYoDUrPROHtz1h+ozwDTVOTHys01MyqAgTs2HuFwOQYhd01Rc9cWI0nDZVzeMvtomkmwDOKhUl2jp3ajxMjcYgHpCjthV0U7ysZr3Ld3TCbyMUKZEpQ5sbRvoBQsWsqVLEwc+s7guBvcvOyRrk1hELBJDLKuC3WvD0PwQ23/4a9yXPtuVZjqynxyFj5+Dsl8SxgiqpzquPQjTyiqYXDRB0VjxaPMMMjGIsUd5b6EZKnx+lRsnzBK+PKrlqpj49fI0u6CpN+DiEwYcc8J6Vua+E/PgblvpMO7OMnEfOjsNT/iev47Gww9oMAtBAzcCNXnv1JP7Rvo5iV5Oan4WmJPxXHwn4YEipasEglLSdaIrnZgmmlsyTMc30qsAUxQ0hlU0aTs6w30MVLfH4le8QLLHocMro6j4DXv2aEri6kWR744ZTKqX4jbJ1wsPpCaAyiqlvubJQCl3OOIEq8LaQK0+haHeZD7U6r1UKc2XGInYAckTGGSVD9QXtj8m5r5rokjv/4Q8JLriHetlstuZfI7ah4/iS+6kBZkd86SPXkz0JJzyhm1PfIVXuTVu+I5Tw3uxI/Cp69LsqvhkHDz4jIydqoLYakFyRAowFUFUz18Xd7dDy0zi7s08Nz++hwdGiP/n1+bA8+nZwdjmDR6CTK0KaIg09wofgRmK+N/jwcHV+cHB9+ObaHZ8PjijP6Mr11+fk//zk5tiFfDJAqXME0e90PZaqTjn1+cXZ6Nvp8NuQPve6gn9DCwD8h5qxT6tZrtBxw/3VeUFsNtDWRhZZsiBD702e+fwQu4g7lh+5OROPy3ISJRPOraO7X0nhLLlOYmnrTIou0ZdbiNxfi4iLrAYsRLB6VBx4jP3ejIa6dTGukqx19eNEYncpzan4Srxeud/ptrdGF8loeI30QVxp6t9/Ki/1xaObXPrDyIwuBakAtXCXiKsnIs4ITRQEHMvZvZrChMscYFqo0KeT92U6BByulEQXxiYjRT6A6i7BDBu7rQISp4xCl/VnnKV4erUI88V1L7NtOxbpo+Y8ECbGz/a1KSpGfx7wMrWSH/u0TD6PwPKKMZ5ePsqDikx4GQbQgHmq/OG/VtAUtMToj6QYNBgvrm7C0PW0AiJUxNtLe3wpwY7U+xmgcZje8Y9wMZ5MLchPrkOBaDXgWxQ8HPOBKMmTK8/CJcNePAg7YwR3fmE8yc41vmvwe7sifmJBTdlXw8p3hhW4hdQIO9bat7YxOudScUNwlZZQb94eCxkuMBnWMgGAKa4lX0vlNsRZYog25UXYaJZyZxAqGFuRdQ/y2rpuwC64gS+wEyGAFgcUkI4cqLa2KOewnC6XpF0xg7bdB+4CLU8cm/xYwQaWHoqdtlI0F5OI/yfCN5YZTn7nubIpDdwWOe+gyWOYSVGC1EIyG6sA7sxwxi77zIeynbsNnXLteZcv0OFRe1GVLPb+jWiBYacAAPPD1DhKXU5fnmsH9M3VJ4XcbY5doid63f+2gCQILf60aU5KS0oLq/ykyxfFS5VJx6QROJAtBna+bSFZ1KTsYXZK41ZcCU8pTdDmrynNdD/0FqTVe+Iwz9LQnfqxoHVsHB4oni4Z+GrG//mo8i7t26+PhyeHFwUHtfwLQ/9c=')
rt2_ROUTER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rt2_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint rt2_scalar(int64_t,int64_t,int64_t,int,int64_t,int);\nint rt2_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def rt2_build_router(configs):
 src = rt2_ROUTER_CUDA.replace('@@MMA@@', '\n'.join((rt2_mma(n) for n in sorted({c[0] for c in configs}))))
 for (tag, a, b) in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
  src = src.replace('@@CALL_' + tag + '@@', ' '.join((f'if constexpr(BN=={n}) rt2_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs}))))
 cases = '\n'.join((f"case {i}: return rt2_launch_t<{','.join(map(str, c))}>(w,x,out,m,stream,query);" for (i, c) in enumerate(configs)))
 src = src.replace('@@CASES@@', cases).replace('@@SCALAR@@', rt2_ROUTER_SCALAR + rt2_ROUTER_SMALL)
 name = 'router2_' + hashlib.sha256(src.encode()).hexdigest()[:12]
 return load_inline(name, rt2_ROUTER_CPP, cuda_sources=src, functions=['rt2_launch', 'rt2_scalar', 'rt2_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def rt2_pack_router(w, nr):
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)
 return torch.stack([p.view(256 // nr, nr, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)

def rt2_prepare_candidates(weights, configs):
 w = weights['router'].contiguous()
 return {'ext': rt2_build_router(configs), 'configs': configs, 'weight': w, 'packed': {nr: rt2_pack_router(w, nr) for nr in {64 if c[3] else c[0] for c in configs}}}

def rt2_run_candidate(x, st, index):
 out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
 stream = torch.cuda.current_stream().cuda_stream
 if index == -1:
  rc = st['ext'].rt2_scalar(st['weight'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
 else:
  cfg = st['configs'][index]
  w = st['packed'][64 if cfg[3] else cfg[0]]
  rc = st['ext'].rt2_launch(index, w.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], stream, 0)
 if rc:
  raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
 return out
rt2_ROUTER_CONFIGS = [(32, 16, 4, 1, 2, 1, 256, 256), (32, 8, 4, 0, 2, 1, 256, 256), (64, 8, 6, 1, 2, 1, 64, 64), (64, 4, 8, 1, 2, 3, 256, 256)]

def prepare_router2(weights) -> dict:
 w = weights.get('router')
 if w is None or w.shape != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
  return {'enabled': False}
 if torch.cuda.get_device_capability(w.device) != (9, 0):
  return {'enabled': False}
 st = rt2_prepare_candidates(weights, rt2_ROUTER_CONFIGS)
 st['small'] = {(16, 8): rt2_pack_small_router(w, 16, 8)}
 st['enabled'] = True
 for m in (1, 8, 16, 32, 64, 128, 256):
  x = torch.zeros((m, 8192), dtype=torch.bfloat16, device=w.device)
  run_router2(x, 'router', 256, st)
 torch.cuda.synchronize(w.device)
 return st

def run_router2(x, layer, n, st) -> torch.Tensor | None:
 if not st.get('enabled') or layer != 'router' or n != 256:
  return None
 if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
  return None
 if x.device != st['weight'].device or x.data_ptr() % 16:
  return None
 m = x.shape[0]
 if not 1 <= m <= 256:
  return None
 if m == 1:
  return rt2_run_candidate(x, st, -1)
 if m <= 16:
  return rt2_run_small_candidate(x, st)
 return rt2_run_candidate(x, st, 0 if m <= 32 else 1 if m <= 64 else 2 if m <= 128 else 3)
rt2_ROUTER_SCALAR = _zsrc('eNqdVm1v4zYM/p5focOAwo59PdtNi2tt5bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKb0nTdnfLh0SiSD7kQ4nMRPF8mzHFg02hyJ1N8Od+2f4ulvNJFD1k5ZplUUR25SYhQnmRjFnGRPSdi4JnRlwWUpE0K5makigSXCqxiRUY1DZpD6Oo2EVrreJevVBq7An5j89rvvdtiLlJnsBew/BmK7Twln52rz0f5LhTm4Sqb4Kz5M+kOW9sIsqarrMy/t7tM1ZwClpnF65Naia2uJnPL9GBhiYsjld34WoW0qfnQTojdb1aLEMUVADkkabBPWiiiFWqJFlCV2ehgWHIrFQ2LjZtyATNURhSwUHOxVZwFcVMquCA0tl0btQWhDy9NVcb7ZmQX7aCPeSMVIUos0yL0lJoGEUdXwV3vrKsHgdoSA0VAFUQoEZcvQ2qMwHMxlKHiM/wrVM/ge6BJQDLYLH0JSJniSFtoNGS0/uleWpI3APTNZO6AD4ugttPM72w6GKJxl0W7+R8DN0n3RVpR5HnLo2+Uqwr0huO36CTIW/IIRLoj9zqaA1pLZYmBIwJdAQcH5g/j9gX7ml4I0dPySOPDp2eVPJYB4p5hpGfN0MAp27cH3ezP3CjDz2SOjQaNb1Wajw6pp26r564Bz7wccGVdEKa5iw1dvAkUwe+Bvmprjvo7lF3bw/yU11v0H2EaEa/3iu6F4NujbqD34tR93l8DhOkUX5jgifQkNpOIatcYqu4X3668MJ3H8w7hW597agxcGCNGVrGkIH1Mr53HklC3Ss/8aGxUdckO4ti7GkWJWVdRHJfxIbTpN3H3tlJ5xEC0/2ROmabHMBhjwzpzh9JAPO2xUpD24GVbqRo9fRTLevlje+pwCACzeqXPgwUhTfOefpD7+p/5n/MwH6lpt7lFfbjjoD+LjxPJogyjkcEvZpFCqZgv2qGVVm1syAfJDDYONNb8k/Fxb5lQA8RnLP0ZO4Grn3perY37/jWVpR6PXVxlbCvVRH/qmBkrivFJUnZUNP+9A+uRgXjLGU2ejc/UFRYVHHMpTQJtIRKFOSj67QOun3Kzosq/5s/yKnrOI6FKZtSJTc3+aYI5OaRR2pugBYMW5YtYP/bHmDs6+trs788EEv+ASvSY+ixi0EEQQBU6yQdWwe80BxFiIGL+XxuHP37MGvbeO0Ph9nYRqeBxOcavMNDv0DCX9DwfhcC7opJD3P/4tx8/OxDdf8FLxbOzA==')
rt2_ROUTER_SMALL = _zsrc('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')

def rt2_pack_small_router(w, s=16, wk=8):
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)

 def units(v):
  v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
  return v.permute(0, 1, 3, 2, 4)
 data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
 ku = 128 // s // wk
 return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def rt2_run_small_candidate(x, st, s=16, wk=8):
 out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
 rc = st['ext'].rt2_small(st['small'][s, wk].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], s, wk, torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f'router small launch failed: {rc}')
 return out
OLAYER_CUDA = _zsrc('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')
OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def olayer_source(configs) -> str:
 ns = sorted({c[0] for c in configs})
 functions = []
 for n in ns:
  regs = ','.join((f'%{i}' for i in range(n // 2)))
  outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
  functions.append(f'OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
 cases = '\n'.join((f"case {i}: return ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for (i, c) in enumerate(configs)))
 calls = '\n'.join((f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns))
 return OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def build_olayer(configs):
 for cfg in configs:
  (bn, bk, st, pipe, pack, hint, *groups) = cfg
  tw = groups[0] if groups else 1
  if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
   raise ValueError(f'unsupported olayer configuration: {cfg}')
 source = olayer_source(configs)
 name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
 return load_inline(name, OLAYER_CPP, cuda_sources=source, functions=['ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def pack_olayer(weight) -> torch.Tensor:
 columns = torch.arange(64, device=weight.device)[None, :] ^ (torch.arange(64, device=weight.device)[:, None] & 7) * 8
 tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
 return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()
OLAYER_PRIMARY_CONFIGS = [(16, 256, 3, 3, 1, 1), (32, 128, 8, 1, 1, 1), (64, 128, 4, 3, 1, 1), (128, 128, 4, 3, 1, 1), (128, 64, 5, 3, 1, 1, 2)]
OLAYER_NEARBY_CONFIGS = {bn: (bn, 128, 4, 3, 1, 1) if bn <= 128 else (bn, 64, min(8, 227000 // ((64 + bn) * 128 + 16)), 3, 1, 1) for bn in range(72, 249, 8)}
OLAYER_CONFIGS = list(dict.fromkeys(OLAYER_PRIMARY_CONFIGS + list(OLAYER_NEARBY_CONFIGS.values())))

def prepare_olayer_candidates(weights, configs) -> dict:
 weight = weights['o'].contiguous()
 return {'enabled': True, 'ext': build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def run_olayer_candidate(x, st, index) -> torch.Tensor:
 pack = st['configs'][index][4]
 weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
 out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
 rc = st['ext'].ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
 return out

def prepare_olayer(weights) -> dict:
 weight = weights.get('o')
 if weight is None or weight.shape != (8192, 1024) or weight.dtype != torch.bfloat16 or (not weight.is_cuda):
  return {'enabled': False}
 if torch.cuda.get_device_capability(weight.device) != (9, 0):
  return {'enabled': False}
 st = prepare_olayer_candidates(weights, OLAYER_CONFIGS)
 st['token_configs'] = {bn: OLAYER_CONFIGS.index(cfg) for (bn, cfg) in OLAYER_NEARBY_CONFIGS.items()}
 for (index, cfg) in enumerate(OLAYER_CONFIGS):
  m = cfg[0] * (cfg[6] if len(cfg) > 6 else 1)
  run_olayer_candidate(torch.zeros((m, 1024), dtype=torch.bfloat16, device=weight.device), st, index)
 torch.cuda.synchronize(weight.device)
 return st

def run_olayer(x, layer, n, st) -> torch.Tensor | None:
 if not st.get('enabled') or layer != 'o' or n != 8192:
  return None
 if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
  return None
 if x.device != st['weight'].device or x.data_ptr() % 16 or (not 1 <= x.shape[0] <= 256):
  return None
 m = x.shape[0]
 if 65 <= m <= 224:
  index = st['token_configs'][(m + 7) // 8 * 8]
  return run_olayer_candidate(x, st, index)
 return None
CPS32_CUDA_TEMPLATE = _zsrc('eNrNPWtz27ay3/MrkJxxDmkpiigpjmPZbh037ek4djKW0/SOr6uhKMpiTJEyST3s1P/97i4AEiRBSU7ae+oZWSAILBaLfeKlJ//yAsefDV2278yGdj+aBYk3cRvjwyeFN4ORtZPPjpOhFyT5PNu/DiMvGU/UzORu6vaTyPaSOAc2DKduZCfe3O1fR+FsGiOsJ/8auiMvcNlpu/UT6/eH7txz3H4fkqMwclwv8OFtv//kCRUADDrMmcbtVt8fLgwnDOKEMrfZ1GRfnzBeYt6FlB1P2Dz0oUXfNZ75w8a1Hw5svxE4jffW3l4Q9m3fDx07cRvzTgNAsq9bzfqWVd9q1bfaD3V2udW56j4DSLm/PfbsIHpmzBtLsy6Td1nyPksuTCzsPzOmpokIRW4yiwJE7kF0Z5brz1L0Zx56w7Q/s406BB2YVXXgL8cYkE1SpF2BtHyRQ5xKrsMdEQe8AVtLxfaxqEyjSWKkeXY9ez9Q0nEBuaFAzniG9RsDgQsQkQEV2VY7w2hIGEHCRoLB90B8xzkMhxmGOJACu8nENkZ+aCfMeO6Yl52rOivSLYdyU32wONKEJgBqxHdANtv3rgN32JhYO8HuDUhrFC4aTug3Ru1Wg8SX/o3KbPF1q1PfelXf2qlvvabH3frWG0ooxVLGhz7XRs8M57J5hf3laUtJt5R0+8rMqiGpRC1KWlmylSXbMjloyoSFBAUivnzJdtnC9a7HSczCEQtBT0BPWZy40z3mhws2uEvcGGSevW8yY277M3hqvmibzA6G7L2V5nVevAbgnRcDD9glHPI6x43SMA1dJ+Oh9+ogvLeUh2MlfaGWulBKwYjyccsPNHMm7IAds+esuXwt/mbdcrGxH0I5hbexIYTvTExd8bFXVZwdHjJrhyrheOSLYSehqTpg88rqNGcm+5MZxjHb32dWyyQsd5vNJn5mAoRVDeJ1e6elgtjVQ2gVIQCe0AEdEh09hHY1hAyH41JVYKqcjQlCaWBULgjcuRsZmRouaC+wbdPuM7PLOIueYWmW2DduUGeDWcKmydKOmWMHQQgPUTh3mZfsMTcYxiwZu2xgx57DBmB7blgcAh/b8MKL45nLFmM3gv9gUxM3KLNnGA3dqD8FpBMDPmxSLxsL4oakrHMn4TxVtVtbDrae6baEqzBvBGRPgGqpEJgTkx0cIFVH/A8oqxJpalYq5HjiTvrQYsmqSU2ZNdLvO/PE7l+7gRt5Tj8J+/HYjtwhgmcV4CM7uDEK3Y7Wd3sGyiPqO4mN9TMCRDkdHlV2amJPbb2VQXgVxkVFB+o3eOf29gQ2GX5kdVZYHGpjA2MzsKO+F3iKPYQcBVcnBMcvNSoqelAu8gAlrJ3hmdiNwU4HLHTzCrFEDPdIU9uRwExABHRh1MPo7plZjZm7nLpOJW7JciVi+DV3GxxGP1mWkOzXV+GJ0NchOXIDUA9Ev7Lw08uGRIdKNSLXd+3YbYjxpGbVRoCFi204fp/3RNOE7KpkD9FlaMVegsEXhn/DVha2thvFNrDYKsgl0HHSt9EP6RiKBNQZ+Tjgr6qCgdTXDGkMDZInUxQHSVzICSdT300gnFju7ZGtR08XXRo+xF8zPw0+HXJ5X+l89j0uPcgEo9QTHqWO8Cj1g0fcDRbiRphr2WUQhr7K00l0l2PogiYIb8qq4Ov/BgwG9Zo1ptB9Nu3Cc8rmAI+GrjG1wRjcNWznduZFKYuVuH7K/WewfwCm1P9nLHb9aapngGjwHxt8SHVNeCOVTbnTqboJb9hTMAZ57bjTSbXjNUaTilKmV1W2CLDmSpmHAlg1ErrPTy2SaDiR/uBb0DND6Diz2SAKweQysMZBbDuJFwZcr7F4HC5iNptCo1CMW3s2smc+hopAPXuIziRapdnEC67ZLIAI1IvHMAhDG+hZ4nUciD6QRTPCaDELbIDvTIF43lHj5GimnosgFxZajIEwzHiqg0WjmTUkifwCQR2yVpP/zXzfBI8GnRNDr9fs4bDDg5DOczRdHDGZMSAl0Viy2gEbNJZdSN/x9B2m73n6HtMLnl5kikGIvVAN45HfX4ZRx8i3AEoBnZaQU06M7AT8pj4vYfTTqn1UDEZzOZJeBzg6jWUd6tbZmlJ3VKooAfxvTdX7jRpYYCk9heMkjFzoRjDvD6hP1s42iiaP/OZI4e0pDH+f97glC/WjAAJenQbnEKmYFtKcxiBxQU/aicv2kcA/n14cQowUzZwEp1V+noD0ZCUKr/abhwArTkAyHc4RYFojGqiz99BAp87OjuDbgoZWgbFWg2lvCKa1GsyuANNaA6a9GsxODoyGfLrAMEznOmhKxni+MC9Fc1hlb+/s/ZVi9oznR8X3R2wbArGrFcGjjm3T95/Uwp+sVAFlHTSgIXLX2Z9/MpFuYbl/TSP7emKDqotC36dWRiGUx/76DFU6fO2zQne6rFbzeTP4d3TZgQ74rMYoiMzc98WlfwUao1TM0hS7KxdraYrdl4u1NcUWvNgD/H9grh+7a/o65n0dl/p6hH0dZ31VZxZnUGmBCAAi426uQOa1olYfsx9gkMCC4bAmlsyxMMdKK2YTDRCMo1Kj7zv8btFzQrXr7AhJuIONIsnNCgD3AsBCALirANDRArAEBlaGwX0FgN0qAPcCgMRgUQHAapnKcAmD/i527KkLwTGYuz1wU25p5mEXwnffDlx6aMMD+EkeOj40qQDPY9sfYfo1pF/TtM7vH87RrGOAPfauxzQ19O+YgVyEAUBnUCZuFET97Lws6VM7ccZGXozPzlWhdbXTOujJAX+6ONtC0x5tSzets4x4KZr++RmiaeiEsQvUMXjV11jVgvxtPmdT4uaUlyPOy0AS6Aeyb5R5CoSMKi3grhxdRlfsjwPAQDpTR8MvtoO0QVLH0APwjGyZJ0x67IdJXGcLLxkz13bGfFjAcyLXKRmDNcIiQOl46nvJixMGHmvi2X6O2I5vxzF7N/GSMsHBj505bs5X2GZx5HBv4exX/v1RfPUuznkq8YY8ESR1gsxc+MeHJqWRN1wClaBol5JAqV95ChwYGQGzNGiBoqo70myMwFFW/wn+TcFP+RBMAfBHHIGpqbpa4O9ANy6N2LuHCMacwpAi+jDWgMGVgIVIG/AMngXlSLkA5R3ssoS7uOAbwhicXwBLvIjCBc9gUxCHhR2BV9C7AP4A/sZZzZGP86X0UhZtsJ9+PfqFGeA44qCFgX9n7jGLBSE7PT2qY3MtfFjyuag66+ATjQu61XWQOD5JhTXrKNAwsMkdu3GjwPXrDMIKgxgE+B4GHJAIZ8l0BpyB3gvAA1c/LWAD9JO32GaMi0SA6wucgaNpWTEpBjyFYuwGMMJWa1fU7ZiIVQ5yo2y6OU/I73Px3ZPP737iCSSIYLCL06PfefL4VHyfie8T8f35XCZkzsknkejx799F1QsB9fjj4ZN+n0c2cnKx3/ftWeCM+wMMYuJ+36C2Jc9cu5O58aRofPogH0Aoz0kAzGIqvfaCj5krtZSzgrlcJFz9Sdl0FYq5sRC8Sd+OrgWJsuRJlvx8rqSV/JNPPL3ayVnt1aQUYDRZfgpmFP7tcaS6uQJnWOAMC5xBAYHqCWaeYOYJZJ6Ua30+xxLw/wf6v5f25jNV/Ux1P2Nl2bWTTwTzEwH9hFA/5cDmPcyCJyX8zYLPUQc+vcDsC1ANF7/kQdEMw7HVgvfCk7MUr66tKfyfX99/EKUP0bctlzj7APIORUgdPMd5gLMPv2fPLXwGEclywPs//5A9gvv97vTjxf8oIHY0zZx/6PcuAFBWrN1CQP3jDz8pmTsYW3wolAR510D8+Kn3H7SbhBz0DSkh0m0ljYsFz9lT3glMAfL4RUgj2FmMCvDDLEGy40J1YE/AmiU4kxIGQw/Vne3vIyVFrFXPixqMG1ZDWAhlG8UKQEUujLwbTSGo7Tt2nOzTy0MDhY7UuruE1wGFljhxQ4sPNN0GWgDsPKjvmObemAPvGU6dX15hPW0F0KLpdEKMExN53l7yVTEg1++nKbV4sp0lX5nAych/26Det5lxwl4y47iH7N0D7r6OvOFP3qRxZ5LrByVakNvkOOV0T7zUESBf6NDALpl5PCN7ITE9/ggNgBxuo/xto6XbRjnaZq+sVr5SWiWtXZP97Ahh3ibJlVAw6Go1O7uIPV/kIL0aL3RY0zuOLMDlhJQrI0qYRfwkvYdspicNQXJKHN0P5LYx+le/DpfocE8X3C1Bj+9Vnaw4ZEDuS7IgyTV/2qIn8rh4cfQquwXwQZJTZbL3F7/AvzYOGhlVHMtlserZZ6gqyA5YoEdJqGwxtHiLG/n8Ep6LdXvYapldisV++V3Rz8Bh1BqNDLJ9VrGE2y9QDypvQ0Mwip+Lr6+RggZ1DWl6B4WgdI2lOUuT6kEWdqFYPaKALau/5PwHhSOTECxWcJql9oDMUP4Gy598Kpa/+A29SS8wMHwFMaszA1kK7BuuxppdjSXGWcV0bjBdPXtOEi4q5FabyPmULr/Cnagss0hWlFQXzWQcR5ECvcYpA1CUxKmQzmqXcETvPq5QxR1UKVnMYRg99gJ7jOOAY6199fk8xYeV1q9oacjSFxDLSFSEYyUYS6LWSRVcByPw3yimau3OSuDUBR/57iEX6yrLNTKE1RB+jbotzS2wG6QkKt6ebDYNLeYybpnjXEWqpqHGS7aLubkIpqBwcGIQhY2XNsFppJytNKdbqMX9RYhWoNwSNaqIWZCBwQRO5GCdqBJ2x/GpAXRALgPJgQ1jRCNGaAkOCpXcNcvFdTTUE07DjrFdlhhoWelgce3BmYplJ+darKKIvUhiTYkvo4Bfg6tgfH8PrRv5kIic0sqI4Bau+bdLBmUmLArilGKKW70QVpHX1lOiqishhOMJ31ZXWr0TwAnBPs1oc1MUG8p0jKJaJnd9NDfN5c/ib6ZTMuiZcg/LlDWyLWEQRChMBKqaLFGNLBk0WgpyFrdoaabFOkbmB7Rbpqivak5emMUYvQwRBVn9l1JttVrBgVk2C1wPun4bY1bIIdsLVheZH0cEHOs3oFbQm6DpGU4Z6gQ4JBF6l49xK4A5Z0mIM9XXbl/udDpgl8+vSAngzAoPqmIvE4PCQHxkh3mNjWo99kBtnHyqmPst6JovB80u+7IPRo/Val/M6hrqrDHW8fdLk8QVcgr8f1AQU0kvnPoysKfbZ59raKvN7fMLQGP77D0A3m63aoJtdA0IZy5yDha3NTl+sbctWKJmfEnBFAF8r1pAkV6nF1TpVlPfIcccyEN35Tz3LY7O7f4xzUrdmnkGM27rtwIcrWQsLnsXV5fn8Dl7f5X5GR22nNObU/i0rhR2xckgPZdW8lvGa3zK7Auw5zp+061QpPxWEgLutyDvg5g+ResPijO5FvmLS8Ty6vLL1aV/peiqhQG6J5N89CGkNoHcL1yBgOIxVS1SFMGnGEVn4abQ3M1Mc+9kyTdZ0mqaK+Qz7X3Aex+QH4C9D3QWsajTMJ4AxZY35QbQA71RyA5M8iikigOntGUWDH1MWlBlZmAITsbgSt1IiPZsOTWrSlrFkiJkE7qUYpRdU7Wmj2H1mJMnBvL0iDzAhMigRlxnUsmWd89N6iyNzR/jglQILu0QILEFXCoEt8L2ZjPQzHacy7MjLooodJ2r7qarWXIB62+QPg3/bWBTHF7LgVodrAQaErs3JhEEpnCQKZqNUSGuGYX+MItr0uxgmWhdEmjInVbFQaqLwgGIlf3yqj56MHWaHyh0jVl5knu8W9y64je44b2L9WpPx6U61z3GBjwUvpyDbPT4rB/2B4qA5KBtHwAr3VQ50tjrjYz/5iyxTikXDCyN5R/qKpQBaljoYIi4/2DK413+8T7/uMgpF+Hg1HDdg3s5qcDz3FzpdcKv+O+qWs2icpLF3asCBiWF8TpTGLuKkrdWDsMj9Pzfq+u9krLPe3Eddt88yKn7Oru3cjm1lUqd/w246Ti4b+IWowE3D/h0J55a9HQvntr0tNBC6cA7K4Xyip4klB16klBe09Oiyi3blEnyddZ54ZtZi62mcCiDZwDjhVXwIzfhm0d474/w38uRNvjYeg++ZqAD720dfzQf58Jz//M5mx9kol4uUzo6JNzzeYeO66w/85QdecpOPOUOPEXrXPj8yD/otMQRd6DFFp/utyneSvdV3YREy0dSk9ahYfwvT3vgatony+yuZdR8RECjV4+9GozhJriv11X5VsFBXqMGFeAuB+4C8F2E7ZpcjF1yGZbt0W4TP7OCPHMVnm/3ka64Vlehcp0jtelJUVuFF5kGK7zIlFnhRZVeU4tZV4qKK7zItF3hRab4Ci9W6MCHv9XAanyzTe2iTk7WzDbMcB/RQbYuhFPb17R1GWeoUzfy3Sw/KU1G8Bwt5peSAqCjMehF4vT9Lh2KwVZ0inMYajI5BL6Uc0AdMzN4bbFfyDTVTUj7QpscGlzGobRGhTKUkKkun/u8Lu5OAT35wzrv152iBiz52HkZE/uUdZRYpTJXslrJk+polImR6sJN1cgjHdv8/Amve0uRFS4g8HkUDZHjxaVh0CJZyjpy1REnEG9Nde4TxFG7MZl2HaEtpJG+vN3uCMWOyZqlPrTUh/aVucqp0UacBZo+/W/TVCsryj5bXAh5HJF1spDS9WCO+lShLWTcdVX6QsZ9V6UxZCw0svDN7L2jDxSamypEhZCSLzcbvUdF8huMvGbc0sPOGOw3RbDPORYHCgesLszk9u1VloSByHvaOmuuzMZ8G5025vNA7OwLUHkaFESBa/4SN4wD1abV7jItv4W44CrDLx55BVNS5cqSwmtc0DDkzh6wSmiLZHzWoYhNo+2VBekQI3BjcUPLzxTCmXLZ4jZtLIVIJobCvd3uOlffHg41C9F8KS+8kZtFeul6XuhrMFXPsg+s+qBVH7Q1AUBugAU1dJxVjAImdhJ5y8JR+N1gt7FsyQBhACTg0QHFBS0lLuBnzQ/4YXN5yBJ6bWo6goz4XQh1yghVhCs5tGSqlabaRVS/Q3j1alcnvsi5rbIMQ6AxsLT+SGlMccPDCth0UqAEvwWftrlO767UwytOBPxFk2CbzNlkgrAszCJpDcQhqgNwWCG5L3zXpyJ8WqdyHjHtU6UDSlNL60R29Z4HDXfQbFWsLrfqdIper+WWY/nWw7KsaubDeAsr5rlYOseFs1rf12etZN3jGnN5e4Ayv7Z9WFxDSetaG9XNHdHQSEEFUvklHN7YYxZrquSSIb9nM31LZZ5vqczyLavm+Kh+Nse3VGb4lsr83lI/u4f4bGbqtV7C6vGnbbRrnYm1SzW6SEDOmZP7mZ3dge/bK/aHLpufs6p61YJX+rOQqmNcqtbWOtHphP5KPfUDDTDHeE84ePRkApabVuW9UqtTzuP4e403Sp2tf3Nv6t/dmQ2sJ+5ELtpOiHtAhRd7Q/7eP7FLD3/lvEBlSFtYM9Ztz5NTn9+8hqtbYNMvvdFO9+KeHL7L96C4TJC53aAjUMTA8REpEBhrxnfviccWPA6zx3Zukqbgxt/kzktO2J8MT6Std4YUjbijm/go3mgybPhhdocJfsB/23pF9xkoLy1xwYn2ZUvc4yBfPqtWWs9yFdvy8geqSJ50Td6YUpOXdGHCkYn0VpUbcc3DbZ5rxRK0DeQdwMeBz7CKzzQjT+cZikOvOC7EAKXZU37OGw8T1Ghvci5YFJ5bv4+D2bdjfkDNIESf04y4vJsn2yVU1IqIBe8ZzqG/cduvX78ZvElv80nbrzMrPeeW9TLb414CzbcuROs2LfxTNiFE6v4DOjHzA1OtA76osZKaxey9YkH9bqDUO1hL+giBNcUagyX3QqTaS3Xwz9/Knfpi73y3VCQKF+jP5ba0n79V93rzczHgqsod0rTpG088YdC9yV6kR8VJ3zrvtGr4vj5ZHQLpwh80T4oAglQ5NBdSuaOQDk4CLBREMddYlEZ0thGso27nz0+uyqNNP8jrHlByUYKNjAfNAk8Aj2E/9tmk7B6WFUU+vDlD1REu6gqLV84Oq/Jcpcv4MSqV5C9fsp9nvs/PBO+Jw7UGmoEczdOZ2AY7DidT2+FbBGOaC7DMYsWL33BeSQ7WdSPbJIl7DfF8bH+ErWYbDum4rDy0q9sMRwvcyHXYEq6sAFFc2o+OGfzMTUAbdSFzix/PIGxEFh6/vShbVnH5xpwv0X2lG0nowhG6OmTeWDx0v00p5bm6yNNBjo+DlVxc5GFSCjUmSC24ONiAh2l5QLDi5qwHtClwnbL7tasZWUcwyCMH9zod29fFoQVnIMmG9uK34she/PaPGVgc0OQ7RvL6vzeCBbdHmpPsLJCV77w8148HdXQ75MX7wh55PF+lbpYvUubs15xV5MdkcKf3/+uMX0FTYxcvaUcM0be0bIVY1ljBTq1ZIlSvJZDqHXely7SlpFtKurBAuGJBUD+gJQ5Wb2zAjtbpggY8PYjfdCcD3seQam3dok4epOQMJ57hWW9ikIw7zz6XB/dbEKrSLtTsJd7HIC5TKklh8dbq9J46ESs6aJl0xZKxF/dF2eLiq+PTQoFRLfPkzOAuAShKLl3h7F6xAt73ADVAImpMnK1TjpUVS7uWmC5GIhnQWI2vDQGUQguFqzSiG16KNFF6q4ZraW7VKA4xreRuQ3cm9lSeaqY+0cjX+NA0S65ULiy1eFjaExN1+cs21sG+Lc9OpHyqXsFRuUdQN2gPq/aQqpoKrYNyMLGstegSApkJTlFhpS5/vE+6s3Uyhalf2tWo3Mh15t+rcsHzuxi7zAe94Ub/jlm4CNJ7XnD3h7j2g1NeXM7RpetCoN1JmLjs+OIoZtNZPOZXyaSXP65TQYXzpTp1lN5yWbZ/KcFxMx0wDh4CXbN3KFJP4dJ1r3TCtLluue0v2SOxyXpSgRkCTdSDayVJxeRiniZBBVGKzXDpV/jXMBbRI8waLm4Xa3DHPw3Trrv6GXHBxvw8KTFzZh748eXcUWyUNBL47oouVdz48x2mNU/j9FQ1YMQRn+sLk8oo3qiq8J7maCvyYh0vU4yyc+HrlmIeVqi1FF++FSgfGm6wkahwR2YZKa1cd8y87caBRZvt1OnAPzfgzlqXIkfBR4ESkcfmDkrquuY4sMdZblPV819RIRWKQKs1OKZaXuWOrXRrb1JZ0wu2Fnoq61U74f5CZ/dRu+FWWpLNLcRa19Spa/hxg9B3rSYUirBS+22olf5iffSgscl/q66pVASb64DKJSVxOZu4VpQi/owa4TQBd5oPmlICLcwLS96tj3c5pDEVvh668+z0Gv7wzi9u8hNdWGw8h3dm9oLnwuujJIm8wSyBEtA58QozT+1lj/yvU1q0+uhGbzFu+ICI1ZmE9lD8LRVw7W5nbnQH0Tu/ngw9N0uUiUEYlwxvV54DW/BYJmaGSIkKJgPdQzEKv5Oud9oFt00CiNzreNtqNps1/LEdn/9kxz/74rbSCIt72+hHF+jWJbyITaaXWZLWT+RDdpOawFdcjyZvT5M3p8lb0+pVCpHokMFNmlb2MFMf4gSERrRGYyquXcfZNryrL7t42p3M94nm+EFaI52Jxpy+grZIV6Qp0pNoSXQkGiL9kHZIN6BZ/pKmaeh7zh00xznrJYPxxxL8cetAPqNM8Dx5Ua1k95+B4Cmnx2xkd7NbsMVrVRZi4/nIrlMnTdQtWKY3cxw3jqWWATEUciaeR3YjmE3OgT1xPg0YFLUpRtx0HRgEx/tctR4aUJJ4twfPb5F96+zNmzepOJEtPT7FtZQJNn58atIiy/EZZp1R1pnIOsGsE8o6MRXUWunPb+AdTrgqc06FPp+Lip9PxFoN5Z5IcJ8I3icO8JPI7WFmj/J6pUaqbn+ia6RoukXcHFXXLzohkpC3z6+jA4RECuofMnGhYdqirHDGtgBa9iLrruBdsyfv/oLO0G4vwP4kq9BOK/Qu8FVb9HuLTt9C85Dep7Ss0ZE1JhLDCSAortqj3bVp0VcZXfi9/r6IdZ4K3YxgnkKuJKuV1X0j3/PXRg9ao5i1x+hXdwolCxdESVYTt0QRbtlNUS8Zv3FAXnMyYSo+hry1bzc9u6T4nXjEC8sDTQ9pf7FuRk3S4pWVXu7G0o3XRjaTS5DSmIsu1ULG1MHRwyhE8Tl4/H4v/eVNEnDuAieliTxiuHsQRWO7OJEiKklhXbN/D4DWMqgKxxR2KW6zlI3T/Sn5uoUb7cq3gePFdDTCuWp0G54yTvm7JDM6K3YKzHR/eBfkvA2Zp/gcK1TrOrXKS2XtaHyfF1x/juQv3KT6Mt9CT3ViEHa9jBc4Mz/dBfbEc1SfBgHWJQ6mYhJ881HQz8LgYxgl9sB3j7krg5CPfD9cuEN54dhDUUXFqEBk46lY8wswoYX35CQch8HIuwZp9XHC7+tD/m2KAbMTcaeK7zTEHXhQfuhN2gbek3f+lnxWjgkUkXf4yTLowlrK+6EgFgzHW3FBYjwR77hbgIJCZpGesDuUyNSX5BHArHnVoDsLNYgLegEqbhB7oXDeeZ257csfKKEr/WjCtvo9Ogi9Fe/vs/rQCxswwG7ZSVrFWotmD9zZ4cz3guuP5JJ8jNyRG+F1byoQpd3qCqStUd7E7ltos6KRQ4P7P2aKO/gZRwL9lspaqe/TxqUIihfwWDbCfhdFYYRe5AE+fXCc2dQOnDuQiyPyxsm3j8HJP535iTeNQvR1wgiDAmJ6YJB4YnYlm7oHB4pP9MN874WFP8VQxKOT8jX3hCYlL81Sw5dC8JLijGPigwLXoy7IFlMAwyX0ue+YVe7vHns0BQRUFJJ4Yuaiq/cweISnDC0lgbBzCokA+zk0jWQSA5Zx2Qnde/1uaQDasq2K7dlid/YCr0uoKFK8KXVZ9eslpfp00R5dMlsNPb3m+RBvZcP9/3WMRMDhQ38eHb/KyEM5GQsxB+00k4dGcaeocpmjvI3WxHK4Bt/KF5iVqq9tMYM5S2GqvwhUHs7yAOKRqhe7FOUWYzk+3wIEyqKn/4+Y7ntDuXjhJc4YhTOmdbYffzw+6r3r/fgjtTN06VeO9jKX8428Uv7/ANbvdnQ=')
CPS32_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
CPS32_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
CPS32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
CPS32_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
CPS32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
CPS32_ESC_CAP = 32

def cps32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def cps32_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def cps32_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cps32_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def cps32_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = cps32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def cps32_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = cps32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def cps32_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def cps32_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def cps32_pack_format(w, fmt, layout=0):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [cps32_units(hi), cps32_units(lo)]
 else:
  mats = [cps32_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([cps32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = cps32_c12_bytes(v)
  if not torch.equal(cps32_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(cps32_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def cps32_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def cps32_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = cps32_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, CPS32_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > CPS32_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = cps32_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.cps32_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
CPS32_TABLE_M = {}
CPS32_MAX_M = 32
CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def cps32_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cps32_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def cps32_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cps32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if cps32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cps32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def cps32_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def cps32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * key[2]
 rows = cps32_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
 pkey = (layer, rows, fmt, layout)
 if pkey not in packs:
  packs[pkey] = cps32_pack_format(w[:rows].contiguous(), fmt, layout)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = cps32_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
  lo = 1 if total_mt == 1 else 4 * total_mt + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
  streams[vkey] = cps32_verify_onehot(ext, inst_id[inst], sp, cps32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and cps32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
CPSSM_CUDA_TEMPLATE = _zsrc('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')
CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
CPSSM_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
CPSSM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
CPSSM_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
CPSSM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
CPSSM_ESC_CAP = 32

def cpssm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def cpssm_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def cpssm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def cpssm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def cpssm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def cpssm_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def cpssm_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [cpssm_units(hi), cpssm_units(lo)]
 else:
  mats = [cpssm_units(w)]
 if fmt in (0, 2):
  data = torch.cat([cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = cpssm_c12_bytes(v)
  if not torch.equal(cpssm_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(cpssm_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def cpssm_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def cpssm_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = cpssm_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > CPSSM_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = cpstage___to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
CPSSM_MAX_M = 32
CPSSM_TABLE_M.update({('router', 8): CPSSM_TABLE_MT['router', 1], ('qkv_gate', 8): CPSSM_TABLE_MT['qkv_gate', 1], ('pooler', 8): CPSSM_TABLE_MT['pooler', 1], ('o', 8): CPSSM_TABLE_MT['o', 1], ('unpooler', 1): CPSSM_TABLE_MT['unpooler', 1], ('unpooler', 8): CPSSM_TABLE_MT['unpooler', 1]})
CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def cpssm_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cpssm_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def cpssm_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def cpssm_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = cpssm_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = cpssm_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = cpssm_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = cpssm_verify_onehot(ext, inst_id[inst], sp, cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def cpstage___to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cpstage___to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def cps_build(family, instances, depth):
 is32 = family == 'm32'
 prefix = 'cps32_' if is32 else 'cpssm_'
 template = CPS32_CUDA_TEMPLATE if is32 else CPSSM_CUDA_TEMPLATE
 cpp = CPS32_CPP if is32 else CPSSM_CPP
 cases = []
 for (i, inst) in enumerate(instances):
  full = inst if len(inst) == (16 if is32 else 14) else inst + (0, 1) if is32 and len(inst) == 14 else inst + (0,) * 7
  args = ', '.join(map(str, (*full, depth)))
  cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
 source = template.replace('@@CASES@@', '\n'.join(cases))
 name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
 return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def prepare_cps(weights, depth=None, layers=CPS_LAYERS, buckets=(2, 4), include_policy=False):
 if depth is None:
  st = {'plans': {}, 'depth': 'selected', 'policy': 0}
  selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
  for (bucket, selected_depth) in ((2, 2), (4, 1)):
   if bucket in buckets and selected_layers:
    part = prepare_cps(weights, selected_depth, selected_layers, (bucket,))
    st['plans'].update(part['plans'])
  return st
 if depth not in (0, 1, 2, 3):
  raise ValueError('ring depth must be 0, 1, 2 or 3')
 st = {'plans': {}, 'depth': depth, 'policy': 0}
 for family in ('sm', 'm32'):
  table = CPSSM_TABLE_MT if family == 'sm' else CPS32_TABLE_MT
  entries = []
  for ((layer, bucket), (inst, cfg)) in table.items():
   is_policy = include_policy and family == 'sm' and (layer in CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
   if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
    continue
   if not is_policy and family == 'sm' and ((layer, bucket) in CPS32_TABLE_MT):
    continue
   if len(inst) == 7:
    rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
    inst = inst + (0, rows, CPSSM_LAYER_K[layer], *cfg)
   entries.append(((layer, 'mt', bucket), (inst, cfg)))
  if include_policy and family == 'sm':
   for ((layer, m), (inst, cfg)) in CPSSM_TABLE_M.items():
    if layer not in CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
     continue
    rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
    if len(inst) == 7:
     inst = inst + (m, rows, CPSSM_LAYER_K[layer], *cfg)
    entries.append(((layer, 'm', m), (inst, cfg)))
  if not entries:
   continue
  twin = cpssm_bf16_twin if family == 'sm' else cps32_bf16_twin
  instances = list(dict.fromkeys((inst for (_, (inst, _)) in entries for inst in (inst, twin(inst)) if inst is not None)))
  ext = cps_build(family, instances, depth)
  ids = {inst: i for (i, inst) in enumerate(instances)}
  plan_fn = cpssm_plan if family == 'sm' else cps32_plan
  (packs, streams) = ({}, {})
  for (key, (inst, cfg)) in entries:
   sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
   chosen = inst
   if sp is None and twin(inst) is not None:
    chosen = twin(inst)
    sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
   if sp is not None:
    st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
 return st

def cps_launch(sp, x, out, n, policy=0, query=0):
 fn = cps32_launch_raw if sp['family'] == 'm32' else cpssm_launch_raw
 return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def run_cps(x, layer, n, st):
 if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
  return None
 m = x.shape[0]
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', cps32_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != CPS32_LAYER_K[layer]) or (not x.is_cuda):
  return None
 if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 rc = cps_launch(sp, x, out, n, st.get('policy', 0))
 if rc:
  raise RuntimeError(f'cpstage launch failed: {rc}')
 return out

def cps_resources(st, layer, m, policy=0):
 sp = st['plans'].get((layer, 'm', m)) or st['plans'][layer, 'mt', cps32_mt(m)]
 x = torch.empty((m, CPS32_LAYER_K[layer]), dtype=torch.bfloat16, device=sp['wp'].device)
 out = torch.empty((m, sp['rows']), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 attrs = cps_launch(sp, x, out, sp['rows'], policy, 2)
 return {'registers': attrs // 1000, 'local_bytes': attrs % 1000, 'active_clusters_or_blocks': cps_launch(sp, x, out, sp['rows'], policy, 1), 'blocks_per_sm': cps_launch(sp, x, out, sp['rows'], policy, 3), 'dynamic_smem': cps_launch(sp, x, out, sp['rows'], policy, 4), 'fmt': sp['fmt'], 'inst': sp['inst'], 'cfg': sp['cfg']}
'BF16 cuBLASLt projections with offline descriptors and a cold-L2 tuning probe.'
import statistics
import time
import torch
from torch.utils.cpp_extension import load_inline
tt_lt__LT_CPP = _zsrc('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')
tt_lt__LT_EXT = None
tt_lt__LT_WORKSPACE_BYTES = 128 << 20
tt_lt__LT_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
tt_lt__LT_H200_FIXED = {(256, 'o'): (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (256, 'unpooler'): (3, 0, [66, 20, 1, 0, 0, 4, 35, 0, 2]), (128, 'qkv_gate'): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, 'qkv_gate'): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (128, 'pooler'): (2, 0, [66, 15, 1, 0, 0, 4, 35, 0, 2]), (256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])}
tt_lt__LT_H200_VERSION = 130101
tt_lt__LT_SEARCH_SEEDS = {(128, 'qkv_gate'): [(2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 4])], (128, 'o'): [(2, 0, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 2, 35, 0, 5]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (128, 'pooler'): [(2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 2, 35, 0, 5])], (128, 'unpooler'): [(2, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (2, 0, [66, 28, 1, 0, 0, 2, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (256, 'qkv_gate'): [(0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 9]), (1, 0, [66, 134, 1, 0, 0, 2, 35, 0, 9]), (0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 2])], (256, 'o'): [(1, 0, [66, 20, 1, 0, 0, 7, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 4, 35, 0, 5]), (3, 0, [66, 20, 1, 0, 0, 5, 35, 0, 3]), (2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5])], (256, 'pooler'): [(3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 4, 35, 0, 3]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 8]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 9])], (256, 'unpooler'): [(2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 5, 35, 0, 2]), (3, 0, [66, 20, 1, 0, 0, 0, 35, 0, 3])]}

def tt_lt__lt_build():
 global tt_lt__LT_EXT
 if tt_lt__LT_EXT is None:
  tt_lt__LT_EXT = load_inline(name='muse_cublaslt_v1', cpp_sources=tt_lt__LT_CPP, with_cuda=True, extra_cflags=['-O3', '-std=c++20'], extra_ldflags=['-l:libcublasLt.so.13'], verbose=False)
 return tt_lt__LT_EXT

def tt_lt__lt_state(weights: dict) -> dict:
 ext = tt_lt__lt_build()
 device = weights['qkv_gate'].device
 workspace = torch.empty(tt_lt__LT_WORKSPACE_BYTES, device=device, dtype=torch.uint8)
 return {'ext': ext, 'version': ext.version(), 'context': ext.Context(workspace), 'configs': {}, 'weights': {name: weights[name].contiguous() for name in tt_lt__LT_LAYERS}, 'packed': {}, 'workspace': workspace}

def tt_lt__lt_plan(st: dict, layer: str, m: int, layout: int, epilogue: int):
 w = st['weights'][layer]
 (n, k) = w.shape
 if layout >= 2:
  if layer not in st['packed']:
   st['packed'][layer] = w.t().contiguous()
  w = st['packed'][layer]
 return st['ext'].Plan(st['context'], w, m, n, k, layout, epilogue)

def tt_lt_prepare_cublaslt(weights: dict) -> dict:
 device = weights['qkv_gate'].device
 props = torch.cuda.get_device_properties(device)
 if (props.major, props.minor) != (9, 0) or not tt_lt__LT_H200_FIXED:
  return {'configs': {}}
 st = tt_lt__lt_state(weights)
 if tt_lt__LT_H200_VERSION is not None and st['version'] != tt_lt__LT_H200_VERSION:
  return {'configs': {}}
 for (key, descriptor) in tt_lt__LT_H200_FIXED.items():
  (m, layer) = key
  (layout, epilogue, config) = descriptor
  plan = tt_lt__lt_plan(st, layer, m, layout, epilogue)
  index = plan.restore(list(config))
  if index < 0 or plan.metadata()[index][:9] != list(config):
   continue
  x = torch.zeros((m, st['weights'][layer].shape[1]), device=device, dtype=torch.bfloat16)
  out = torch.empty((m, st['weights'][layer].shape[0]), device=device, dtype=torch.bfloat16)
  if plan.run(x, out, index) == 0:
   st['configs'][key] = (plan, index)
 torch.cuda.synchronize(device)
 return st

def tt_lt_run_cublaslt(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 entry = st['configs'].get((x.shape[0], layer))
 if entry is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
  return None
 w = st['weights'][layer]
 if x.device != w.device or w.dtype != torch.bfloat16 or tuple(w.shape) != (n, x.shape[1]) or x.data_ptr() % 256:
  return None
 out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
 rc = entry[0].run(x, out, entry[1])
 if rc:
  raise RuntimeError(f'cuBLASLt launch failed: status {rc}')
 return out
'Experimental lossless BF16 register-A WGMMA projections, one launch per call.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def tt_mw_mma(n):
 nr = n // 2
 regs = ','.join((f'%{i}' for i in range(nr)))
 outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
 inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
 ops = ['wgmma.fence.sync.aligned;']
 for kk in range(4):
  ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
  ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
 ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
 return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
tt_mw_CUDA_PREFIX = '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t t0,uint32_t t1,uint32_t* a) {\n  uint32_t cm=c&0x77777777u,lo=prmt(t0,t1,cm),hi=prmt(t0,t1,cm>>16);\n  a[0]=prmt(l0,lo,0x5140u)|((c<<12)&0x80008000u);\n  a[1]=prmt(l0,lo,0x7362u)|((c<<8)&0x80008000u);\n  a[2]=prmt(l1,hi,0x5140u)|((c<<4)&0x80008000u);\n  a[3]=prmt(l1,hi,0x7362u)|(c&0x80008000u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
tt_mw_CUDA_CORE = _zsrc('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')
tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def tt_mw_cuda_source(instances):
 cases = []
 attrs = []
 for (i, inst) in enumerate(instances):
  (bn, r, wk, pf, q, c, cap) = inst
  params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
  cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
  attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
 return tt_mw_CUDA_PREFIX + '\n'.join((tt_mw_mma(n) for n in (16, 32, 64))) + tt_mw_CUDA_CORE + tt_mw_CUDA_MAP + tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def tt_mw_build(instances):
 src = tt_mw_cuda_source(instances)
 name = 'm32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
 return load_inline(name, tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
tt_mw_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

def tt_mw__to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_mw__to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_mw_sm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_mw_sm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def tt_mw_sm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def tt_mw_sm_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_mw_pack_weight(w, r, wk, split, compressed=True):
 (n, k) = w.shape
 total = wk * split
 rw = r // 16
 if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
  return None
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
 esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
 (table, max_esc, escapes) = (0, 0, 0)
 if compressed:
  (data, b, entries) = tt_mw_sm_c12_bytes(v)
  if not torch.equal(tt_mw_sm_c12_unpack(data, b, entries), v):
   return None
  table = tt_mw_sm_table(b) & 18446744073709551615
  (er, ec, el, ev, xor7) = entries
  escapes = er.numel()
  if escapes:
   boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
   rank = torch.bucketize(ec, boundaries, right=True)
   seq = ec - k // 64 * rank // total
   group = (er // rw * total + rank) * rw + er % rw
   max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
   if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
    return None
   order = torch.argsort(group * (1 << 14) + seq)
   gs = group[order]
   ss = seq[order]
   pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
   entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
   esc[gs, pos] = tt_mw__to_i32(entry)
 else:
  data = tt_mw_sm_bf16_bytes(v)
 payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
 return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_mw_launch_raw(ext, idx, pk, x, out):
 return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_mw_verify_onehot(ext, idx, pk, w, m=32):
 (n, k) = w.shape
 eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
 out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
 for start in range(0, k, m):
  if tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
   return False
 return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def tt_mw_prepare_m32w(weights, configs=None) -> dict:
 configs = tt_mw_CONFIGS if configs is None else configs
 device = next(iter(weights.values())).device
 state = {'plans': {}, 'device': device}
 if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
  return state
 instances = []
 for (inst, split) in configs.values():
  for c in (inst[5], False):
   twin = inst[:5] + (c,) + inst[6:]
   if twin not in instances:
    instances.append(twin)
 ext = tt_mw_build(instances)
 state['ext'] = ext
 packs = {}
 for ((m, layer), (inst, split)) in configs.items():
  if layer not in weights:
   continue
  w = weights[layer]
  for compressed in (inst[5], False):
   actual = inst[:5] + (compressed,) + inst[6:]
   idx = instances.index(actual)
   if ext.info(idx) % 10000:
    continue
   key = (layer, inst[1], inst[2], split, compressed)
   if key not in packs:
    packs[key] = tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
   pk = packs[key]
   if pk is None or not tt_mw_verify_onehot(ext, idx, pk, w, m):
    continue
   if m == 32 and (not tt_mw_verify_onehot(ext, idx, pk, w, 33)):
    continue
   state['plans'][m, layer] = (idx, pk)
   break
 return state

def tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
  return None
 m = x.shape[0]
 if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
  return None
 key = (16 if m <= 16 else 32, layer)
 plan = st['plans'].get(key)
 if plan is None:
  return None
 (idx, pk) = plan
 if n != pk['n'] or x.shape[1] != pk['k']:
  return None
 out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
 rc = tt_mw_launch_raw(st['ext'], idx, pk, x, out)
 if rc:
  raise RuntimeError(f'm32w launch failed: {rc}')
 return out
V9_SELECTED_ROUTES = [{'m': 32, 'layer': 'o', 'name': 'ol_1', 'module': 'ol_v8', 'config': [32, 128, 8, 1, 1, 1]}, {'m': 32, 'layer': 'pooler', 'name': 'cps', 'module': 'cps_v8'}, {'m': 32, 'layer': 'unpooler', 'name': 'mw', 'module': 'mw'}, {'m': 128, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 128, 'layer': 'unpooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'qkv_gate', 'name': 'lt', 'module': 'lt'}]

def prepare_v9(weights: _AchitaDenseWeights) -> dict:
 state = {name: w.contiguous() for (name, w) in weights.items()}
 hi = weights['router'].to(torch.bfloat16)
 state['router_hi_t'] = hi.t().contiguous()
 state['router_lo_t'] = (weights['router'] - hi.float()).to(torch.bfloat16).t().contiguous()
 state['router_weight'] = weights['router']
 state['sm'] = prepare_small_m(weights)
 state['m32'] = prepare_m32(weights)
 state['wide2'] = prepare_wide2(weights)
 state['router'] = prepare_router(weights)
 state['qkv'] = prepare_qkv(weights)
 if state['qkv'].get('large') is not None:
  state['qkv']['is_enabled'] = True
 state['router2'] = prepare_router2(weights)
 state['olayer'] = prepare_olayer(weights)
 state['cps'] = prepare_cps(weights)
 state['new_lt'] = tt_lt_prepare_cublaslt(weights)
 state['new_mw'] = tt_mw_prepare_m32w(weights)
 for row in V9_SELECTED_ROUTES:
  (m, layer) = (row['m'], row['layer'])
  weight = weights[layer]
  n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
  x = torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device)
  candidate_v9({'x': x, 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}, state)
 torch.cuda.synchronize(weights['qkv_gate'].device)
 return state

def candidate_sglang(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
 if layer != 'router':
  return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
 if x.shape[0] <= 8:
  return F.linear(x.float(), state['router_weight'])
 output = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
 output += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
 return output

def candidate_v8(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 m = x.shape[0]
 output = None
 if layer == 'router':
  if 2 <= m <= 8:
   output = run_small_m(x, layer, n, state['sm'])
  elif 65 <= m <= 192:
   output = run_router2(x, layer, n, state['router2'])
  if output is None:
   output = run_router(x, layer, n, state['router'])
 elif layer == 'qkv_gate' and m == 128:
  output = run_qkv(x, layer, n, state['qkv'])
 elif layer == 'o' and 65 <= m <= 224:
  output = run_olayer(x, layer, n, state['olayer'])
 elif layer in ('pooler', 'unpooler') and 9 <= m <= 16:
  output = run_cps(x, layer, n, state['cps'])
 if output is None and 9 <= m <= 32:
  output = run_m32(x, layer, n, state['m32'])
 if output is None and m <= 32:
  output = run_small_m(x, layer, n, state['sm'])
 if output is None and 64 < m <= 256:
  output = run_wide2(x, layer, n, state['wide2'])
 return candidate_sglang(x, layer, n, state) if output is None else output

def candidate_base(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'router' and x.shape[0] <= 16:
  output = run_router(x, layer, n, state['router'])
  return candidate_sglang(x, layer, n, state) if output is None else output
 return candidate_v8(data, state)

def candidate_v9(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 m = x.shape[0]
 if layer == 'o' and m == 32 and (n == 8192) and (x.shape[1] == 1024) and (x.dtype == torch.bfloat16) and x.is_contiguous() and (x.data_ptr() % 16 == 0):
  output = run_olayer_candidate(x, state['olayer'], OLAYER_CONFIGS.index((32, 128, 8, 1, 1, 1)))
  if output is not None:
   return output
 if layer == 'pooler' and m == 32:
  output = run_cps(x, layer, n, state['cps'])
  if output is not None:
   return output
 if layer == 'unpooler' and m == 32:
  output = tt_mw_run_m32w(x, layer, n, state['new_mw'])
  if output is not None:
   return output
 if layer == 'pooler' and m == 128:
  output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
  if output is not None:
   return output
 if layer == 'unpooler' and m == 128:
  output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
  if output is not None:
   return output
 if layer == 'pooler' and m == 256:
  output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
  if output is not None:
   return output
 if layer == 'qkv_gate' and m == 256:
  output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
  if output is not None:
   return output
 return candidate_base(data, state)
'H200 small-m table refinements. Unclaimed shapes return None.'
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = _zsrc('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')
tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v10_sm2_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v10_sm2_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_sm2_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v10_sm2_SMALLM2_ESC_CAP = 32

def tt_v10_sm2_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v10_sm2_smallm2_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
  cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v10_sm2_smallm2_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_sm2_smallm2_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_sm2_smallm2_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_sm2_smallm2_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_sm2_smallm2_c12_bytes(v):
 return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v10_sm2_smallm2_c12_unpack(data, base, esc):
 return b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc)

def tt_v10_sm2_smallm2_table(base):
 return b2q_tt_v10_m32x_m32x_table(base)

def tt_v10_sm2_smallm2_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [tt_v10_sm2_smallm2_units(hi), tt_v10_sm2_smallm2_units(lo)]
 else:
  mats = [tt_v10_sm2_smallm2_units(w)]
 if fmt in (0, 2):
  data = torch.cat([tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = tt_v10_sm2_smallm2_c12_bytes(v)
  if not torch.equal(tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(tt_v10_sm2_smallm2_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > tt_v10_sm2_SMALLM2_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = tt_v10_sm2_smallm2_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 if sp['fmt'] == 4:
  return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_sm2_smallm2_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v10_sm2_smallm2_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v10_sm2_smallm2_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def tt_v10_sm2_smallm2_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
tt_v10_sm2_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def tt_v10_sm2_prepare_smallm2(weights) -> dict:
 st = {'plans': {}}
 entries = []
 for ((layer, m), (inst, cfg)) in tt_v10_sm2_SMALLM2_ROUTES.items():
  w = weights.get(layer)
  if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
   continue
  if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
   continue
  entries.append(((layer, 'm', m), inst, cfg))
 if not entries:
  return st
 instances = list(dict.fromkeys((v for (_, inst, _) in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
 ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
 st['ext'] = ext
 ids = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, inst, cfg) in entries:
  layer = key[0]
  w = weights[layer]
  plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
  if plan is None:
   plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][layer, key[2]] = plan
 return st

def tt_v10_sm2_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
 if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
  return None
 plan = st['plans'].get((layer, x.shape[0]))
 if plan is None or n != plan['rows'] or x.shape[1] != tt_v10_sm2_SMALLM2_LAYER_K[layer]:
  return None
 if x.device != plan['wp'].device or x.data_ptr() % 16:
  return None
 out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
 rc = tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
 return out if rc == 0 else None
tt_v10_sm2_SMALLM2_ROUTES = {key: value for (key, value) in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}
'Hopper BF16 projections tuned with cold-L2 H200 graph replays.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_tune_CUDA = _zsrc('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')
tt_v10_tune_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,int,int,int,uint64_t,int);\n'

def tt_v10_tune_source(configs: list) -> str:
 ns = sorted({c[0] for c in configs})
 functions = []
 for n in ns:
  regs = ','.join((f'%{i}' for i in range(n // 2)))
  outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
  functions.append(f'D void mma{n}(float* a,uint64_t w,uint64_t x) {{asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;":{outs}:"l"(w),"l"(x));}}')
 cases = '\n'.join((f"case {i}:return launch_t<{','.join(map(str, c))}>(x,w,y,m,n,k,stream,query);" for (i, c) in enumerate(configs)))
 return tt_v10_tune_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CALL@@', '\n'.join((f'if constexpr(BN=={n}) mma{n}(a,aw,ax);' for n in ns))).replace('@@CASES@@', cases)

def tt_v10_tune_build(configs: list):
 for c in configs:
  (bn, wg, st, pack, split, push, pw, px, hint, order, sw, prod) = c
  assert bn in (128, 192, 256) and wg in (1, 2) and (st >= 2) and (split in (1, 2, 3, 4))
  assert st * ((64 * wg + bn) * 128 + 16) + 16 <= 227328
  assert prod in (0, 1) and (prod or wg == 1)
  assert pack in (0, 1) and push in (0, 1) and (pw in range(4)) and (px in range(4))
  assert hint in range(4) and order in (0, 1) and (sw in (0, 5))
  if push and split > 1:
   assert split * ((bn // 8 + split - 1) // split) * (wg * 128) * 16 <= st * (64 * wg + bn) * 128
 src = tt_v10_tune_source(configs)
 name = 'm256tune_' + hashlib.sha256(src.encode()).hexdigest()[:16]
 return load_inline(name, tt_v10_tune_CPP, cuda_sources=src, functions=['launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=True, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def tt_v10_tune_pack_weight(w: torch.Tensor, rows: int) -> torch.Tensor:
 (n, k) = w.shape
 col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
 return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()

def tt_v10_tune_run_candidate(x: torch.Tensor, w: torch.Tensor, ext, index: int, n: int) -> torch.Tensor:
 out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
 rc = ext.launch(index, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], n, x.shape[1], torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f'm256tune launch failed {rc}')
 return out
tt_v10_tune_CONFIGS = []
tt_v10_tune_ROUTES = {}

def tt_v10_tune_prepare_m256tune(weights: dict) -> dict:
 state = {'routes': {}}
 if not tt_v10_tune_CONFIGS:
  return state
 device = next(iter(weights.values())).device
 if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
  return state
 state.update(ext=tt_v10_tune_build(tt_v10_tune_CONFIGS), weights={}, routes=tt_v10_tune_ROUTES)
 for (layer, index) in tt_v10_tune_ROUTES.items():
  w = weights[layer].contiguous()
  cfg = tt_v10_tune_CONFIGS[index]
  state['weights'][layer] = tt_v10_tune_pack_weight(w, 64 * cfg[1]) if cfg[3] else w
 return state

def tt_v10_tune_run_m256tune(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 index = st['routes'].get(layer)
 if index is None or x.ndim != 2 or (not 129 <= x.shape[0] <= 256):
  return None
 if x.dtype != torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16:
  return None
 w = st['weights'][layer]
 if x.device != w.device or tuple(w.shape) != (n, x.shape[1]) or w.dtype != torch.bfloat16:
  return None
 return tt_v10_tune_run_candidate(x, w, st['ext'], index, n)
tt_v10_tune_RL_CUDA = _zsrc('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')
tt_v10_tune_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def tt_v10_tune_rl_source(configs) -> str:
 ns = sorted({c[0] for c in configs})
 functions = []
 for n in ns:
  regs = ','.join((f'%{i}' for i in range(n // 2)))
  outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
  functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
 cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for (i, c) in enumerate(configs)))
 calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
 return tt_v10_tune_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def tt_v10_tune_build_rl(configs):
 for cfg in configs:
  (bn, bk, st, pipe, pack, hint, *groups) = cfg
  tw = groups[0] if groups else 1
  if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
   raise ValueError(f'unsupported rl configuration: {cfg}')
 tt_v10_tune_source = tt_v10_tune_rl_source(configs)
 name = 'rl_' + hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
 return load_inline(name, tt_v10_tune_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)
'Lossless BF16 projection refinements; only confirmed exact shapes are claimed.'
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_warps_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_warps_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_warps_W0_DC2_ESC_CAP = 32

def tt_v10_warps_wp0_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_warps_wp0_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_warps_wp0_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_warps_wp0_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_v10_warps_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_warps_wp0_c12_bytes(v, decode=0):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = tt_v10_warps_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def tt_v10_warps_wp0_c12_unpack(data, base, esc, decode=0):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def tt_v10_warps_wp0_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_warps_wp0_pack_format(w, fmt, layout=0, decode=2):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [tt_v10_warps_wp0_units(hi), tt_v10_warps_wp0_units(lo)]
 else:
  mats = [tt_v10_warps_wp0_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([tt_v10_warps_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = tt_v10_warps_wp0_c12_bytes(v, decode)
  if not torch.equal(tt_v10_warps_wp0_c12_unpack(data, base, esc, decode), v):
   return None
  if decode in (1, 2, 4) and (not 1 <= base <= 119):
   return None
  parts.append(data)
  if decode in (1, 2, 4):
   scale = 127 + 2 * (base - 64) << 7
   tabs.append(scale | scale << 16)
  elif decode in (3, 5):
   tabs.append(base << 8 | base << 24)
  else:
   tabs.append(tt_v10_warps_wp0_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_warps_wp0_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, tt_v10_warps_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > tt_v10_warps_W0_DC2_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = tt_v10_warps_wp0_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
tt_v10_warps_W0_CUDA = _zsrc('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')
tt_v10_warps_W0_CPP = '\n\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint wp_bde7122f9b4f_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n\n'
tt_v10_warps_W0_INSTANCES = [(1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (0, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1)]
tt_v10_warps_WARPS_FAMILY_SPECS = {0: {'name': 'wp_bde7122f9b4f', 'launch': 'wp_bde7122f9b4f_launch', 'family': 'dc5', 'decode': 5}}
tt_v10_warps_WARPS_ROUTES = {('qkv_gate', 16): {'family': 0, 'inst': (1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), 'cfg': (1, 4, 8, 4), 'rows': 2304, 'k': 8192}}
tt_v10_warps_WARPS_PACKERS = {0: (tt_v10_warps_wp0_pack_format, tt_v10_warps_wp0_stream)}
tt_v10_warps_WARPS_SOURCES = {0: (tt_v10_warps_W0_CPP, tt_v10_warps_W0_CUDA, tt_v10_warps_W0_INSTANCES)}

def tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0) -> int:
 return ext[sp['family']](inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), x.shape[0], n, x.shape[1], *cfg, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_warps_warps_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8) -> bool:
 (n, k) = exact.shape
 x = torch.zeros((rows_per_launch, k), device=exact.device, dtype=torch.bfloat16)
 out = torch.empty((rows_per_launch, n), device=exact.device, dtype=torch.bfloat16)
 for column in range(0, k, rows_per_launch):
  count = min(rows_per_launch, k - column)
  x.zero_()
  x[torch.arange(count, device=x.device), torch.arange(column, column + count, device=x.device)] = 1
  if tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg):
   return False
  if not torch.equal(out[:count], exact[:, column:column + count].t()):
   return False
 return True

def tt_v10_warps_prepare_warps(weights) -> dict:
 state = {'plans': {}, 'ext': {}}
 entries = []
 for ((layer, m), route) in tt_v10_warps_WARPS_ROUTES.items():
  weight = weights.get(layer)
  if weight is None or weight.dtype != torch.bfloat16 or (not weight.is_cuda):
   continue
  if weight.ndim != 2 or weight.shape[0] < route['rows'] or weight.shape[1] != route['k']:
   continue
  if torch.cuda.get_device_capability(weight.device) != (9, 0):
   continue
  entries.append(((layer, m), route))
 for family in sorted({r['family'] for (_, r) in entries}):
  spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
  (cpp, cuda, instances) = tt_v10_warps_WARPS_SOURCES[family]
  ext = load_inline(spec['name'], cpp, cuda_sources=cuda, functions=[spec['launch']], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
  state['ext'][family] = getattr(ext, spec['launch'])
 packs = {}
 streams = {}
 for (key, route) in entries:
  (layer, m) = key
  family = route['family']
  spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
  weight = weights[layer][:route['rows']].contiguous()
  (n, k) = weight.shape
  (pack_format, stream) = tt_v10_warps_WARPS_PACKERS[family]
  cfg = route['cfg']
  original = route['inst']
  for fmt in (1, 0):
   inst = (fmt, *original[1:])
   pack_key = (family, layer, fmt, n)
   if pack_key not in packs:
    packs[pack_key] = pack_format(weight, fmt, decode=spec['decode']) if spec['decode'] is not None else pack_format(weight, fmt)
   packed = packs[pack_key]
   if packed is None:
    continue
   stream_key = pack_key + (inst[2], *cfg)
   if stream_key not in streams:
    if spec['family'].startswith('rag_'):
     (rows, columns, size) = packed['data'].shape
     target = cfg[1] * cfg[2] * cfg[3]
     padded = torch.zeros((rows, target, size), device=weight.device, dtype=packed['data'].dtype)
     padded[:, :columns].copy_(packed['data'])
     packed = dict(packed, data=padded)
    streams[stream_key] = stream(packed, inst[2], *cfg)
   sp = streams[stream_key]
   if sp is None:
    continue
   index = tt_v10_warps_WARPS_SOURCES[family][2].index(inst)
   plan = dict(sp, family=family, inst_id=index, cfg=cfg, rows=n, k=k, inst=inst)
   x = torch.empty((m, k), device=weight.device, dtype=torch.bfloat16)
   out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
   attr = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 2)
   occupancy = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 1)
   required = n // (16 * inst[2] * cfg[0]) if cfg[3] > 1 else (n // (16 * inst[2] * cfg[0]) + 131) // 132
   if attr < 0 or attr % 1000 or occupancy < required:
    continue
   if not tt_v10_warps_warps_verify_onehot(state['ext'], index, plan, weight, cfg, inst[1] * 8):
    continue
   state['plans'][key] = plan
   break
 return state

def tt_v10_warps_run_warps(x, layer, n, st) -> torch.Tensor | None:
 if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
  return None
 plan = st['plans'].get((layer, x.shape[0]))
 if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
  return None
 if x.device != plan['wp'].device or x.data_ptr() % 16:
  return None
 out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
 rc = tt_v10_warps_warps_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
 return out if rc == 0 else None
'Lossless m32 pooler and qkv projections, with an optional unpooler route.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_m32x_M32X_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_m32x_M32X_ESC_CAP = 32
tt_v10_m32x__CFG = (1, 8, 16, 1)
tt_v10_m32x__CUDA = _zsrc('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')
tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x_m32x_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_m32x_m32x_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_m32x_m32x_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_m32x_m32x_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_m32x_m32x_c12_bytes(v, decode=0):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def tt_v10_m32x_m32x_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [tt_v10_m32x_m32x_units(hi), tt_v10_m32x_m32x_units(lo)]
 else:
  mats = [tt_v10_m32x_m32x_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = tt_v10_m32x_m32x_c12_bytes(v, decode)
  if not torch.equal(tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
   return None
  if decode in (1, 2, 4) and (not 1 <= base <= 119):
   return None
  parts.append(data)
  if decode in (1, 2, 4):
   scale = 127 + 2 * (base - 64) << 7
   tabs.append(scale | scale << 16)
  elif decode in (3, 5):
   tabs.append(base << 8 | base << 24)
  else:
   tabs.append(tt_v10_m32x_m32x_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > tt_v10_m32x_M32X_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = tt_v10_m32x_m32x_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def tt_v10_m32x__build():
 name = 'm32x_pooler_' + hashlib.sha256(tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
 return load_inline(name, tt_v10_m32x__CPP, cuda_sources=tt_v10_m32x__CUDA, functions=['m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__make_plan(ext, weights, fmt):
 pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
 if pack is None:
  return None
 stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__CFG)
 if stream is None:
  return None
 return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=tt_v10_m32x__CFG)

def tt_v10_m32x__run(x, plan):
 out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
 rc = tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, tt_v10_m32x__CFG)
 if rc:
  raise RuntimeError(f'm32x launch failed: {rc}')
 return out
tt_v10_m32x__W_CUDA = _zsrc('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')
tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint m32x_w_info(int);\n'

def tt_v10_m32x__w_c12_bytes(v):
 return tt_v10_m32x_m32x_c12_bytes(v, 5)

def tt_v10_m32x__w_c12_unpack(data, base, esc):
 return tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def tt_v10_m32x__w_table(base):
 return base << 8 | base << 24

def tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
 (n, k) = w.shape
 total = wk * split
 rw = r // 16
 if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
  return None
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
 esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
 (table, max_esc, escapes) = (0, 0, 0)
 if compressed:
  (data, b, entries) = tt_v10_m32x__w_c12_bytes(v)
  if not torch.equal(tt_v10_m32x__w_c12_unpack(data, b, entries), v):
   return None
  table = tt_v10_m32x__w_table(b) & 18446744073709551615
  (er, ec, el, ev, xor7) = entries
  escapes = er.numel()
  if escapes:
   boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
   rank = torch.bucketize(ec, boundaries, right=True)
   seq = ec - k // 64 * rank // total
   group = (er // rw * total + rank) * rw + er % rw
   max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
   if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
    return None
   order = torch.argsort(group * (1 << 14) + seq)
   gs = group[order]
   ss = seq[order]
   pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
   entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
   esc[gs, pos] = tt_v10_m32x_m32x_to_i32(entry)
 else:
  data = tt_v10_m32x_m32x_bf16_bytes(v)
 payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
 return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
 return ext.m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
 (n, k) = w.shape
 eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
 out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
 for start in range(0, k, m):
  if tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
   return False
 return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))

def tt_v10_m32x__w_build():
 name = 'm32x_unpooler_' + hashlib.sha256(tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
 return load_inline(name, tt_v10_m32x__W_CPP, cuda_sources=tt_v10_m32x__W_CUDA, functions=['m32x_w_launch', 'm32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__w_make_plan(ext, weights, fmt):
 pack = tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
 if pack is None:
  return None
 return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def tt_v10_m32x__w_run(x, plan):
 out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
 rc = tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
 if rc:
  raise RuntimeError(f'm32x unpooler launch failed: {rc}')
 return out

def tt_v10_m32x__w_verify_random(plan, weights):
 for m in (9, 16, 17, 24, 32):
  x = torch.randn((m, 4096), device=weights.device, dtype=torch.bfloat16)
  x[-1].zero_()
  x[-1, m * 131 % 4096] = 16
  ref = (x.double() @ weights.double().T).to(torch.bfloat16).double()
  out = tt_v10_m32x__w_run(x, plan).double()
  diff = out - ref
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
   return False
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
   return False
 return True
tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
tt_v10_m32x__Q_CUDA = _zsrc('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')
tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def tt_v10_m32x__q_build():
 name = 'm32x_qkv_' + hashlib.sha256(tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
 return load_inline(name, tt_v10_m32x__Q_CPP, cuda_sources=tt_v10_m32x__Q_CUDA, functions=['m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__q_make_plan(ext, weights, fmt):
 pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
 if pack is None:
  return None
 stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__Q_CFG)
 if stream is None:
  return None
 plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=tt_v10_m32x__Q_CFG)
 x = weights[:32]
 out = torch.empty((32, 2304), dtype=torch.bfloat16, device=weights.device)
 capacity = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 1)
 attr = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 2)
 if capacity < 144 or attr < 0 or attr % 1000:
  return None
 return plan

def tt_v10_m32x__q_run(x, plan):
 out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
 rc = tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG)
 if rc:
  raise RuntimeError(f'm32x qkv launch failed: {rc}')
 return out

def tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
 state = {'plans': {}, 'device': None}
 configs = [('pooler', (4096, 8192), tt_v10_m32x__build, tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), tt_v10_m32x__q_build, tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), tt_v10_m32x__w_build, tt_v10_m32x__w_make_plan)]
 for (layer, shape, build, make_plan) in configs:
  if layer == 'unpooler' and (not include_unpooler):
   continue
  w = weights.get(layer)
  if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
   continue
  if torch.cuda.get_device_capability(w.device) != (9, 0):
   continue
  if state['device'] is not None and state['device'] != w.device:
   continue
  state['device'] = w.device
  ext = build()
  for fmt in (1, 0):
   plan = make_plan(ext, w, fmt)
   if plan is None:
    continue
   if layer == 'pooler':
    ok = tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, 32) and tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
   elif layer == 'qkv_gate':
    ok = tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, 32) and tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
   else:
    ok = tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and tt_v10_m32x__w_verify_random(plan, w)
   if ok:
    state['plans'][layer] = plan
    break
 return state

def tt_v10_m32x_run_m32x(x, layer, n, st) -> torch.Tensor | None:
 expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
 if expected is None or x.ndim != 2 or x.shape[0] != 32:
  return None
 if (n, x.shape[1]) != expected or x.dtype != torch.bfloat16 or (not x.is_cuda):
  return None
 if x.device != st.get('device') or not x.is_contiguous():
  return None
 plan = st.get('plans', {}).get(layer)
 if plan is None:
  return None
 if layer == 'pooler':
  return tt_v10_m32x__run(x, plan)
 if layer == 'qkv_gate':
  return tt_v10_m32x__q_run(x, plan)
 return tt_v10_m32x__w_run(x, plan)
V10_ROUTES = [{'m': 8, 'layer': 'pooler', 'name': 'smallm2', 'module': 'v10_sm2'}, {'m': 16, 'layer': 'qkv_gate', 'name': 'warps', 'module': 'v10_warps'}, {'m': 256, 'layer': 'o', 'name': 'm256tune', 'module': 'v10_tune', 'config': [256, 64, 5, 7, 1, 1, 1]}, {'m': 32, 'layer': 'qkv_gate', 'name': 'm32x', 'module': 'v10_m32x'}, {'m': 32, 'layer': 'pooler', 'name': 'm32x', 'module': 'v10_m32x'}]
V10_SELECTED_ROUTES = [r for r in V9_SELECTED_ROUTES if (r['m'], r['layer']) not in {(v['m'], v['layer']) for v in V10_ROUTES}] + V10_ROUTES

def prepare_v10_tune(weights: _AchitaDenseWeights) -> dict:
 weight = weights.get('o')
 if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16 or (tuple(weight.shape) != (8192, 1024)):
  return {}
 if torch.cuda.get_device_capability(weight.device) != (9, 0):
  return {}
 return {'ext': tt_v10_tune_build_rl([(256, 64, 5, 7, 1, 1, 1)]), 'weight': tt_v10_tune_pack_weight(weight.contiguous(), 64)}

def run_v10_tune(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor | None:
 if layer != 'o' or n != 8192 or (not state) or (x.ndim != 2) or (tuple(x.shape) != (256, 1024)):
  return None
 if x.dtype != torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16 or (x.device != state['weight'].device):
  return None
 output = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
 rc = state['ext'].rl_launch(0, x.data_ptr(), state['weight'].data_ptr(), output.data_ptr(), 256, torch.cuda.current_stream(x.device).cuda_stream, 0)
 if rc:
  raise RuntimeError(f'm256tune O launch failed: {rc}')
 return output

def prepare_v10(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v9(weights)
 state['v10_sm2'] = tt_v10_sm2_prepare_smallm2(weights)
 state['v10_warps'] = tt_v10_warps_prepare_warps(weights)
 state['v10_tune'] = prepare_v10_tune(weights)
 state['v10_m32x'] = tt_v10_m32x_prepare_m32x(weights, include_unpooler=False)
 for route in V10_ROUTES:
  (m, layer) = (route['m'], route['layer'])
  weight = weights[layer]
  data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': layer}}
  candidate_v10(data, state)
 torch.cuda.synchronize(weights['qkv_gate'].device)
 return state

def candidate_v10(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 m = x.shape[0]
 output = None
 if layer == 'pooler' and m == 8:
  output = tt_v10_sm2_run_smallm2(x, layer, n, state['v10_sm2'])
 elif layer == 'qkv_gate' and m == 16:
  output = tt_v10_warps_run_warps(x, layer, n, state['v10_warps'])
 elif layer == 'o' and m == 256:
  output = run_v10_tune(x, layer, n, state['v10_tune'])
 elif layer in ('qkv_gate', 'pooler') and m == 32:
  output = tt_v10_m32x_run_m32x(x, layer, n, state['v10_m32x'])
 return candidate_v9(data, state) if output is None else output
V11_ROUTES = [{'m': 32, 'layer': 'unpooler', 'name': 'm32x_unpooler', 'module': 'v11_m32x'}]
V11_SELECTED_ROUTES = [r for r in V10_SELECTED_ROUTES if (r['m'], r['layer']) != (32, 'unpooler')] + V11_ROUTES

def prepare_v11(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v10(weights)
 state['v11_m32x'] = tt_v10_m32x_prepare_m32x({'unpooler': weights['unpooler']}, include_unpooler=True)
 weight = weights['unpooler']
 data = {'x': torch.zeros((32, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': 32, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': 'unpooler'}}
 candidate_v11(data, state)
 torch.cuda.synchronize(weight.device)
 return state

def candidate_v11(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'unpooler' and x.shape[0] == 32:
  output = tt_v10_m32x_run_m32x(x, layer, n, state['v11_m32x'])
  if output is not None:
   return output
 return candidate_v10(data, state)
'Lossless BF16 small-token projection routes validated on H200.'
import torch
from torch.utils.cpp_extension import load_inline
tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE = _zsrc('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')
tt_v12_smtrace1_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v12_smtrace1_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v12_smtrace1_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v12_smtrace1_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v12_smtrace1_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v12_smtrace1_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v12_smtrace1_SMTRACE_ESC_CAP = 32

def tt_v12_smtrace1_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v12_smtrace1_smtrace_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
  cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v12_smtrace1_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, tt_v12_smtrace1_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v12_smtrace1_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v12_smtrace1_smtrace_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def tt_v12_smtrace1__smtrace_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v12_smtrace1__smtrace_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v12_smtrace1_smtrace_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_v12_smtrace1__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_v12_smtrace1_smtrace_c12_bytes(v):
 return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v12_smtrace1_smtrace_c12_unpack(data, base, esc):
 return b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc)

def tt_v12_smtrace1_smtrace_table(base):
 return b2q_tt_v10_m32x_m32x_table(base)

def tt_v12_smtrace1_smtrace_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [tt_v12_smtrace1_smtrace_units(hi), tt_v12_smtrace1_smtrace_units(lo)]
 else:
  mats = [tt_v12_smtrace1_smtrace_units(w)]
 if fmt in (0, 2):
  data = torch.cat([tt_v12_smtrace1_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = tt_v12_smtrace1_smtrace_c12_bytes(v)
  if not torch.equal(tt_v12_smtrace1_smtrace_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(tt_v12_smtrace1_smtrace_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def tt_v12_smtrace1_smtrace_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, tt_v12_smtrace1_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > tt_v12_smtrace1_SMTRACE_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = tt_v12_smtrace1__smtrace_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
tt_v12_smtrace1_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
tt_v12_smtrace1_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
tt_v12_smtrace1_SMTRACE_MAX_M = 32
tt_v12_smtrace1_SMTRACE_TABLE_M.update({('router', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1]})
tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def tt_v12_smtrace1_smtrace_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v12_smtrace1_smtrace_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v12_smtrace1_smtrace_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v12_smtrace1_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def tt_v12_smtrace1_smtrace_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = tt_v12_smtrace1_smtrace_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = tt_v12_smtrace1_smtrace_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = tt_v12_smtrace1_smtrace_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v12_smtrace1_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v12_smtrace1_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def tt_v12_smtrace1_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
 table_m = dict(tt_v12_smtrace1_SMTRACE_TABLE_M if table_m is None else table_m)
 table_mt = dict(tt_v12_smtrace1_SMTRACE_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = tt_v12_smtrace1_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, tt_v12_smtrace1_SMTRACE_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if tt_v12_smtrace1_smtrace_bf16_twin(inst):
   instances.append(tt_v12_smtrace1_smtrace_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = tt_v12_smtrace1_smtrace_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and tt_v12_smtrace1_smtrace_bf16_twin(inst):
   plan = tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v12_smtrace1_smtrace_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 return st

def tt_v12_smtrace1_smtrace_run_core(x, layer, n, st):
 m = x.shape[0]
 if not 1 <= m <= tt_v12_smtrace1_SMTRACE_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v12_smtrace1_smtrace_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v12_smtrace1_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
  return None
 x = x.contiguous()
 if x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 if tt_v12_smtrace1_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out
tt_v12_smtrace1_SMTRACE_ROUTES = {('unpooler', 1): ((1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1, 8, 2, 4), (1, 8, 2, 4)), ('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def tt_v12_smtrace1_prepare_smtrace(weights):
 usable = {}
 for ((layer, m), (inst, cfg)) in tt_v12_smtrace1_SMTRACE_ROUTES.items():
  w = weights.get(layer)
  if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
   usable[layer] = w
 routes = {key: value for (key, value) in tt_v12_smtrace1_SMTRACE_ROUTES.items() if key[0] in usable}
 return tt_v12_smtrace1_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_3c2c50fe46c8')

def tt_v12_smtrace1_run_smtrace(x, layer, n, st):
 if (layer, x.shape[0]) not in tt_v12_smtrace1_SMTRACE_ROUTES:
  return None
 return tt_v12_smtrace1_smtrace_run_core(x, layer, n, st)
V12_SM2_O1 = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1))}

def prepare_v12_o1(weights) -> dict:
 st = {'plans': {}}
 entries = []
 for ((layer, m), (inst, cfg)) in V12_SM2_O1.items():
  w = weights.get(layer)
  if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
   continue
  if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
   continue
  entries.append(((layer, 'm', m), inst, cfg))
 if not entries:
  return st
 instances = list(dict.fromkeys((v for (_, inst, _) in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
 ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_o1_v12')
 st['ext'] = ext
 ids = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, inst, cfg) in entries:
  layer = key[0]
  w = weights[layer]
  plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
  if plan is None:
   plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][layer, key[2]] = plan
 return st
V12_ROUTES = [{'m': 1, 'layer': 'unpooler', 'name': 'smtrace1', 'module': 'v12_smtrace1'}, {'m': 1, 'layer': 'o', 'name': 'smallm2_o1', 'module': 'v12_o1'}]
V12_SELECTED_ROUTES = [r for r in V11_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in V12_ROUTES}] + V12_ROUTES

def prepare_v12(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v11(weights)
 state['v12_smtrace1'] = tt_v12_smtrace1_prepare_smtrace(weights)
 state['v12_o1'] = prepare_v12_o1(weights)
 for route in V12_ROUTES:
  (m, layer) = (route['m'], route['layer'])
  weight = weights[layer]
  n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
  data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
  candidate_v12(data, state)
 torch.cuda.synchronize()
 return state

def candidate_v12(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'unpooler' and x.shape[0] == 1:
  output = tt_v12_smtrace1_run_smtrace(x, layer, n, state['v12_smtrace1'])
  if output is not None:
   return output
 if layer == 'o' and x.shape[0] == 1:
  output = tt_v10_sm2_run_smallm2(x, layer, n, state['v12_o1'])
  if output is not None:
   return output
 return candidate_v11(data, state)
'Lossless BF16 small-token projection routes validated on H200.'
import torch
from torch.utils.cpp_extension import load_inline
tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE = _zsrc('eNrNPG1X27jS3/kVKnvg2mDSOEmBEqBLabu3h0L3ENjuc3hYH8dxiItjp7bz1i7//c6MJFt+C3T37t1tD4ksj0aj0bxpJGXtBy9w/OnAZYfOdGBb0TRIvLHbGB2vFd70h+ZuvjpOBl6Q5Ots/y6MvGQ0ViuT5cS1ksj2kjiHNgwnbmQn3sy17qJwOokR19oPA3foBS7rnb9hljVwZ57jWhYUh2HkuF7gw0vLWlvD99B/h8VjwA0w/mCuOWEQJ1S9xSY6+7bGOMysCyU7HrNZ6EOPvqut+4PGnR/2bb8ROI0P5sFBEFq274eOnbiNWacRt1vs20bT2DCNjZax0X4w2M1G57a7Dphy/w7Y+lG0rs0aC92QxWVW/JoV5zoC++vaRNeRoMhNplGAxD3w8UwLA1qIAc1Cb5AOaPqkEcEIpnUj+G+T3G5ZiUK1K6iWr3KUE+xjxCPlQDiQa6rkfjctk2icaGmtbWQQfaUcF6gbCOq0dWzf6AtigI0M+Mg22hlJAyIJCjayDL774jvOkThIScSZTMkbj21t6Id2wrRNR7/p3BqsyLkczU31weRUE52AqBEvgXG2790F7qAxNneD/XvQ2CicN5zQbwzbrQapMH0My5LxbaNjbLwwNnaNjT163Dc2XlJBAUuFHwa9PVzXnJvmLQ6Yl02l3FLK7Vs9a4a8Eq2oaGbFVlZsy2K/KQsmchS4+Pw522dz17sbJTELhywEWwEjZXHiTg6YH85Zf5m4Meg9+9Bk2sz2p/DU3GnrzA4G7IOZ1nV29gB5Z6fvJcD3AW9z2ijP08B1Min6oM7CB1N5OFXKVyrUlQIFU1oQtyhMwORkQhePho2oMY/sSV704K+TSZ5oJeXvlMtbXnyQG9YIOHWknbJN1lzs7/H/U337qlkBj7CigcBf3Qxn8CinZMAV2ZnRXLwwO82pziHNlZB77d2WhGwVIU0jJaiAtL0aVGIFccm5kCCU/iM/wYE7cyMts7IF2wRAk+663mVc/C4QmiX2vRsYrD9N2CRZ2DFz7CAI4SEKZy7zkgPmBoOYJSOX9e3Yc1gffMs9i0OYFRteeHE8ddl85EbwCT4zcYMK0QujgRtZE6A70eCPjY2yM6CpS8omdRzOUku6seFg95n8JFxgvCHTtARmORVwfayzoyOY9b0h/zfVC3ya6PUWNx67Yws6LTkuaQqzfizLmSW2decGbuQ5VhJa8ciO3AHiZ3X4Izu414oK9PjQp2AdIstJbGyvKFHOSkf1wxrbE7vakSDCGv+h0gPtG3x4BweCnIxAciwrnAr18RR/0rcjyws8xedBjUKsE0J8l/oNlT6AizygCVtnhCZ2o7/bATfcvEUykcQDMsZ2JEgTGIFemPgwWq7rK0hzFxPXqSUuWaykDL9mboPjsJJFiUrLWEUoYn+UyqEbwCdxsGwF6GVD0kNQjcj1XTt2G2JKqV+1F5DjUieOb/GxVPQhBytFRAwaurEX4NaFe39qN3O7ciDFThBsFeoy7jixbIw3OpqiBwajWAZCU1U9cAYqpjWGHiliKSqF5C/UhOOJ7yawdFgcHJBPx6AWQxc+zd+ygAz+OhTdvqiKzw+4DqEgDNOgd5jGvMM05B3yiFcoHVFeJTL9MPTzgp1Ey5xUFwxCeF+2CN/+P2Awr3esMYHxs0kXnlNZB3w0eY2JDZ5h2bCdL1MvSsWsJPoTHiobrAloSgxYZ7HrT1JzA1yDT+zwITU54b20OeVRp1YnvGfPwDHkrORuR7GSd7h2VKwzvaxzTEA2t8487MemkbCBfuqeRM+JjPxeg7kZwMiZzfpRCA6YQc9BbDuJFwbcvLF4FM5jNp1ApwDGvT8b2lMfF4bAPnuAYSP6p+nYC+7YNID1phePYBYGNjC0LO44FRYwpmKO0X+WRAHf6oL2fJTFOdJkR0WeIdh8BNxh2rNqfDSraneS2zuI8Ji1mvzf1Pd1CHUwZtFqzJw9GHT4wqOzic6M0ycr+mQxGgu2fcT6jUUXykteXmL5Ky9/xfKcl+eplRAWILUTo6FvLcKoo+W7AAuBwUzIeSgmeQwBlcUhNCttaqGV0JqLoYxGIABqLAxoa7BHoJYEVdQG/u+Rpl+f1MEcoWp4HCdh5MI4gpnVp0GZu1uop3y9N0Meb01ADiw+5JYEsqIAFrqVFp2jJLhKVDOahsQFswmhOztEFr87vzqGpVE0dRLWG78bgyZlAPk3h81jwBRD2A/BKs0X+NqIJuriA6DvGOziBL5N6GYFFnM1lvbTsLRWY9kXWFqrsbRXY9nNYangXOVSMEwTHJSI0Tbn+g3vDpscHFx8uFUcoLZ5Unh9wraYuXu7YrFYJbLp+2sV+NpMzVA2Pg06ohCe/f47E+UWwv0wiey7sQ0WLwp9n3oZhgCPw/UZmnb4OmT5wXTZ9rbPe8F/JzcdoN9n26x5C02ycH5+49+CtSiBmRVgyzJYqwLsaxmsXQE252AP8PnAXD92HxnqiA91VBzqCQ51lA1VTSZOoc0c+wc6Rt0cQBbAol0fsVcwReDGcFITU9aYWGPKhrnEAqyl0aDR9xK/W/ScUHuDnSAPd7Fb5Llei+KrQDEXKJY1KDo1KExBhZlR8bUGxX49iq8ChaRiXoPCbOnKrAnv/jZ27IkL62ZwegcQtHxhh4fM3Ge/M98OXHpowwNETR6GQViBL0e2P8TyHpT3KJvz68dL9PG49saMAGWE/hUz0I4wAOwMYOJGQd8vLivUfWInzkjLK/PFpaq7LpeXgihgYAdy6rLjY+AWLK/b5rQi37KIOBQmWfbewUIbRqHtA3s03nQPm5pQjxYDGFaS6lSmIy7TwBMYCMpxlAUMRIyqNRC7nNxEt+y3I6BAhlYng88wYrQEwOsYRgBxki3rpF/3wyQ22NxLRsy1nRGfF4ijKJBKRuCOEARYHU98L9k5YxDAJp7t57jt+HYcs7djL6ngOMS1U8fNxQtbLI4cHjFcvOffP4uv3tUlLyXegBeCxCDUzIUPPjcpk7zBAtgEoF0qAqve8xJEMXJdzNJVDICqIUmzMYTAWf0QEpyin/A5mADin3EKJno+4oKoBwZyo8XeV1jU6BOYVRwATDfQcCuwIdkaPEN8QTVSN8CMB/ss4TEvxIkwDZdXIBU7UTjnFWwCKjG3I4gMelcgIiDjmNAc+pgqpZcStMHevD/5iWkQQOK8hYG/1A+YyYKQnZ+fGNhdCx8WPFVlsA4+0cxgnG2A1vEcFrY0UKlhbpMlu3ejwPUNBgsNjWQERB/mHIgIp8lkCsKBEQzgg9g/BbAB+9lr7DPGPSKgdQdTdJSRFTkzECtUZTeAOTZb+6JtR0eqcpgbZR/OpUJ+X4rvnnx++4YXkCFCxK7OT37lxdNz8X0hvs/E96dLWZA1Z9ei0Dtesyy+pJFZRsvy7WngjKw+rl5iy9Koj0w67tzxTFsruhwLdAFY4jkJIJpPZJheCClzUAuZHczVIouMtbLDKoC5sVCysWVHd4IZWfEsK366VMpK/dk1L68ObFZHMikH2BjU6fQcnCd8HHCiujmACwS4QIALABCknmHlGVaeQeVZudWnS4SAz1f0eZCO5hM1/URtP2FjObSza8J5TUivEet1Dm0+qMxHTyLCLMQZuZaUTjg1WwAmgjVTCdzaFcD/fv/ho4A+xui1DHHxERQZQEjPN3HJf/Hx1+y5hc8g+1kNBPeXH7NHiK/fnv989X8Kit2Kbi4/Wr0rQJSBtVuIyDr9+Eap3MWlw8cCJChyBcafr3v/Rp9IxMHYkBOi3AZnuMmeccKxBATjFxGKqKYxWrOP0+QKrfxy4gb2GLxTgomSMBh4aLts/xC5JxZPRl6bYHKwGeJCLFuoOYAqcmFy3WgC61TLsePkkF4ea6hXZKPdBbwOaLWIeRnaaqB8Gqg6+G2wxTEl15gD7xmmyG9usV1lAzCJWWIFsw7dUnxPI5buKkszpMFvzpKgv0N+jNCjvx8sMMpDL8E9IUYZLwzux3kFxioGuMOclmyRSmyhZT/gRvmNN5bRvqKRn6AVgaNhnGNQQl1tMLSY83v5/Byei2172GEP++tBH3eRN8AulkWwn35VtP4506g38IQUISkNS7T9BO2g8RZ0tAWEFl/fzVHsaGjIpCUAAfQ2S2sWOrWDKhxCsXlEwX/WfoE8QNc+j3QisNjAaZb6Aw4D/D3Cn10X4a9+wXjECzRcCaE708YYRus4gaberTDvmKRSEk3pzswmSZVoktvHoPhFho2KvKFSZssiAZnfkZELAoo3CQDXn6CeJH1QztqX6MQYMQaMqc6/UmJVTeuxHRwnMh8nuPLVp8uUAlaxHUIbDWYFiJrwl28fqoaikm9RMogrVKwpWL0hhFSDo6McdPVwj2CwR0etV7nR7Jj6FgzyoFSZGx6r3FTBIRLmrcurratftsC8TpVGD4XBVWxGyPVYXi7GSwstQ3PxTvybVkkIui9uknXZQj0PAqEFLmpE4AuqRpZkm8wO9FsKfeZf0FJMim000nPwsthcF+1V0efALMaYZoBEyOY/lVqrzQoh1QJVc6F0Drq6hTEr1JCh3ERftEUukfMC80Dzm97V7c0l/F18uM2Uq8MWM3pzDn8t8cKeJnwbGDq62byl1QMunHgYFdfkapSlxme+1PgMSw0wLbDW+KxXt6jO8KRZnfkNdnt78/n2xr/NTdpcg0nIWBDDeCVbofYz5yTMgK6ysygWzzDqkHLx7VEKA05hABSe06iCKkUqzhZ6MpiyjFa0kgGQtZ9OGNjIlm5ARCnMBRramOZUVSuYKM6N4PammefGQltM9DpYswybS5IUte+huzI9FXMuxJieIi6APKCsaLBwigXSquMBY4OJiCRbxjLbcW4uTrhcogR2brtPzY3JfNhfIIoVE13XRmnl8FYOtOpgI0en4Y1IfGEeHJwHWJcXPNsw9AeZZ0urg0VSadegI3dS5wlVO8cRiC2C8vYAmkGDEnaFoTEzz3KPD8uDYYHbh+9tWC9cPW4DquSk6GMJDjvwUOJVT8W0Hl9f4HgABBYSZ9c664NTu8/BKaNfqcV/TCQes1CFVDjN5W9qLksDEybsFwRdvzHlcZl//Jp/nOf0mTbWkEnAlUPiRKpyvLbSAVerX6r1ajxeyOKfcE8h9gK6f4yphV0JWl9KfhjQBX7iEhzPn+Fi+9rMjSILFMk07D+JjMdsdFFmwAOsFJoccpcjdwH5PuJ2QSBRs13S7EV7uN/Ev2k3h6CCz/ivn9pxNNbCquN+Zj812uqLpXjRKr74Kl60iy/m3aoeOzkwM+vxRfGF7HG3+EL2uFd8UejxYa26/FcIc4UlfGReV0pucbIKMfIUk/85NQfKaZ23zT7rRb7TITY0x7gW2qfja4ig3Atjg7CikmPga+AjolnP8LVF+l7X87sCh0J5jzWuaABfIgz/gRRPquq5+3AxW9zGJc8jjsSd4OKn5K7yeiBOEFTxogj+sEKS/joL/12OP9fyC2/5RTr/ytktmp7z85NHjc+j4U/V8QHph3Dqb7KtNfj+cgtupqKa74TWvQKj81v1QQUmd0CrmrVvu2v1xHFzJmnKnoiUKvGpNKO5BefY1jDkaoqQizsZSZCh9mgUO6xUjvyEYWZOL3UGegZhe7FH4MUf7fahVhUKcfraWl0MVB0dUdpTlUs0KzzXVkghKM7XBpHDGQPKRQkmywS9d7LHFjwOssd2TvkLpvM+ZzfH7HeGW4+Pq7SiYLtCw3J8Kh5oHTT8MDvCin8d+HtB59iUl+LkevXLljjAJ1+u1+vAeq5hW576o4Z0amxbHpjdltcwsODIQnqo9l6c7/ui50RDrBJs1BT4c+BvUJ8+Kc08JbeLU69kC0gAKpwuP9ODeeZtyiFuccHOLVstC6fTsmO+FakRqZsUD8kT2lnyqqjESAcfG0ZQL9323t7L/kvlTHdKgcHMdE8zG2mWXi4h5yvM6LG15T9lrRipy0TaQnnFVGOGL7ZZyeJg9UERsDrdkTqcR5kfIbKmCE9NuWRNTY66fLt8LXPqIsvdLYFE4RxTVrnk8+VrNSXLN01+/z3dWKHcLO54Qff6U3Iz3+Xx/6i/XzV91REjbXCEaPNkxgdNv6J4oEsO7jaV/ECOfWgzUf0wuqzQQUznIFpHzbbno0e5v/VKnuhDfUW91TK50wtyAHKFtB+ycTnKqDIQMsGFjbbYBZqMcG4ogq3XrQ9ULa6zYnxnTWX08+fs3dT3+bGPA3F8QkMHQHsQ0H+gq6ncBjsNxxPb4cnMmGkAY+rFhle/AHiii+m6a6TdUVYUzz9YQ+w1S43ScQh5LKMqJUjJA5Q17IlvYOHGtqjgW1wB7eRB5QZtnxAxogZPV5Q9qjheOeMrum906pQOldLx0Flj/tD9Y4YoL8mr5ThYKcVFGSZDsM0Eo4UUB0+QYZRgKYrfI3rAnYLUKcFUt2JmHSEg3zm5d+nc7hWnFsKAJJvaq1+KU3v1yz9manFKkz8xl3d/5xwWQh7pRrL9eTM/fHl2CzelqjbRxftjDTcm9W7ltF+8z/m+8yuyNt3/8QI1b5hxPDea5CVF9xl54D+qTSPIWUVsq54vk1Ycr7bIsqmUW0q5nZ+v2g3JulkrCmrh8B0O0aCzdriNj990vA6P1qXmOS8wVUtHKQBOPMWjPCQHmRBefCrN6h+jqM6OUL83eLJOHI0v6Vvx5wfSS0j8mTnog6rAkpEXWwJWK6L16Sa2Vq/dtEDDHBOAUshW2EIvNsCTe9ACdGGbiQ3v56xXB+2a4pwAMkmDzrb5lhdgKfRQOBYZ3XMosjnpCUnXrDghWZxjOiu5BcMZ2xN5pIXGRFO/zaemWQqbcktPky89eyK3Uzw4+Rj2wmpOnqQkUVWPU9ZlASqn7WFVKl+1Uk7OTEGAAy682qKhKZTBqUFuLY0yuxXGM3Kd2fcZT4jZrkYu88EUuNG/YhbOg/QQLmYCxYFMzkdxbLJLBzmhl3GYuOz06iRmk2k84ud805t6j1mVVp1VSW8iVvgrmPKrXx5NB0f5oyx0G5fOdDT17hPzh9/pEL5zz6M0zYGI5PDQQFKT+dKSoGb4RXQigNbmUZYETyPpIA2lu9W5SiFLg5j7YZCozAzzEy65s0cozqRV3RVU1RyS/hNOLM8a5eDNFid8Vg1Melm+l5oTldIRKMCoo/AYeBMtyk5Drc4TrkqaK1Q/I6rzK65H/TQrXzYsk1VwijiV6Awdg860cc/oFJ21iLif7q/TgC0nKD0uGatU+G9RxBrVSnWPU1cpOjySk4HcfSr61XqWYkzVrTqm+69GdkUhe5IE1djb77GpTwjFHKNC4p6wqHvUJAmLVGuGnmge/hLD8FDhw/4+pX+6vtfucIgLJuKOJB0XUXkSThIIJfkEKjBo9ndM+fshM9CxdEmBrwfuLDtAg78e9pObvKF72NomvNOzF7wWXp8kSeT1pwlAwPDEK6w8txc9ilXO6UL6z270GmPmj0iYwSS2h+LPQUEY9GXqRktYofLLFxjlmAImBvVcMLw0PnOZiONjWFXzkmigM7A6FJ/zmzW98y4s4CWCyL2Lt8xms7mNvxjm898c+mdcP6mYS3H/hH5Chg6W43USWV5kRdoFkA/ZfRBBmbjkIe+AyPsf8u6HUWcMacQZ3qRpZg9T9SFOQElEbzR74pcjMHuEd4vUa/PueHZI/MU/5CvylPjJeSn4iDxE/iHviG/EM+IX8EpKMBcVNYZFCXwHTEvlMmZDOz0RpsnXquTG2ubQNohQHe0BwvSmjuPGsbQMoDRCK8Tz0G4E0/ElCBPmeECc0A7i2pBuLcAy7pAbxWMNIEnSevD8GoXNYC9fvkyFn7abTs8xqz/Gzk/PdUr3n15g1QVVXYiqM6w6o6ozXSGtlf4gEB75x/2BSwL6dCkafjoTuwZUeybRXRO+a47wWtT2sLJHdb1SJ3U3BujWAWUG6B5J1c4H0gd1h/ySDNAiStD0mIl7VWlnssEF2wBs2YtspEL0dB5A0cl7ftYXCD/LGrTTBr0rfNUWQ96gk3rQPZQPqSxbdGSLsaRwDARSrmM/g3qRcYOuv1DKAddTz4T9RAzPoFYy08zavpTv+WutBx3RfZkedGTu6gXIwlUCKWCl+wTPGT/NK89Sj5lKiiZvEO3rqcUfqoklggdOAgl4jr862/PCxHskQhXIUStITIEkXaLQpQsUwgKK6uatQnN+3SO3NhdXj9IYQe2sh32d8TsFVY0e8p4QHIk1WAY5fyjrFK+4wpw8Zko4VNZPpX/e4VZjKH9kKrUS+T56qqNF7EaZMnC4b5aBPfYc1e8iQkNSoSuG0Ne/C/tFGPwcRond991T7m4R84nvh3N3IO9oPBS1M0bdkZ2nYs1vpEEPH8i9nYbB0LvDX73DdMy3h/zblAJmJzf8+IvvNMRtIYAfeOO2hjeKLl9TZMUpARB520nCYKBlKu8HglkwHcRxmiLxjjs0FC9yBvSEw6FCpr5SSoCy5m2DrmtVEC74BaS4QeyFIszkbWa2L38ciC4/UUKt/v0SrySueP81aw+jsIECHJadpFXgtE5EranOWOpITTUeLESDb6MojCx+2hNs1yuq/Og404kdOEuQvxOKzMR4Y4oIuTht+o5eF2Uc1KChcDGGuPF86ifeJArRIYeRghVnNB7ruXD1gx0nRKeM2IXQ0QlBxbMD9TPoesdM2ZDN3Bldh3670IBs2Vcpk6dceznW5hO9FiR/7eFYW9T9tE2pPV0JpuuK9djTO8HHeFkH70oYGPCBT8agCX1zbYCnHNKE0I4OpsjTi4BHvcsk7zXqCIdbd608wLTU/NEeM5zTFKf6y1Hl6SxPIP5ax84+LRvKQTNfzgKLsjD1fxE8/9mYOZ57iTNCrxBT2v7HH09Pem97P/5I/Qxc+j2sgyxCeCl/a+A/Rt0jxg==')
tt_v13_smtrace8_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v13_smtrace8_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v13_smtrace8_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v13_smtrace8_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v13_smtrace8_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v13_smtrace8_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v13_smtrace8_SMTRACE_ESC_CAP = 32

def tt_v13_smtrace8_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
 return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v13_smtrace8_smtrace_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
  cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v13_smtrace8_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, tt_v13_smtrace8_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v13_smtrace8_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v13_smtrace8_smtrace_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def tt_v13_smtrace8__smtrace_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v13_smtrace8__smtrace_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v13_smtrace8_smtrace_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = tt_v13_smtrace8__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def tt_v13_smtrace8_smtrace_c12_bytes(v):
 return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc):
 return b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc)

def tt_v13_smtrace8_smtrace_table(base):
 return b2q_tt_v10_m32x_m32x_table(base)

def tt_v13_smtrace8_smtrace_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [tt_v13_smtrace8_smtrace_units(hi), tt_v13_smtrace8_smtrace_units(lo)]
 else:
  mats = [tt_v13_smtrace8_smtrace_units(w)]
 if fmt in (0, 2):
  data = torch.cat([tt_v13_smtrace8_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = tt_v13_smtrace8_smtrace_c12_bytes(v)
  if not torch.equal(tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(tt_v13_smtrace8_smtrace_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def tt_v13_smtrace8_smtrace_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, tt_v13_smtrace8_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > tt_v13_smtrace8_SMTRACE_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = tt_v13_smtrace8__smtrace_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
tt_v13_smtrace8_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_MAX_M = 32
tt_v13_smtrace8_SMTRACE_TABLE_M.update({('router', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1]})
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def tt_v13_smtrace8_smtrace_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v13_smtrace8_smtrace_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v13_smtrace8_smtrace_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v13_smtrace8_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def tt_v13_smtrace8_smtrace_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = tt_v13_smtrace8_smtrace_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = tt_v13_smtrace8_smtrace_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = tt_v13_smtrace8_smtrace_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v13_smtrace8_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v13_smtrace8_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def tt_v13_smtrace8_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
 table_m = dict(tt_v13_smtrace8_SMTRACE_TABLE_M if table_m is None else table_m)
 table_mt = dict(tt_v13_smtrace8_SMTRACE_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = tt_v13_smtrace8_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, tt_v13_smtrace8_SMTRACE_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if tt_v13_smtrace8_smtrace_bf16_twin(inst):
   instances.append(tt_v13_smtrace8_smtrace_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = tt_v13_smtrace8_smtrace_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and tt_v13_smtrace8_smtrace_bf16_twin(inst):
   plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v13_smtrace8_smtrace_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 return st

def tt_v13_smtrace8_smtrace_run_core(x, layer, n, st):
 m = x.shape[0]
 if not 1 <= m <= tt_v13_smtrace8_SMTRACE_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v13_smtrace8_smtrace_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v13_smtrace8_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
  return None
 x = x.contiguous()
 if x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 if tt_v13_smtrace8_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out
tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def tt_v13_smtrace8_prepare_smtrace(weights):
 usable = {}
 for ((layer, m), (inst, cfg)) in tt_v13_smtrace8_SMTRACE_ROUTES.items():
  w = weights.get(layer)
  if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
   usable[layer] = w
 routes = {key: value for (key, value) in tt_v13_smtrace8_SMTRACE_ROUTES.items() if key[0] in usable}
 return tt_v13_smtrace8_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def tt_v13_smtrace8_run_smtrace(x, layer, n, st):
 if (layer, x.shape[0]) not in tt_v13_smtrace8_SMTRACE_ROUTES:
  return None
 return tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)
V13_ROUTES = [{'m': 8, 'layer': 'unpooler', 'name': 'smtrace8', 'module': 'v13_smtrace8'}]
V13_SELECTED_ROUTES = [r for r in V12_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in V13_ROUTES}] + V13_ROUTES

def prepare_v13(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v12(weights)
 state['v13_smtrace8'] = tt_v13_smtrace8_prepare_smtrace(weights)
 for route in V13_ROUTES:
  (m, layer) = (route['m'], route['layer'])
  weight = weights[layer]
  n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
  data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
  candidate_v13(data, state)
 torch.cuda.synchronize()
 return state

def candidate_v13(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'unpooler' and x.shape[0] == 8:
  output = tt_v13_smtrace8_run_smtrace(x, layer, n, state['v13_smtrace8'])
  if output is not None:
   return output
 return candidate_v12(data, state)
'QKV candidates. Production routes are enabled only after measured validation.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def tt_v14_qkv_qbig_mma(n):
 regs = ','.join((f'%{i}' for i in range(n // 2)))
 outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
 return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'
tt_v14_qkv_QKVBIG_CUDA = _zsrc('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')

def tt_v14_qkv_build_qkv(configs, profile=False):
 ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
 source = tt_v14_qkv_QKVBIG_CUDA.replace('@@MMA@@', '\n'.join((tt_v14_qkv_qbig_mma(n) for n in ns)))
 source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
 source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
 source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return qb_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for (i, c) in enumerate(configs))))
 digest = hashlib.sha256(source.encode()).hexdigest()[:16]
 namespace = 'qb_' + digest
 begin = source.index('#define QD')
 args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
 source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
 cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
 return load_inline('qbig_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def tt_v14_qkv_qbig_pack(w, groups, kt=1):
 rows = (2304 + groups - 1) // groups
 if kt == 2:
  return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
 if 2304 % groups == 0:
  return w.contiguous()
 out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
 for g in range(groups):
  (lo, hi) = (g * 2304 // groups, (g + 1) * 2304 // groups)
  out[g, :hi - lo].copy_(w[lo:hi])
 return out.reshape(-1, 8192)

def tt_v14_qkv_prepare_qbig_candidates(weights, configs, profile=False):
 w = weights['qkv_gate']
 return {'ext': tt_v14_qkv_build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): tt_v14_qkv_qbig_pack(w, g, kt) for (g, kt) in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

def tt_v14_qkv_run_qbig_candidate(x, st, index, stats=None, query=0):
 out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
 cfg = st['configs'][index]
 rc = st['ext'].qb_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
 if query:
  return rc
 if rc:
  raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
 return out
tt_v14_qkv_EXPERIMENT_CONFIGS = [(64, 36, 128, 3, 8, 539, 5), (64, 36, 128, 3, 8, 6683, 5), (64, 36, 256, 3, 4, 219, 5)]
tt_v14_qkv_ENABLED_ROUTES = {128: 0}

def tt_v14_qkv_prepare_qkvbig(weights: dict) -> dict:
 w = weights.get('qkv_gate')
 if w is None or w.shape != (2304, 8192) or w.dtype != torch.bfloat16 or (not w.is_cuda):
  return {'routes': {}}
 if torch.cuda.get_device_capability(w.device) != (9, 0) or 'H200' not in torch.cuda.get_device_name(w.device):
  return {'routes': {}}
 if not tt_v14_qkv_ENABLED_ROUTES:
  return {'routes': {}}
 state = tt_v14_qkv_prepare_qbig_candidates(weights, tt_v14_qkv_EXPERIMENT_CONFIGS)
 state['routes'] = dict(tt_v14_qkv_ENABLED_ROUTES)
 for (m, index) in state['routes'].items():
  x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
  tt_v14_qkv_run_qbig_candidate(x, state, index)
 torch.cuda.synchronize(w.device)
 return state

def tt_v14_qkv_run_qkvbig(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 if layer != 'qkv_gate' or n != 2304 or x.ndim != 2 or (x.shape[1] != 8192):
  return None
 index = st.get('routes', {}).get(x.shape[0])
 if index is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
  return None
 if x.device != st['device'] or x.data_ptr() % 16:
  return None
 return tt_v14_qkv_run_qbig_candidate(x, st, index)
V14_ROUTES = [{'m': 128, 'layer': 'qkv_gate', 'name': 'qkvbig', 'module': 'v14_qkv'}]
V14_SELECTED_ROUTES = [r for r in V13_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in V14_ROUTES}] + V14_ROUTES

def prepare_v14(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v13(weights)
 state['v14_qkv'] = tt_v14_qkv_prepare_qkvbig(weights)
 for route in V14_ROUTES:
  (m, layer) = (route['m'], route['layer'])
  weight = weights[layer]
  n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
  data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
  candidate_v14(data, state)
 torch.cuda.synchronize()
 return state

def candidate_v14(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'qkv_gate' and x.shape[0] == 128:
  output = tt_v14_qkv_run_qkvbig(x, layer, n, state['v14_qkv'])
  if output is not None:
   return output
 return candidate_v13(data, state)
V15_ROUTES = [{'m': 256, 'layer': 'unpooler', 'name': 'ltsearch_transposed', 'module': 'lt', 'descriptor': [66, 20, 1, 0, 0, 4, 35, 0, 2], 'layout': 3, 'epilogue': 0, 'algorithm_workspace_bytes': 0}]
V15_SELECTED_ROUTES = [r for r in V14_SELECTED_ROUTES if (r['m'], r['layer']) not in {(n['m'], n['layer']) for n in V15_ROUTES}] + V15_ROUTES

def prepare_v15(weights: _AchitaDenseWeights) -> dict:
 state = prepare_v14(weights)
 for route in V15_ROUTES:
  (m, layer) = (route['m'], route['layer'])
  weight = weights[layer]
  n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
  data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
  candidate_v15(data, state)
 torch.cuda.synchronize()
 return state

def candidate_v15(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 (x, layer, n) = (data['x'], data['config']['layer'], data['config']['n'])
 if layer == 'unpooler' and x.shape[0] == 256:
  output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
  if output is not None:
   return output
 return candidate_v14(data, state)
_bundle_base_prepare = prepare_v15
_bundle_base_kernel = candidate_v15
'Experimental lossless nine-instruction BF16 decode. No route is promoted. See FINDINGS.txt.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
dd_decodediet_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
dd_decodediet_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def dd_decodediet_dd_cap2_b_cps32_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0
dd_decodediet_CPSSM_CUDA_TEMPLATE = _zsrc('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')
dd_decodediet_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
dd_decodediet_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
dd_decodediet_CPSSM_ESC_CAP = 32

def dd_decodediet_dd_cap2_b_cpssm_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = dd_decodediet_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = dd_decodediet_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def dd_decodediet_dd_cap2_b_cpssm_table(base):
 return base * 16843009

def dd_decodediet_dd_cap2_b_cpssm_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [dd_decodediet_dd_cap2_b_cpssm_units(hi), dd_decodediet_dd_cap2_b_cpssm_units(lo)]
 else:
  mats = [dd_decodediet_dd_cap2_b_cpssm_units(w)]
 if fmt in (0, 2):
  data = torch.cat([dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v)
  if not torch.equal(dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(dd_decodediet_dd_cap2_b_cpssm_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def dd_decodediet_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, dd_decodediet_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > dd_decodediet_CPSSM_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = dd_decodediet_cpstage___to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
dd_decodediet_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
dd_decodediet_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def dd_decodediet_dd_cap2_b_cpssm_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def dd_decodediet_dd_cap2_b_cpssm_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def dd_decodediet_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = dd_decodediet_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = dd_decodediet_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = dd_decodediet_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def dd_decodediet_cpstage___to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def dd_decodediet_cpstage___to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
dd_decodediet_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def dd_decodediet_cps_build(family, instances, depth):
 is32 = False
 prefix = 'dd_cap2_b_cpssm_'
 template = dd_decodediet_CPSSM_CUDA_TEMPLATE
 cpp = dd_decodediet_CPSSM_CPP
 cases = []
 for (i, inst) in enumerate(instances):
  full = inst if len(inst) == 14 else inst + (0,) * 7
  args = ', '.join(map(str, (*full, depth)))
  cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
 source = template.replace('@@CASES@@', '\n'.join(cases))
 name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
 return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def dd_decodediet_prepare_cps(weights, depth=None, layers=dd_decodediet_CPS_LAYERS, buckets=(2, 4), include_policy=False):
 if depth is None:
  st = {'plans': {}, 'depth': 'selected', 'policy': 0}
  selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
  for (bucket, selected_depth) in ((2, 2), (4, 1)):
   if bucket in buckets and selected_layers:
    part = dd_decodediet_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
    st['plans'].update(part['plans'])
  return st
 if depth not in (0, 1, 2, 3):
  raise ValueError('ring depth must be 0, 1, 2 or 3')
 st = {'plans': {}, 'depth': depth, 'policy': 0}
 for family in ('sm',):
  table = dd_decodediet_CPSSM_TABLE_MT
  entries = []
  for ((layer, bucket), (inst, cfg)) in table.items():
   is_policy = include_policy and family == 'sm' and (layer in dd_decodediet_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
   if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
    continue
   if not is_policy and family == 'sm' and ((layer, bucket) in dd_decodediet_CPS32_TABLE_MT):
    continue
   if len(inst) == 7:
    rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
    inst = inst + (0, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
   entries.append(((layer, 'mt', bucket), (inst, cfg)))
  if include_policy and family == 'sm':
   for ((layer, m), (inst, cfg)) in dd_decodediet_CPSSM_TABLE_M.items():
    if layer not in dd_decodediet_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
     continue
    rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
    if len(inst) == 7:
     inst = inst + (m, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
    entries.append(((layer, 'm', m), (inst, cfg)))
  if not entries:
   continue
  twin = dd_decodediet_dd_cap2_b_cpssm_bf16_twin
  instances = list(dict.fromkeys((inst for (_, (inst, _)) in entries for inst in (inst, twin(inst)) if inst is not None)))
  ext = dd_decodediet_cps_build(family, instances, depth)
  ids = {inst: i for (i, inst) in enumerate(instances)}
  plan_fn = dd_decodediet_dd_cap2_b_cpssm_plan
  (packs, streams) = ({}, {})
  for (key, (inst, cfg)) in entries:
   sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
   chosen = inst
   if sp is None and twin(inst) is not None:
    chosen = twin(inst)
    sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
   if sp is not None:
    st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
 return st

def dd_decodediet_cps_launch(sp, x, out, n, policy=0, query=0):
 fn = dd_decodediet_dd_cap2_b_cpssm_launch_raw
 return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def dd_decodediet_run_cps(x, layer, n, st):
 if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
  return None
 m = x.shape[0]
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', dd_decodediet_dd_cap2_b_cps32_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != dd_decodediet_CPS32_LAYER_K[layer]) or (not x.is_cuda):
  return None
 if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 rc = dd_decodediet_cps_launch(sp, x, out, n, st.get('policy', 0))
 if rc:
  raise RuntimeError(f'cpstage launch failed: {rc}')
 return out
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def dd_decodediet_tt_mw_mma(n):
 nr = n // 2
 regs = ','.join((f'%{i}' for i in range(nr)))
 outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
 inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
 ops = ['wgmma.fence.sync.aligned;']
 for kk in range(4):
  ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
  ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
 ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
 return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
dd_decodediet_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = prmt(L0, low_high, 0x5140u);\n  a[1] = prmt(L0, low_high, 0x7362u);\n  a[2] = prmt(L1, high_high, 0x5140u);\n  a[3] = prmt(L1, high_high, 0x7362u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
dd_decodediet_tt_mw_CUDA_CORE = _zsrc('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')
dd_decodediet_tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
dd_decodediet_tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
dd_decodediet_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def dd_decodediet_tt_mw_cuda_source(instances):
 cases = []
 attrs = []
 for (i, inst) in enumerate(instances):
  (bn, r, wk, pf, q, c, cap) = inst
  params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
  cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
  attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
 return dd_decodediet_tt_mw_CUDA_PREFIX + '\n'.join((dd_decodediet_tt_mw_mma(n) for n in (16, 32, 64))) + dd_decodediet_tt_mw_CUDA_CORE + dd_decodediet_tt_mw_CUDA_MAP + dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def dd_decodediet_tt_mw_build(instances):
 src = dd_decodediet_tt_mw_cuda_source(instances)
 name = 'dd_cap2_b_m32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
 return load_inline(name, dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])

def dd_decodediet_tt_mw__to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def dd_decodediet_tt_mw__to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def dd_decodediet_tt_mw_sm_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = dd_decodediet_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def dd_decodediet_tt_mw_sm_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = dd_decodediet_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_tt_mw_sm_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def dd_decodediet_tt_mw_sm_table(base):
 return base * 16843009

def dd_decodediet_tt_mw_pack_weight(w, r, wk, split, compressed=True):
 (n, k) = w.shape
 total = wk * split
 rw = r // 16
 if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
  return None
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
 esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
 (table, max_esc, escapes) = (0, 0, 0)
 if compressed:
  (data, b, entries) = dd_decodediet_tt_mw_sm_c12_bytes(v)
  if not torch.equal(dd_decodediet_tt_mw_sm_c12_unpack(data, b, entries), v):
   return None
  table = dd_decodediet_tt_mw_sm_table(b) & 18446744073709551615
  (er, ec, el, ev, xor7) = entries
  escapes = er.numel()
  if escapes:
   boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
   rank = torch.bucketize(ec, boundaries, right=True)
   seq = ec - k // 64 * rank // total
   group = (er // rw * total + rank) * rw + er % rw
   max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
   if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
    return None
   order = torch.argsort(group * (1 << 14) + seq)
   gs = group[order]
   ss = seq[order]
   pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
   entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
   esc[gs, pos] = dd_decodediet_tt_mw__to_i32(entry)
 else:
  data = dd_decodediet_tt_mw_sm_bf16_bytes(v)
 payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
 return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def dd_decodediet_tt_mw_launch_raw(ext, idx, pk, x, out):
 return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
 (n, k) = w.shape
 eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
 out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
 for start in range(0, k, m):
  if dd_decodediet_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
   return False
 return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
dd_decodediet_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def dd_decodediet_tt_mw_prepare_m32w(weights, configs=None) -> dict:
 configs = dd_decodediet_tt_mw_CONFIGS if configs is None else configs
 device = next(iter(weights.values())).device
 state = {'plans': {}, 'device': device}
 if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
  return state
 instances = []
 for (inst, split) in configs.values():
  for c in (inst[5], False):
   twin = inst[:5] + (c,) + inst[6:]
   if twin not in instances:
    instances.append(twin)
 ext = dd_decodediet_tt_mw_build(instances)
 state['ext'] = ext
 packs = {}
 for ((m, layer), (inst, split)) in configs.items():
  if layer not in weights:
   continue
  w = weights[layer]
  for compressed in (inst[5], False):
   actual = inst[:5] + (compressed,) + inst[6:]
   idx = instances.index(actual)
   if ext.info(idx) % 10000:
    continue
   key = (layer, inst[1], inst[2], split, compressed)
   if key not in packs:
    packs[key] = dd_decodediet_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
   pk = packs[key]
   if pk is None or not dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m):
    continue
   if m == 32 and (not dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
    continue
   state['plans'][m, layer] = (idx, pk)
   break
 return state

def dd_decodediet_tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
  return None
 m = x.shape[0]
 if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
  return None
 key = (16 if m <= 16 else 32, layer)
 plan = st['plans'].get(key)
 if plan is None:
  return None
 (idx, pk) = plan
 if n != pk['n'] or x.shape[1] != pk['k']:
  return None
 out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
 rc = dd_decodediet_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
 if rc:
  raise RuntimeError(f'm32w launch failed: {rc}')
 return out
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
dd_DECODEDIET_ROUTES = (('pooler', 16), ('unpooler', 16))

def dd_prepare_decodediet(weights) -> dict:
 routes = dd_DECODEDIET_ROUTES
 state = {'routes': frozenset(routes)}
 if not routes:
  return state
 if any((m == 16 and layer in ('pooler', 'unpooler') for (layer, m) in routes)):
  layers = tuple((layer for (layer, m) in routes if m == 16 and layer in ('pooler', 'unpooler')))
  state['cps'] = dd_decodediet_prepare_cps(weights, layers=layers, buckets=(2,))
 return state

def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
 m = x.shape[0] if x.ndim == 2 else 0
 if (layer, m) not in state.get('routes', ()):
  return None
 return dd_decodediet_run_cps(x, layer, n, state['cps'])
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
rw_SELECTED = ()
rw_SOURCES = {128: _zsrc('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'), 256: _zsrc('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')}

def rw__prepare(weights: dict, arms: tuple) -> dict:
 w = weights.get('router')
 if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
  return {'enabled': False}
 if torch.cuda.get_device_capability(w.device) != (9, 0):
  return {'enabled': False}
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)
 packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)
 state = {'enabled': True, 'weight': w, 'packed': packed, 'arms': {}}
 for arm in arms:
  label = 'rwdl_' + str(arm)
  source = rw_SOURCES[arm]
  cpp = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'
  ext = load_inline('routerwide_' + hashlib.sha256(source.encode()).hexdigest()[:14], cpp, cuda_sources=source, functions=[label + '_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
  state['arms'][arm] = (ext, getattr(ext, label + '_launch'))
  x = torch.zeros((arm, 8192), device=w.device, dtype=torch.bfloat16)
  rw_run_routerwide_candidate(x, state, arm)
 torch.cuda.synchronize(w.device)
 return state

def rw_prepare_routerwide(weights: dict) -> dict:
 return rw__prepare(weights, rw_SELECTED) if rw_SELECTED else {'enabled': False}

def rw_prepare_routerwide_candidates(weights: dict) -> dict:
 return rw__prepare(weights, (128, 256))

def rw_run_routerwide_candidate(x: torch.Tensor, state: dict, arm: int) -> torch.Tensor:
 out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
 rc = state['arms'][arm][1](0, state['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream(x.device).cuda_stream, 0)
 if rc:
  raise RuntimeError(f'routerwide launch failed: {rc}')
 return out

def rw_run_routerwide(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
 if not st.get('enabled') or layer != 'router' or n != 256 or (x.ndim != 2):
  return None
 if x.shape[1] != 8192 or x.shape[0] not in rw_SELECTED or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
  return None
 if x.device != st['weight'].device or x.data_ptr() % 16:
  return None
 return rw_run_routerwide_candidate(x, st, x.shape[0])
'Router completion experiment. No route passed H200 confirmation.'
import hashlib
rt_SELECTED = ()
rt__CUDA = _zsrc('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')
rt__CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def rt__pack(w, s=16, wk=8):
 import torch
 hi = w.to(torch.bfloat16)
 lo = (w - hi.float()).to(torch.bfloat16)

 def units(v):
  v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
  return v.permute(0, 1, 3, 2, 4)
 data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
 ku = 128 // s // wk
 return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def rt_prepare_routertail_candidates(weights) -> dict:
 import torch
 from torch.utils.cpp_extension import load_inline
 w = weights.get('router')
 if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
  return {'enabled': False}
 if torch.cuda.get_device_capability(w.device) != (9, 0):
  return {'enabled': False}
 name = 'routertail_' + hashlib.sha256(rt__CUDA.encode()).hexdigest()[:12]
 ext = load_inline(name, rt__CPP, cuda_sources=rt__CUDA, functions=['routertail_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
 st = {'enabled': True, 'ext': ext, 'weight': w.contiguous(), 'packed': rt__pack(w)}
 for m in (8, 16, 17):
  x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
  rt_run_routertail_candidate(x, st)
 torch.cuda.synchronize(w.device)
 return st

def rt_run_routertail_candidate(x, st):
 import torch
 if not st.get('enabled') or x.ndim != 2 or x.shape[1] != 8192 or (not 2 <= len(x) <= 32):
  return None
 if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != st['weight'].device:
  return None
 if not x.is_contiguous() or x.data_ptr() % 16:
  return None
 out = torch.empty((len(x), 256), device=x.device, dtype=torch.float32)
 rc = st['ext'].routertail_small(st['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 8, torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f'routertail launch failed: {rc}')
 return out

def rt_prepare_routertail(weights) -> dict:
 return {'enabled': False}

def rt_run_routertail(x, layer, n, st):
 return None
'Lossless BF16 projection refinements; only confirmed exact shapes are claimed.'
import torch
from torch.utils.cpp_extension import load_inline
mt_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
mt_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
mt_W0_DC2_ESC_CAP = 32

def mt_wp0_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def mt_wp0_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def mt_wp0_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def mt_wp0_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = mt_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def mt_wp0_c12_bytes(v, decode=0):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = mt_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def mt_wp0_c12_unpack(data, base, esc, decode=0):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def mt_wp0_table(base):
 t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
 t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
 v = t0 | t1 << 32
 return v - 2 ** 64 if v >= 2 ** 63 else v

def mt_wp0_pack_format(w, fmt, layout=0, decode=2):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [mt_wp0_units(hi), mt_wp0_units(lo)]
 else:
  mats = [mt_wp0_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([mt_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = mt_wp0_c12_bytes(v, decode)
  if not torch.equal(mt_wp0_c12_unpack(data, base, esc, decode), v):
   return None
  if decode in (1, 2, 4) and (not 1 <= base <= 119):
   return None
  parts.append(data)
  if decode in (1, 2, 4):
   scale = 127 + 2 * (base - 64) << 7
   tabs.append(scale | scale << 16)
  elif decode in (3, 5):
   tabs.append(base << 8 | base << 24)
  else:
   tabs.append(mt_wp0_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def mt_wp0_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def mt_wp0_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = mt_wp0_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, mt_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > mt_W0_DC2_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = mt_wp0_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
mt_M16TRACE_ROUTES = {'o': {'cfg': (2, 4, 4, 1), 'inst': (1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), 'decode': 0, 'rows': 8192, 'k': 1024, 'variant': 'next_early'}}
mt_M16TRACE_SOURCES = {'o': ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n', _zsrc('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'), 'm16trace_59fcb5b72994_launch')}

def mt__launch(fn, index, plan, x, out, query=0) -> int:
 return fn(index, plan['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(), x.shape[0], plan['rows'], plan['k'], *plan['cfg'], plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream, query)

def mt__onehot(fn, index, plan, weight) -> bool:
 x = torch.zeros((16, plan['k']), device=weight.device, dtype=torch.bfloat16)
 out = torch.empty((16, plan['rows']), device=weight.device, dtype=torch.bfloat16)
 ids = torch.arange(16, device=weight.device)
 for column in range(0, plan['k'], 16):
  x.zero_()
  x[ids, column + ids] = 1
  if mt__launch(fn, index, plan, x, out):
   return False
  if not torch.equal(out, weight[:, column:column + 16].t()):
   return False
 return True

def mt_prepare_m16trace(weights) -> dict:
 state = {'plans': {}}
 for (layer, route) in mt_M16TRACE_ROUTES.items():
  weight = weights.get(layer)
  if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16:
   continue
  if weight.ndim != 2 or tuple(weight.shape) != (route['rows'], route['k']):
   continue
  if torch.cuda.get_device_capability(weight.device) != (9, 0):
   continue
  (cpp, cuda, name) = mt_M16TRACE_SOURCES[layer]
  ext = load_inline(name, cpp, cuda_sources=cuda, functions=[name], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
  fn = getattr(ext, name)
  for (index, fmt) in enumerate((1, 0)):
   pk = mt_wp0_pack_format(weight, fmt, decode=route['decode'])
   if pk is None:
    continue
   sp = mt_wp0_stream(pk, 1, *route['cfg'])
   if sp is None:
    continue
   plan = dict(sp, **route, fn=fn, index=index)
   x = torch.empty((16, route['k']), device=weight.device, dtype=torch.bfloat16)
   out = torch.empty((16, route['rows']), device=weight.device, dtype=torch.bfloat16)
   attr = mt__launch(fn, index, plan, x, out, 2)
   if attr < 0 or attr % 1000:
    continue
   if not mt__onehot(fn, index, plan, weight):
    continue
   state['plans'][layer] = plan
   break
 return state

def mt_run_m16trace(x, layer, n, state) -> torch.Tensor | None:
 if x.ndim != 2 or x.shape[0] != 16 or (not x.is_cuda) or (x.dtype != torch.bfloat16) or (not x.is_contiguous()):
  return None
 plan = state['plans'].get(layer)
 if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
  return None
 if x.device != plan['wp'].device or x.data_ptr() % 16:
  return None
 out = torch.empty((16, n), device=x.device, dtype=torch.bfloat16)
 rc = mt__launch(plan['fn'], plan['index'], plan, x, out)
 if rc:
  raise RuntimeError(f'm16trace launch failed: {rc}')
 return out
BUNDLE_QKV_CUDA = _zsrc('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')
BUNDLE_ROUTES = {'router128': (128, 'router'), 'router256': (256, 'router'), 'qkv8': (8, 'qkv_gate'), 'router8': (8, 'router'), 'router16': (16, 'router'), 'o16': (16, 'o'), 'pooler16': (16, 'pooler'), 'unpooler16': (16, 'unpooler')}
dd_DECODEDIET_ROUTES = (('pooler', 16), ('unpooler', 16))

def prepare_bundle(weights) -> dict:
 state = {'dd': dd_prepare_decodediet(weights), 'rw': rw_prepare_routerwide_candidates(weights), 'rt': rt_prepare_routertail_candidates(weights), 'mt': mt_prepare_m16trace(weights)}
 symbol = 'qs_literal_owner5_decode5_launch'
 cpp = SM_CPP.replace('sm_launch', symbol)
 ext = load_inline('bundle_owner_' + hashlib.sha256(BUNDLE_QKV_CUDA.encode()).hexdigest()[:12], cpp, cuda_sources=BUNDLE_QKV_CUDA, functions=[symbol], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
 pk = tt_v10_m32x_m32x_pack_format(weights['qkv_gate'], 1, 0, 5)
 sp = None if pk is None else sm_stream(pk, 1, 1, 4, 8, 4)
 state['qkv'] = None if sp is None else dict(sp, fn=getattr(ext, symbol))
 return state

def run_bundle(x, layer, n, st) -> torch.Tensor | None:
 if x.ndim != 2:
  return None
 m = x.shape[0]
 if layer in ('pooler', 'unpooler') and m == 16:
  return dd_run_decodediet(x, layer, n, st['dd'])
 if layer == 'router' and m in (128, 256):
  if n != 256 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
   return None
  if not st['rw'].get('enabled') or x.device != st['rw']['weight'].device or x.data_ptr() % 16:
   return None
  return rw_run_routerwide_candidate(x, st['rw'], m)
 if layer == 'router' and m in (8, 16):
  return rt_run_routertail_candidate(x, st['rt']) if n == 256 else None
 if layer == 'o' and m == 16:
  return mt_run_m16trace(x, layer, n, st['mt'])
 if layer == 'qkv_gate' and m == 8:
  sp = st['qkv']
  if sp is None or n != 2304 or x.shape[1] != 8192 or (x.dtype != torch.bfloat16) or (not x.is_cuda) or (not x.is_contiguous()):
   return None
  if x.device != sp['wp'].device or x.data_ptr() % 16:
   return None
  out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
  rc = sp['fn'](1, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, 8192, 1, 4, 8, 4, sp['t01'], sp['u01'], torch.cuda.current_stream(x.device).cuda_stream, 0)
  if rc:
   raise RuntimeError(f'bundle qkv launch failed: {rc}')
  return out
 return None

def _achita_dense_prepare(weights: _AchitaDenseWeights) -> dict:
 state = _bundle_base_prepare(weights)
 state['bundle'] = prepare_bundle(weights)
 return state

def _achita_dense_kernel(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 output = None
 if 'bundle' in state:
  output = run_bundle(data['x'], data['config']['layer'], data['config']['n'], state['bundle'])
 return _bundle_base_kernel(data, state) if output is None else output
V16_ROUTES = [{'m': 128, 'layer': 'router', 'name': 'router128', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 256, 'layer': 'router', 'name': 'router256', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 8, 'layer': 'qkv_gate', 'name': 'qkv8', 'module': 'v16_bundle', 'owner_family': 'owner5_decode5'}, {'m': 8, 'layer': 'router', 'name': 'router8', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'router', 'name': 'router16', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'o', 'name': 'o16', 'module': 'v16_bundle', 'owner_family': 'm16trace_early'}, {'m': 16, 'layer': 'pooler', 'name': 'pooler16', 'module': 'v16_bundle', 'owner_family': 'decodediet'}, {'m': 16, 'layer': 'unpooler', 'name': 'unpooler16', 'module': 'v16_bundle', 'owner_family': 'm32x_round3_decoder9'}]
SELECTED_ROUTES = [r for r in V15_SELECTED_ROUTES if (r['m'], r['layer']) not in {(q['m'], q['layer']) for q in V16_ROUTES}] + V16_ROUTES
_b2_base_prepare = _achita_dense_prepare
_b2_base_kernel = _achita_dense_kernel
b2p_tt_v10_m32x_M32X_ESC_CAP = 32
b2p_tt_v10_m32x__CFG = (1, 8, 16, 1)
b2p_tt_v10_m32x__CUDA = _zsrc('eNrNPW1X27jS3/sr1O6h107cNE5SoATYQ2l37x4K7SF0u8/hYXMcxyEpjh1s5wW6/Pc7M5Js+SWJA713tz0hsiyNRqOZ0czoJT/1ncHIc9j740b3/YfjT+8/sDfPnv008mx32nfYvj3tW91g6kWjsVMbHmbf9Abmdjo7jPojL0rnWe61H4yi4VjNjO4mTjcKrFEUpsD6/sQJrGg0c7rXgT+dhAjr2U8Cz99+ec+63b4zG9lOtwvJgR/Yzshz4WW3++wZvof2Wyxodie+7zYb9OUE3f7brm1NGl2rO242Fl23P9ds3wsjKl5hE519f8Z43VkbUlY4ZjPfBUxcR3vh9mvXrt+z3Jpn1z6ae3ue37Vc17etyKnNWrWw2WDft+rGlmlsNYyt5oPBLrdaV+0XACn1b4+9OAheaLPaQjdk8i5J3ifJuY6F3RfaRNcRocCJpoGHyD3wfk5LdnQhOjrzR/24o9NSPYWeTZf17Ed3BXoQleiNI3ojq6R6RDDWdQp7BB2CbphqN34YjpNgHGlxactIavaUdJjBui+w1l5g/VpPIAlkZ0B3ttVMUO0TqpCwkMTw3RPfYQr1fow6jvxatMdjSxu4vhUx7aWtX7auDJaldKovdfXB5L0h/AFQLbwDQlvu6Npz+rWxue3t3oCmCPx5zfbd2qDZqJHqoD+DPId932oZW2+MrW1ja4ced42tt5RQisXCBcSoDl5o9mX9CgnB06aSbijp5pWeVEMailqUNJNkI0k2ZbJXlwkTKQ3Uff2a7bK5M7oeRiHzB8wHHQU9ZWHkTPaY689Z7y5yQtAr7GOdaTPLncJT/VVTZ5bXZx/NOK/1ageAt171RhHQvc/rHNfKj1/fsbu+208476M6Qh9N5eFYSV+opS6UUjDcfFDTXMDsMTtgx+wlqy92xL9pO19s6PpQrpSsIArYsj3WiwANR5sDYoeHzNwmcDjAZQEgyQBxA/r2xmzVpzr7i2naMdvfZ2ZDpz7v1ut1/EwFcPMxwHea2w0V+G4x7EZ52NBrIFQR4q1i2M3HwE7wPs4B3VRNjqdusZYMbct1lmpHqFYLPNIcC1VFrlCPHGCxaiyPrhXedHsjK1SQXoqk60+aKRUOpNoZDPCD6RaQDD+YdqxCzJ+gyEER/HglkNDJj8DmSboaDge1oDYPrEl6zoJPK+mZqCX7d6wXSTqoy+4QVCkwpuSvHf4feK4KSBZpBygfVxKtLKu6qSaQ+CRS9RiRV6Fw+XmUcMt+ZpFpPh6MxAaYK2VTe740qMsxnOfMnEBLzMuM8QVG/qT9Qm8zPl+eYWkWWTeOZ7DeNGKTaGGFzLY8z4eHwJ85bBTtMcfrhywaOqxnhSOb9cDYvmGhD/S04MUoDKcOmw+dAP6CcxE53gZzpR/0IXMC/Yw0+LCxkbeOib2ivC059mexCbm1ZSNaCZ9HnLFHA9C+EfBhLIj6WGcHB0j0Af8HfFmOrpNH6NZw7Iy7gGTO8pcqJcGr27VnkdW9djwnGNndyO+GQytw+tgu27TdwPJutKzCWE/CKZhLQdeOLKyvKI2UFgweo7MnVvEcgw0t0d4qnlC/xsmxtyfQTBBfN+1QG08xyHtW0B15I8WZgBylE7YPjnlseKt4Q7lgBLhi7aQDkVXrbbfA76lfIfqI+h5Zs1YgUBYQoR/AQH5w90J/BMrOYuLYS5GOFisxxq+ZU+MwutEih33XWNUBhP5o7AeOBxqQKJ7XYvSyJvGkUrXAcR0rdGqCNQgftXWQn9KN226X972gbUkcyYKCSNC8tQD/SvhZT21+bhV2PNs4FlvVZPk2w6hroaPY0hS5NBg5oS02U8UVR7iAbULAhFzNrJDKcYIcfzxxncgBXtrbI2cMoxroc3I2+p542PBpUXjjTVHgZo/LNDLaII56DOKgxyCOeQx4yEMoAcK8iCV7QJNyAhUFdylpyigu/yavub7/v8eAP65ZbQJ0YZM2PMcyBvBosGsTC2bOu5pl305HQczGOZGb8FgJmAwAJkeYFyx03EmsFtHYNajBh1g1+jdSN+apEWtH/4Y9hwkypeW3WyW0/DUGJ5VZhyotm7ihO3zW4fEgrBoIHe7G07fAKJIu/jtQi32gCLNYL/DBcGFg03ihZUcj3+NqmIVDfx6y6QQahWLcmmIDa+pihBHIavUxPoDz8XQ88q7Z1BuADgmHMDp9CwhdXmxw6LpAyAKeQLujNEthLV30NW1ZcwrWSxiWgvZYfT4EKjPt+WbtEzeVQVuO8itE7JA16vzf1HV1MFnRxtQ2VPdWv9/iEa/WSzQOeP9lRo80YW3BqgesV1u0IX3H03eYvufpe0zPeXoeaz+hwdbqv+HA7S78oKWlmwbNh0apz8dUMOMYDOYuL6F146pd1H5afTGQViUYsrWFAXUNtqbUHZXKSjP/t6bqfakG5lhqwzEJIz9woH/erNujzprbFdQ/PDA5wzGpTIAvu5wUDVmoG3jabLMZjzdF9QubmNFwRg5MH+BUsn0ckl9OLw5ZGAVTO2KnAOSXMeiIpEj23X79EKCF4JSCA0NjDHZNQIN79hGaaBns7Ai+TWhqJRxzNZxmWTiN1XB2BZzGOjjN1XC2U3AKaLhRMMOPY/60ZqG9nOuXEhEEtrd39vFKMRa0l0e5AkeswsztqxUhjyJBiN9/UQt/MWNlm/Rdg4bIwWN//cVEuoHlfpoE1vXYAn0f+K5LrQx8KI+kcBlOePC1z7IdarNq1eXt4L+jyxb0wGVVRiGMxH2bX7pXoJ1yxcyCYnf5Yo2CYvf5Ys2CYnNe7AH+PjDHDZ01nR3yzg7znT3Czg6TzqrrclOoNUcMAJNhO1UgcSlwphqyn2GYYCLHgY1MmWNijikrlgqdXdavUIXS9x1+N+g5IrgGO0LqbiM6OBr6xqDvBei5AH23BHRrQ9CmwNpMsL5fAnp3c9D3ArTEer4EtNnQFb4Q9tOHEGA6zPFg2t8Dc/GWAuq77C/mWp5DD014AHt1hAYoRcTheWi5A0zvQHqHFkb++HSOVhRGhSjmh/b8v0IGEuh7AJ1BmbCW0Tdn5xuom4kV2UMtrUrOzlW94RQuiqCpDRLi4LoDBfObZtGiyCLgpWjx5Jepjr3TdoFsGq+6g1VNyK/w1YucPMXSFHBpAlpBB1F+AsWUQmRUeQXr8OgyuGJ/HgAG0qg96n+zbCQajkEIPQAL1ZJ5woIJXT8KDTYfRUPmWPaQjxdYsGTCRkOYMLEIDEE4cUfRqxMGLkU0stzUKNiuFYbsw3gUbTAS4IFMbSdlGVVYGNjcNjr7jX9/Fl+di3OeikZ9nvAig5pkDvzhYxYTb9RfAPmgaJuSQMLfeArsOBlRYbEfCkVV46tewxi++kdwfAx+wsdmAoA/49BM9HK2KNh90MFLLRzdg7uqT4ALsGPAHoDblWgFu6PBM1hYlCNlDKYcb5dF3DsBSxyG7fwCuOhV4M95BpuAaM2tAGydzgWwFMgKrjEOXFy9pJeyaI29/+3oV6aByY3j7Hvunb7HTOb57PT0yMDmGviw4MFYg7XwiUYMPSIDpJdHabGmgcoBeCG6YzdO4DmuwcBV1IinQFSARwAJfxpNpsBMaJMBPPDS4gIWQD95h22GuF0EcH2FwWlaJBVRYWBDVAmOB2NvNnZF3ZaOWKUg1/K2COcW+X0uvjvy+cN7nkCCCNa7OD36gyePT8X3mfg+Ed9fz2VC5px8EYkO//5DVL0QUI8/Hz7rdrlTKuPu3a5rTT172O2h/xl2uxpvu7Gen66d8Ux7lp1MuyBVQMSRHUET84l0eTLmdqrUQkbGU7lIVONZfirOFHNCIa7jrhVcC/IlyZMk+fVcSSv5J194erXZttpOiynAaOX6FMwC+LPHkWqnCpxhgTMscAYFBKonmHmCmSeQeZKv9fUcS8Dfn+nvXtybr1T1K9X9ipVl106+EMwvBPQLQv2SAps2p7O2obCus1aUAVx8gfkXoDgufk0Do2jT2aev8J5k+yVrvNkuKHJsNnCKEhasqVizTZibXrLnAKOg2r9/+/gJ6mHRQ7T7i9oG1ZG0DnbD2ac/FGzwGaQtyQHn6PxT8gi+yYfTzxf/p4Aowv/8U7dzAYCSYs0GAurS3rY4cxtdr0+ZkqA6CiB+/tL5N5KEkIO+IUlEuqmkW5I8+IwpQB6/CGkEOw1Rl36aRjhGuPvNs8Ywl0YYaPO9/gg1p+XuIyWFM2qkJRMGGashLIRSQSnEEI0DjOIEk8CJQAWE0T69PNRQRmmGcBbw2iNvHeN6tMRH8VtQKGBlwEwQUjCX2fCe4dLR5RXWK6wACjmOEIUYhUqLwoJvdwFy/XEaU4snm0nyjQ6Mj8xagZmiwrQT9pppxx2Uhg4Iw3Uw6r8fjWt3OtmqUKIBuXWOU0pVhYsiAqQLHWrYJT2NZ2DNJabHn6EBENsKimsFJ80Kil2FvTEb6Upxlbh2VfazJWS/QoIuoaDX2ai3dhF7vi5IajicF2FN7ziyAJcTUi4mKn4m8ZM0UZLgXeyCpXQ+2jjIbUO07n7rL9BDmMy57YP25huDDALIgNzXNBlF1/xpi57I3uPF0aZtZ8B7UUrzyd5f/Ap/mjhoND/jWC6yVc9QFQmyAxZozxIqWwwnz/mNfH4Nz9m6HWw1zy7ZYr/+oahz4DBqjUYG2T6pmMPtV6gHlSvQEIzi1+zra6SgRl1Dmt5BIShdZXHOQqd6kIVdyFYPyF9N6i84/0HhQCcEsxXseq49IDOUv8HyJ1+y5S9+R5N15GnovoOYGUxDloLpEHdF6e2CiRtDyCXCvfG68kuSfQEqtapKtq90RRS+RTWauPiiZLn1ZOmskm9DFTHOAsqVuBvSCdxcv9AfCZeo7xaqocRL0rQOe4VUwrFD/ih89fU8xodtsIRLi6DmplXFUipV5j0RDCy704oVaQsDHb+T59jYnW7QkLrqKWs9lAwPKIuW0usvGPo1U0Eu7MNucMRwUuhIhGLfaiYdtxnGkeIpBGq8ZruYm3LhMsoQo7qoCHhpHexfytmKc9qZWtz0BacMyi1Q2wvXDIULpuexZIoTVfrvOD5VgA7IJSA5sH6IaIQILcKBpJK7er54EQ2LCVfA9qG1iTQDTkrXs+tm9kQsstrXYmlQbKYWK6h8bRCsMVwM5hvuaJXUhURg55b7BIfx+aqSmwanYh5EnEr0Abe3YytZzl1PvWWd9MfgYfOzB7nlbQGcUO/SogefWkNNiXopCnF818Xps774RfybFqlGtLm5xajLGmX2wYN3pbAkTEo051ZpzgZ0ct7f/Bbn1Em2jpZYPM2GLuqrcwQvzEJ06/qInKz+a662Wi1jqi3qGRmCWa2Cjj7kkJXxEj2MCo0VuBBvQbGh3URhME4z6gSYXgHa0ZsYUMDQ08jH9Yprpys3ax+wy5dXpFIwgsW9zXCUCFVmiD6zQz7noAOkih5OSuEIlNHJlyXh/owG+3ZQb7Nv+zDNs2r1m768hrpQgHXc/dyywBLpB9k5KC38kqIYhNSQFpWzr1W0W/TK+QUgWDn7CE1Wmo2qYKyipoVhG9gH89uqHOFwVBFMU9W+xWCyAJ6qbFBRrNM2qmZQU0/QARzIQ3vlmsctjtvt/jGFAW/1NAtqt8atAEcLW/PLzsXV5Tl8zj5eJZZViy1m9OYUPo0rhaExxlbMx0s5MeFCHqP8Boy7jhOLlqtiTsyJibTNuJygdIBAP0dLBZRvdC3stfklYnt1+e3q0r0qpe/mGuivRHugJSQ1EuR+40oIlJeuaqKsGD/HmEPinIt5oZ7MC9tJ8m2SNOv6CtmO6eNx+nhkmSB9vKI5OqsX0fsC5Zg2LjSgFNrukO3pZONINQkmfEPPmB4haVKV3YFlOIG9q3KbnnEeXUz0ZTDM8jCEUyx0OHmBu7o6828iQCEnaQgk7RBJgbWR7bXQYFK5b7LZd2ywOC6yiYm1RFHQVhxSE4DlEkWxxE5IlhiYZduXZ0dc9FHIW1ftsiupcu30vyDtBdxcYnazeS0barWwEmhk7N6QRB0YyUZGqtcGGc9x4Lv9xHOMs71FVGg+QUPOZJmnqZpTHIDYqJLfpILWlkGxmUzXmJkm+Yh3i8/z+A1uRudivZot4t8i1yTEBkYoyikHQOvw0Cv2B4qATKGV0QNWulnmKGCvS5kh5Vli3SSQmdBpLP9U1x81UPdC19cW7E+mPN6lH+/Tj/OUQhKmVhWXr7i9FasCnpsq/Xi1oPgnqvpOYiUkpbtXGdwK5hmhU3YSnbKrzCqmvnKoNphZ/ruzyyg3vaRtzha7rx+UnGAMdm+WLFtdOY3wfz0+wR3c13GLXo9PVfh0J54a9HQvnpr0NC+E0oJ3ZgzlDT1JKNv0JKHs0NN8mXn5dOZLQyvriZSbp7bqwnT2XgCsV2bGYi7DjRt4MBv4MPkYBngTm3oxVQ2dmNHW8Wd9MzeG2+Av2ewgUT/5MrlT5cJFmbXo8Nv6c/LJMfnklHzqkHywzo1J80Z5Pkm7HkQiIxxVgVDtIh13xB0RsW+u/bgJJTPGaQQF95ZjHguYx9o/O6qA52dVq7ifBYbIujqoL5qD3Tp+phl55Ap8mQNSbt8hra3K2cugRg0mTxnjYvMXMyO2JQi1XplnCfVHeUI5HLgDwHcRtqNzneWQ3bWeWEq7G3pHhYoZZ58Zko+eFB2deZGo68yLRHNnXixT4mox80rR55kXiWrPvEi0fObFCoX/8DdZKQWmb1mTYr245hTyFHfuHSRLnrgCc02HNnAhJbbSP0zTaydkP5yjsfEtp8vooCQa6bgytUtHJLGVotmh7xdkcgh8lfKAOqYn8JpiI56ul9v3ty/U3aHGBR3gFMwTDKVqUpTPnQ0H933BZPDzOrfDmaCazzk3abkUpyqKaLRqXig7S4ihLFBAWqwqy6qeDT2KdKCM170llxZXs3jArIDI4fxS02hlOGYqudSOseRbXQ2DgwgXHnCg/Xw44dNIX95WWkK7Y7Jqqg8N9aF5pa+y+gpd/QxNn//dNC2UImV/Pa6wbUbkIlmI6XowQx2s0BYy7toqfSHjvq3SGDLmBbLwaPbeLna/6mVVpUJIyZflRm+jEEqJkc+TucylPhh/qYv4C+dlHEIcSkNMupXbqyQJQ5S2aYpsAyVA9jgKlpYAT+ym9VCtauSzgs/yGs+WAD0ny/0IWvH1cZeB9Ha5o+tNSMkr6047uOqlyR1vMJPh/CXd4RY5yAXzgLI/w8egiDa/od0Y5DHrcm3rNm4shkjTEnnXu+11PpDV72+0L4OvK/s3cldVJ15c9t2CPqi3OfVMo9cwes0Czyg19IJORdyYdY/GVhSMFpnLoHa93dqiIT2nHhCHu03kMDUUh4nftnTAr1uSZ+uBHnpBR5BFn4RQK4/QEj8uhZZMNeJUM4vqEwS+UFVvKPLI7Y283IPv0jMLrZvcaOOeoUe1SseQci034NPU1+n3lfp+xXGjHxTlLBNwS4RnkQkGFk5Eh6hcwGSG5L6wnp8L126dAtsgZrdMo+TiguvEfPWmnQK+oVBjqK7wF+mhYi2Z2gHA9/Xm5bsgmMlbWBGkZHGAEgOPT+tzoTTe47aG/C4WJThaOcwurMV1zVJ1Uwe2CqRgCVJlV/w4Gj9mbW+ZLDOUkSRMu1CCtAslRLtYFqCl+kmAdqGEZxdKcHZRHJpFfMoZG4V2ymqeoX3ta82Ztet3RV6KXEgh0zg5Fwjft1fsz6JsfvBz2asGvCo+760a7blqzUIDP17lWanbfqYB5hjvCROTnnTAsmxV3iu1OuVsJhOPtpSJDMaj+2k8uZslZmk8NFB+jgY/DqaKbD/JSv0ndvbhR8Y5lrromW0JRftYsyH1R28KKFqxLV7LpWMr2e1mfMt+Zt+zYo/g3k8UTzC0RAqEzZzy7a7isQGP/eSxmQo+ZZyQm9Th7zH7i+Hh1vXGl6JNt4sCOtnLtvo110+u18IP2Itbb+juGuWluF2w+GVDXOYjX75YrvBepCo25Q1AVJGs/aq8zKsq79jFhC0T8YVfN+Kun9s094o9DRaQtwcfGz79ZfxWMPJ0OCk79IqhRAywQSSZ33eBZ4aqdAQh5QQLG7LbxWHuWiE/7KpRF17SuoG8tS7ZAJfVtYgf7zOuNLx1mjs7b3tvS9xzF2NmMDM+TatsQo8PueQa5ftngnU7Z/4pO2ECdRMMHZn7malzDr6ospyKxuy9bMHiDW6xNbJ+UBBYXazRmHJDTqz5VCfk/J08qiMOz7RzRQJ/jjZn6kzL+Tv14AY/GAfmtDy6QIoUT0hiMKHM9rqNfLnHxuBWDd/3Z6vdtCIXDac2RWhB3myK/izdRkuHsPEu06gu465ZOUWHAMHa6nmedKBZnm38Wd6UgzKNsq0lPKhneAJ4DPuxz8Z5c3QT5ZJ2zs5Q3fhzQ2H+pTF0VdKXaUZ+wlIdjNev2S9T1+WXFeyJw/0aTiqp0Yjj1TV27I8nls13zIYU4zD1bMWL3zGSJofxupbsGcatt3gKvzvAVpP9t3QoX14aULTzk3ZBID9iS7gyBURx6DgIZvDjeB7tbIfMLX5yi7ARWXjI/yI/T4uLi2Z88fM73f9E1zvRRU2z2vyh/Th1leb3LLd7KQ73VvJ3lrtJXVSZILXgb68Ed9MiimDSH8GUQLUMPyrbxNsFY24L1tlw2K/jUd/JDjoYHVEy6Be/Z8f84vd/zJDjUEdPGOPrf+LYZgwvOTklxwTNNFnkXSR4Hq/oGIp4nzmIgsc11RMpWZqd/ZaaY/nJNjws8T+NcWb0PnbxkjZUEX1zC4KIZZVlZr01i6/qVSpySsCDHTJtKumGks4sva5Yai0e0Cxvl7x/Bklg0HUzeEwZv+mGGbxdJp4DipbL0o1JnrHDKd5BQayT8O3Z1/yw/1hUl+kqQugSb5cRF97lZDr7azzxFavCw7VxBiwqFg1HYVeUzS6F2y4twWjLNQiZU7ibA4qSUZk5CpytgLfXQA2QoioTR3WV06PZ0o4pgupIJA0aq/L1OICSaSFzZVBww0uRXotvD3LMgtuDsoNP6+oV6M7YmsiLFahPxBNVPjT1nDGXcqZN7kx3RGiy7KVC61q9zUddYt5WrxpaFoIpHM6HVVupVb2Hs5BymjmvA+lCFZkJZllm3TR9vlea2gZNubHN3C5Q4IFjz56qwMH2vBg6zAUt5AT/Cpk/9+IrsHCXjrjeiFNeXELUpmuRoN2xHzns+OIoZJNpOOS3bMU3Gq9TaJmD7Jspt/i25/wMHA8F7pYEZsMz5Wv2hQXlLg+g29jpKHt93bLoD9n/UmYNL8NAXoEXh+tT0ZIQbJpa3hJyZZvhukTheU2bBxtMrLg9IVuDuyux23ndLl5REKzPD6GTACTTEL9bIXW3BEonKYn2ii4tuSftCZN7msbxlQ+AEUd8VlyY1Ez5S8dL8etGZ+iRsw28vjdILsdYtzD2sEKxxr3nm8bS7nGJLWelb5LOo1uoc1p6WVsEWQttENugO1S4QWKvNZ5KjuFGjQiPrbyRFhv2KenocHEoqzD/FvW2REkVajSOaaEccbNfGv03sR4oVjqF0GM9tGwH5g90BTbahfnImfFHzHhPMOlto4C7SwQg1up8ofKX6vmS+vcfrXkfCqycv0mrPkKxlddpS5cyxSWh4r5u8q/K0NafROAicYZS6uI8/8qUv3KE1/DEvjW+7juz5GAu/kjsr070nn4NQXsJ7/TkBc+F10dRFIx60whKQLfFK8w8tRYdspxPafn0sxO8Q1/wEyJmMAntIftrnWCU306d4I7VGb9AE21uU5QJQe0sGP50w8xhwj8NmSZSooLOQMuS38lvTe2ctsHglgAC5zqsmPV6vYo/AOvyn3j8Z18tusHYiztH6Yex6Jo/vChUphdJktbr5ENy06foibi+U97uKW/2lLd6GssmBaJQAjeqm8nDVH0IIxBO0RqNtvh9GIzU4j2zZX68whnP9mmc8IPjg2ND48LHRIwHjgWOA44B0Z9oT3RHmiO9kdZA5/R9gRPfHdl3gAjnxtcMeAZL8MetA/mMcsTz5KXxUkR+gaGIpSNkAyveP6DJ16r8hNrLgWVQ93XUblimM7VtJwylngPRFbIpngdWzZuOz4GlMUoLTI2zA0Ze6GbK8cjb51PFoQYlid878PwOWd5gb9++jUWQLI3jU1zVG2Pjx6c6Lfcdn2HWGWWdiawTzDqhrBNdQa0R/3gaXieI64PnVOjruaj49USsGlLuiQT3heB94QC/iNwOZnYor5NrZNlFhHSjIQXkxCWGRvHyJyIJefv8ilRASKSg/iHnG6VFWeGMbQG05EXSXcHVekdeQwmdob2RgP1JUqEZV+hc4Kum6PcWXUYAzUN6n9KyRkvWGEsMx4CguCKWdrbHRd8kdKGLTym0h9btc6HPEcxzyJVkNZO6b+V7/lrrQGsUoegw+iHWTMnMXYWS1cSFhYRbcmnha8avc5H3UI2Zio8mL5DdjU8hKlY5TE1UHmh6SHv7i2KukhZvzPieUZacHk1WAQhS7C3T/Y7ImEVwimFkYjYpePyqyeL7/STg1B1/ShNpxHCvLYpGJRs2E5WksK7Z7QpAqwlUhWMye3orLGbjeJdVum7mctWCn+bAS1JpiFP16GZWZaAy1yAnlFZmN5jcu/07L2WjyDzFUlmhXNcpVl4qaWcjW+oV160D+bOCsS5Nt91RjSJs1chjDMbR+zvPGo9s1UZCgIbETlemC1ffCPqZ7332g8jquc4xN40Q8pHr+nOnL2+yfMiqrxCVi2w8Fnl+TzO08JFMi2PfG4yu8fdeMfT7/SH9NsaAWZG47sq1a+KqVijfH42bGl7nev6OrGOOCRSRV83KMmgsm8r7viAWDMc7cY9vOBbvuDGBQkRTJj1hdyiRqDbJPYBZ/apGV+sWIC7oBag4XjjyhWvB68wsV/7MGt08S0H95e/ReOiseH+f1IdeWIABdsuK4irmWjQ7YB73p+7Iu/5M5srnwBk4Ad4JqgJR2l1egTQ5SqLYxw5tLmnkUOO2kR7jDjbIkUC/obJWbBc1cbmK/A+8RwJhfwgCP0Db8wCfPtn2dGJ59h3IxRFZ9+QrhOA0nE7daDQJfLSD/ACdDGJ6YJBwrLclmzoHB4q99PNs75WJv6eUxaMV8zW3ksY5C85U3aGMMxTjjGPignIvRl2QLSSHiEvoS9fWlxnNe2xjCgioKCThWE95ax9h8AhP6fhKAmHnFBIB9jNoGskkBizhshP6pYcPCw3Qlm0tOeggzjnM8W6YJUWyF3ovlv1uWa4+3blKd6Evhx7/eMEhXqmJJ2kM9F/AGERbH43Cpf6KcsodPBXaKSmPeeNeaOX+YHlpuo7lcNdHI11gmqu+tsUE5jSGqf58YX448wOIRx1f7ZLXXN435FEnIF3ijf0vfMSnuobhfBTZQxTbkFZpgQUcVt+TPy9c0i3eN42WYcL/OvxvvNmGv636221j13zbgNxdw9yGrx0sgkxtLAwkAvZ9bJwZJwb0E7oIvesY2ANEXODLUW1zvMyN8ar/t/GiSxgc+knLvcScfyt/hOY/MQ8H0Q==')
b2p_tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2p_tt_v10_m32x_m32x_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def b2p_tt_v10_m32x_m32x_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def b2p_tt_v10_m32x_m32x_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def b2p_tt_v10_m32x_m32x_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = b2p_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def b2p_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = b2p_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def b2p_tt_v10_m32x_m32x_table(base):
 return base * 16843009

def b2p_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [b2p_tt_v10_m32x_m32x_units(hi), b2p_tt_v10_m32x_m32x_units(lo)]
 else:
  mats = [b2p_tt_v10_m32x_m32x_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([b2p_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = b2p_tt_v10_m32x_m32x_c12_bytes(v, decode)
  if not torch.equal(b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
   return None
  if decode in (1, 2, 4) and (not 1 <= base <= 119):
   return None
  parts.append(data)
  if decode in (1, 2, 4):
   scale = 127 + 2 * (base - 64) << 7
   tabs.append(scale | scale << 16)
  elif decode in (3, 5):
   tabs.append(base * 16843009)
  else:
   tabs.append(b2p_tt_v10_m32x_m32x_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def b2p_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, b2p_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > b2p_tt_v10_m32x_M32X_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = b2p_tt_v10_m32x_m32x_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.r3_pool32_pooler_d9_cap2_a_m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2p_tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2p_tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def b2p_tt_v10_m32x__build():
 name = 'r3_pool32_pooler_d9_cap2_a_m32x_pooler_' + hashlib.sha256(b2p_tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
 return load_inline(name, b2p_tt_v10_m32x__CPP, cuda_sources=b2p_tt_v10_m32x__CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__make_plan(ext, weights, fmt):
 pack = b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
 if pack is None:
  return None
 stream = b2p_tt_v10_m32x_m32x_stream(pack, 1, *b2p_tt_v10_m32x__CFG)
 if stream is None:
  return None
 return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=b2p_tt_v10_m32x__CFG)

def b2p_tt_v10_m32x__run(x, plan):
 out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
 rc = b2p_tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, b2p_tt_v10_m32x__CFG)
 if rc:
  raise RuntimeError(f'm32x launch failed: {rc}')
 return out
b2p_tt_v10_m32x__W_CUDA = _zsrc('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')
b2p_tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint r3_pool32_pooler_d9_cap2_a_m32x_w_info(int);\n'

def b2p_tt_v10_m32x__w_c12_bytes(v):
 return b2p_tt_v10_m32x_m32x_c12_bytes(v, 5)

def b2p_tt_v10_m32x__w_c12_unpack(data, base, esc):
 return b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def b2p_tt_v10_m32x__w_table(base):
 return base << 8 | base << 24

def b2p_tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
 (n, k) = w.shape
 total = wk * split
 rw = r // 16
 if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
  return None
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
 esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
 (table, max_esc, escapes) = (0, 0, 0)
 if compressed:
  (data, b, entries) = b2p_tt_v10_m32x__w_c12_bytes(v)
  if not torch.equal(b2p_tt_v10_m32x__w_c12_unpack(data, b, entries), v):
   return None
  table = b2p_tt_v10_m32x__w_table(b) & 18446744073709551615
  (er, ec, el, ev, xor7) = entries
  escapes = er.numel()
  if escapes:
   boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
   rank = torch.bucketize(ec, boundaries, right=True)
   seq = ec - k // 64 * rank // total
   group = (er // rw * total + rank) * rw + er % rw
   max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
   if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
    return None
   order = torch.argsort(group * (1 << 14) + seq)
   gs = group[order]
   ss = seq[order]
   pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
   entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
   esc[gs, pos] = b2p_tt_v10_m32x_m32x_to_i32(entry)
 else:
  data = b2p_tt_v10_m32x_m32x_bf16_bytes(v)
 payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
 return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
 return ext.r3_pool32_pooler_d9_cap2_a_m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def b2p_tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
 (n, k) = w.shape
 eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
 out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
 for start in range(0, k, m):
  if b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
   return False
 return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))

def b2p_tt_v10_m32x__w_build():
 name = 'r3_pool32_pooler_d9_cap2_a_m32x_unpooler_' + hashlib.sha256(b2p_tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
 return load_inline(name, b2p_tt_v10_m32x__W_CPP, cuda_sources=b2p_tt_v10_m32x__W_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_w_launch', 'r3_pool32_pooler_d9_cap2_a_m32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__w_make_plan(ext, weights, fmt):
 pack = b2p_tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
 if pack is None:
  return None
 return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def b2p_tt_v10_m32x__w_run(x, plan):
 out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
 rc = b2p_tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
 if rc:
  raise RuntimeError(f'm32x unpooler launch failed: {rc}')
 return out

def b2p_tt_v10_m32x__w_verify_random(plan, weights):
 for m in (9, 16, 17, 24, 32):
  x = torch.randn((m, 4096), device=weights.device, dtype=torch.bfloat16)
  x[-1].zero_()
  x[-1, m * 131 % 4096] = 16
  ref = (x.double() @ weights.double().T).to(torch.bfloat16).double()
  out = b2p_tt_v10_m32x__w_run(x, plan).double()
  diff = out - ref
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
   return False
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
   return False
 return True
b2p_tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
b2p_tt_v10_m32x__Q_CUDA = _zsrc('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')
b2p_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.r3_pool32_pooler_d9_cap2_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2p_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2p_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def b2p_tt_v10_m32x__q_build():
 name = 'r3_pool32_pooler_d9_cap2_a_m32x_qkv_' + hashlib.sha256(b2p_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
 return load_inline(name, b2p_tt_v10_m32x__Q_CPP, cuda_sources=b2p_tt_v10_m32x__Q_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__q_make_plan(ext, weights, fmt):
 pack = b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
 if pack is None:
  return None
 stream = b2p_tt_v10_m32x_m32x_stream(pack, 1, *b2p_tt_v10_m32x__Q_CFG)
 if stream is None:
  return None
 plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=b2p_tt_v10_m32x__Q_CFG)
 x = weights[:32]
 out = torch.empty((32, 2304), dtype=torch.bfloat16, device=weights.device)
 capacity = b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG, 1)
 attr = b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG, 2)
 if capacity < 144 or attr < 0 or attr % 1000:
  return None
 return plan

def b2p_tt_v10_m32x__q_run(x, plan):
 out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
 rc = b2p_tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG)
 if rc:
  raise RuntimeError(f'm32x qkv launch failed: {rc}')
 return out

def b2p_tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
 state = {'plans': {}, 'device': None}
 configs = [('pooler', (4096, 8192), b2p_tt_v10_m32x__build, b2p_tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), b2p_tt_v10_m32x__q_build, b2p_tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), b2p_tt_v10_m32x__w_build, b2p_tt_v10_m32x__w_make_plan)]
 for (layer, shape, build, make_plan) in configs:
  if layer == 'unpooler' and (not include_unpooler):
   continue
  w = weights.get(layer)
  if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
   continue
  if torch.cuda.get_device_capability(w.device) != (9, 0):
   continue
  if state['device'] is not None and state['device'] != w.device:
   continue
  state['device'] = w.device
  ext = build()
  for fmt in (1, 0):
   plan = make_plan(ext, w, fmt)
   if plan is None:
    continue
   if layer == 'pooler':
    ok = b2p_tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__CFG, 32) and b2p_tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
   elif layer == 'qkv_gate':
    ok = b2p_tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__Q_CFG, 32) and b2p_tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
   else:
    ok = b2p_tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and b2p_tt_v10_m32x__w_verify_random(plan, w)
   if ok:
    state['plans'][layer] = plan
    break
 return state

def b2p_tt_v10_m32x_run_m32x(x, layer, n, st) -> torch.Tensor | None:
 expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
 if expected is None or x.ndim != 2 or x.shape[0] != 32:
  return None
 if (n, x.shape[1]) != expected or x.dtype != torch.bfloat16 or (not x.is_cuda):
  return None
 if x.device != st.get('device') or not x.is_contiguous():
  return None
 plan = st.get('plans', {}).get(layer)
 if plan is None:
  return None
 if layer == 'pooler':
  return b2p_tt_v10_m32x__run(x, plan)
 if layer == 'qkv_gate':
  return b2p_tt_v10_m32x__q_run(x, plan)
 return b2p_tt_v10_m32x__w_run(x, plan)
b2q_tt_v10_m32x_M32X_ESC_CAP = 32

def b2q_tt_v10_m32x_m32x_units(w, layout=0):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 if layout == 9:
  v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
 elif layout == 5:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
 else:
  v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def b2q_tt_v10_m32x_m32x_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def b2q_tt_v10_m32x_m32x_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def b2q_tt_v10_m32x_m32x_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = b2q_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def b2q_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = b2q_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def b2q_tt_v10_m32x_m32x_table(base):
 return base * 16843009

def b2q_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [b2q_tt_v10_m32x_m32x_units(hi), b2q_tt_v10_m32x_m32x_units(lo)]
 else:
  mats = [b2q_tt_v10_m32x_m32x_units(w, layout)]
 if fmt in (0, 2):
  data = torch.cat([b2q_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = b2q_tt_v10_m32x_m32x_c12_bytes(v, decode)
  if not torch.equal(b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
   return None
  if decode in (1, 2, 4) and (not 1 <= base <= 119):
   return None
  parts.append(data)
  if decode in (1, 2, 4):
   scale = 127 + 2 * (base - 64) << 7
   tabs.append(scale | scale << 16)
  elif decode in (3, 5):
   tabs.append(base * 16843009)
  else:
   tabs.append(b2q_tt_v10_m32x_m32x_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def b2q_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, b2q_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > b2q_tt_v10_m32x_M32X_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = b2q_tt_v10_m32x_m32x_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
b2q_tt_v10_m32x__Q_CUDA = _zsrc('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')
b2q_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2q_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2q_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def b2q_tt_v10_m32x__q_build():
 name = 'm32x_qkv_' + hashlib.sha256(b2q_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
 return load_inline(name, b2q_tt_v10_m32x__Q_CPP, cuda_sources=b2q_tt_v10_m32x__Q_CUDA, functions=['r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
b2o_OLAYER_CUDA = _zsrc('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')
b2o_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_o32_o_direct_a_ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def b2o_olayer_source(configs) -> str:
 ns = sorted({c[0] for c in configs})
 functions = []
 for n in ns:
  regs = ','.join((f'%{i}' for i in range(n // 2)))
  outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
  functions.append(f'OD void r3_o32_o_direct_a_ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
 cases = '\n'.join((f"case {i}: return r3_o32_o_direct_a_ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for (i, c) in enumerate(configs)))
 calls = '\n'.join((f'if constexpr(BN=={n}) r3_o32_o_direct_a_ol_mma{n}(a,aw,ax);' for n in ns))
 return b2o_OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def b2o_build_olayer(configs):
 for cfg in configs:
  (bn, bk, st, pipe, pack, hint, *groups) = cfg
  tw = groups[0] if groups else 1
  if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
   raise ValueError(f'unsupported olayer configuration: {cfg}')
 source = b2o_olayer_source(configs)
 name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
 return load_inline(name, b2o_OLAYER_CPP, cuda_sources=source, functions=['r3_o32_o_direct_a_ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2o_pack_olayer(weight) -> torch.Tensor:
 columns = torch.arange(64, device=weight.device)[None, :] ^ (torch.arange(64, device=weight.device)[:, None] & 7) * 8
 tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
 return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()

def b2o_prepare_olayer_candidates(weights, configs) -> dict:
 weight = weights['o'].contiguous()
 return {'enabled': True, 'ext': b2o_build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': b2o_pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def b2o_run_olayer_candidate(x, st, index) -> torch.Tensor:
 pack = st['configs'][index][4]
 weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
 out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
 rc = st['ext'].r3_o32_o_direct_a_ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, 0)
 if rc:
  raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
 return out
BUNDLE2_ROUTES = {'qkv16': (16, 'qkv_gate'), 'pooler32': (32, 'pooler'), 'o32': (32, 'o'), 'unpooler32': (32, 'unpooler')}
BUNDLE2_UNPOOL32 = 'decodediet'
BUNDLE2_UNPOOL1 = 'v16'

def b2_prepare_qkv(weights) -> dict:
 w = weights['qkv_gate']
 ext = b2q_tt_v10_m32x__q_build()
 cfg = (1, 4, 8, 4)
 for fmt in (1, 0):
  pk = b2q_tt_v10_m32x_m32x_pack_format(w, fmt, 0, 5)
  if pk is None:
   continue
  plan = b2q_tt_v10_m32x_m32x_stream(pk, 1, *cfg)
  if plan is None:
   continue
  idx = 0 if fmt else 1
  plan.update(ext=ext, idx=idx, cfg=cfg, n=2304)
  x = w[:16]
  out = torch.empty((16, 2304), device=w.device, dtype=w.dtype)
  capacity = b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 1)
  attr = b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 2)
  if capacity < 144 or attr < 0 or attr % 1000:
   continue
  if b2q_tt_v10_m32x__q_verify_onehot(ext, idx, plan, w, cfg, 16) and b2q_tt_v10_m32x__q_verify_random(ext, idx, plan, w, cfg, [9, 15, 16]):
   return plan
 return {}

def prepare_bundle2(weights) -> dict:
 state = {'qkv': b2_prepare_qkv(weights), 'pooler': b2p_tt_v10_m32x_prepare_m32x({'pooler': weights['pooler']}), 'o': b2o_prepare_olayer_candidates(weights, [(32, 128, 8, 1, 1, 1)])}
 state['diet'] = dd_decodediet_tt_mw_prepare_m32w(weights, configs={(32, 'unpooler'): dd_decodediet_tt_mw_CONFIGS[32, 'unpooler']})
 return state

def run_bundle2(x, layer, n, st) -> torch.Tensor | None:
 if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
  return None
 (m, k) = x.shape
 if layer == 'qkv_gate' and m == 16 and (n == 2304) and (k == 8192):
  p = st['qkv']
  if not p or x.device != p['wp'].device:
   return None
  out = torch.empty((m, n), device=x.device, dtype=x.dtype)
  rc = b2q_tt_v10_m32x__q_launch_raw(p['ext'], p['idx'], p, x, out, n, p['cfg'])
  if rc:
   raise RuntimeError(f'bundle2 qkv launch failed: {rc}')
  return out
 if layer == 'pooler' and m == 32:
  return b2p_tt_v10_m32x_run_m32x(x, layer, n, st['pooler'])
 if layer == 'o' and m == 32 and (n == 8192) and (k == 1024) and (x.device == st['o']['weight'].device):
  return b2o_run_olayer_candidate(x, st['o'], 0)
 if layer == 'unpooler' and m == 32:
  return dd_decodediet_tt_mw_run_m32w(x, layer, n, st['diet'])
 return None

def _achita_dense_prepare(weights: _AchitaDenseWeights) -> dict:
 state = _b2_base_prepare(weights)
 state['bundle2'] = prepare_bundle2(weights)
 return state

def _achita_dense_kernel(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 out = None
 if 'bundle2' in state:
  out = run_bundle2(data['x'], data['config']['layer'], data['config']['n'], state['bundle2'])
 return _b2_base_kernel(data, state) if out is None else out
V17_ROUTES = [{'name': 'qkv16', 'm': 16, 'layer': 'qkv_gate', 'module': 'v17_bundle2', 'owner_variant': 'push_d9_cap6'}, {'name': 'o32', 'm': 32, 'layer': 'o', 'module': 'v17_bundle2', 'owner_variant': 'direct'}, {'name': 'pooler32', 'm': 32, 'layer': 'pooler', 'module': 'v17_bundle2', 'owner_variant': 'd9_cap2'}, {'name': 'unpooler32', 'm': 32, 'layer': 'unpooler', 'module': 'v17_bundle2', 'owner_variant': 'decodediet'}]
SELECTED_ROUTES = [r for r in SELECTED_ROUTES if (r['m'], r['layer']) not in {(q['m'], q['layer']) for q in V17_ROUTES}] + V17_ROUTES
from concurrent.futures import ThreadPoolExecutor as _PreparePool

def _achita_dense_prepare(weights: _AchitaDenseWeights) -> dict:
 state = {name: w.contiguous() for (name, w) in weights.items()}
 device = weights['qkv_gate'].device
 hi = weights['router'].to(torch.bfloat16)
 state['router_hi_t'] = hi.t().contiguous()
 state['router_lo_t'] = (weights['router'] - hi.float()).to(torch.bfloat16).t().contiguous()
 state['router_weight'] = weights['router']
 torch.cuda.synchronize(device)
 jobs = [('sm', prepare_small_m, weights, {}), ('m32', prepare_m32, weights, {}), ('wide2', prepare_wide2, weights, {}), ('router', prepare_router, weights, {}), ('qkv', prepare_qkv, weights, {}), ('router2', prepare_router2, weights, {}), ('olayer', prepare_olayer, weights, {}), ('cps', prepare_cps, weights, {}), ('new_lt', tt_lt_prepare_cublaslt, weights, {}), ('new_mw', tt_mw_prepare_m32w, weights, {}), ('v10_sm2', tt_v10_sm2_prepare_smallm2, weights, {}), ('v10_warps', tt_v10_warps_prepare_warps, weights, {}), ('v10_tune', prepare_v10_tune, weights, {}), ('v10_m32x', tt_v10_m32x_prepare_m32x, weights, {'include_unpooler': False}), ('v11_m32x', tt_v10_m32x_prepare_m32x, {'unpooler': weights['unpooler']}, {'include_unpooler': True}), ('v12_smtrace1', tt_v12_smtrace1_prepare_smtrace, weights, {}), ('v12_o1', prepare_v12_o1, weights, {}), ('v13_smtrace8', tt_v13_smtrace8_prepare_smtrace, weights, {}), ('v14_qkv', tt_v14_qkv_prepare_qkvbig, weights, {}), ('bundle', prepare_bundle, weights, {}), ('bundle2', prepare_bundle2, weights, {})]

 def run_job(fn, selected_weights, options):
  with torch.cuda.device(device):
   return fn(selected_weights, **options)
 with _PreparePool(max_workers=4) as pool:
  pending = [(name, pool.submit(run_job, fn, selected_weights, options)) for (name, fn, selected_weights, options) in jobs]
  for (name, future) in pending:
   state[name] = future.result()
 if state['qkv'].get('large') is not None:
  state['qkv']['is_enabled'] = True
 for (routes, fn) in ((V9_SELECTED_ROUTES, candidate_v9), (V10_ROUTES, candidate_v10), (V11_ROUTES, candidate_v11), (V12_ROUTES, candidate_v12), (V13_ROUTES, candidate_v13), (V14_ROUTES, candidate_v14), (V15_ROUTES, candidate_v15)):
  for route in routes:
   (m, layer) = (route['m'], route['layer'])
   weight = weights[layer]
   n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
   data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device), 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}
   fn(data, state)
 torch.cuda.synchronize(device)
 return state
v18d_SM_CUDA_TEMPLATE = _zsrc('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')
v18d_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18d_native_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
v18d_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
v18d_SM_ESC_CAP = 32

def v18d_v18d_native_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
  cases.append(f'    case {i}: return v18d_native_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return v18d_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def v18d_v18d_native_build(instances, name='dd_cap2_b_small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, v18d_SM_CPP + extra_cpp, cuda_sources=v18d_v18d_native_cuda_source(instances) + extra_cuda, functions=['v18d_native_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def v18d_v18d_native_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def v18d__to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def v18d__to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def v18d_v18d_native_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = v18d__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def v18d_v18d_native_c12_bytes(v):
 (r, c) = v.shape[:2]
 dev = v.device
 lo = v & 255
 sign = v >> 15
 eh = v >> 8 & 127
 hist = torch.bincount(eh.flatten(), minlength=128).cpu()
 window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(window.argmax())
 d = eh - base
 inwin = (d >= 0) & (d < 8)
 d = torch.where(inwin, d, torch.zeros_like(d))
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
 code = v18d__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
 lo8 = lo.to(torch.uint8)
 data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
 (er, ec, el, ev) = (~inwin).nonzero().unbind(1)
 xor7 = eh[er, ec, el, ev] ^ base
 return (data, base, (er, ec, el, ev, xor7))

def v18d_v18d_native_c12_unpack(data, base, esc):
 (r, c) = data.shape[:2]
 dev = data.device
 p = data.view(r, c, 3, 32, 16)
 lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
 code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
 nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
 pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
 d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
 sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
 val = (sg << 15 | base + d << 8).to(torch.int32) | lo
 (er, ec, el, ev, xor7) = esc
 val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
 return val

def v18d_v18d_native_table(base):
 return base * 16843009

def v18d_v18d_native_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [v18d_v18d_native_units(hi), v18d_v18d_native_units(lo)]
 else:
  mats = [v18d_v18d_native_units(w)]
 if fmt in (0, 2):
  data = torch.cat([v18d_v18d_native_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = v18d_v18d_native_c12_bytes(v)
  if not torch.equal(v18d_v18d_native_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(v18d_v18d_native_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def v18d_v18d_native_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, v18d_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > v18d_SM_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = v18d__to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.v18d_native_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
v18d_SM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
v18d_SM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
v18d_SM_MAX_M = 32
v18d_SM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
v18d_SM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
v18d_SM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
v18d_SM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def v18d_v18d_native_mt(m):
 return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def v18d_v18d_native_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def v18d_v18d_native_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def v18d_v18d_native_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if v18d_v18d_native_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def v18d_v18d_native_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def v18d_v18d_native_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def v18d_v18d_native_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = v18d_v18d_native_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = v18d_v18d_native_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = v18d_v18d_native_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = v18d_v18d_native_verify_onehot(ext, inst_id[inst], sp, v18d_v18d_native_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and v18d_v18d_native_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def v18d_prepare_small_m(weights, table_m=None, table_mt=None, name='dd_cap2_b_small_m_gemv'):
 table_m = dict(v18d_SM_TABLE_M if table_m is None else table_m)
 table_mt = dict(v18d_SM_TABLE_MT if table_mt is None else table_mt)
 entries = [((layer, 'm', m), v) for ((layer, m), v) in table_m.items()] + [((layer, 'mt', mt), v) for ((layer, mt), v) in table_mt.items()]
 specialized = []
 for (key, (inst, cfg)) in entries:
  if len(inst) == 7:
   (layer, kind, m) = key
   fixed_m = m if kind == 'm' else 0
   rows = v18d_v18d_native_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
   inst = inst + (fixed_m, rows, v18d_SM_LAYER_K[layer], *cfg)
  specialized.append((key, (inst, cfg)))
 entries = specialized
 instances = []
 for (_, (inst, _)) in entries:
  instances.append(inst)
  if v18d_v18d_native_bf16_twin(inst):
   instances.append(v18d_v18d_native_bf16_twin(inst))
 instances = list(dict.fromkeys(instances))
 st = {'plans': {}}
 if not instances:
  return st
 ext = v18d_v18d_native_build(instances, name)
 st['ext'] = ext
 inst_id = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams) = ({}, {})
 for (key, (inst, cfg)) in sorted(entries):
  layer = key[0]
  plan = v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None and v18d_v18d_native_bf16_twin(inst):
   plan = v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, v18d_v18d_native_bf16_twin(inst), cfg)
  if plan is not None:
   st['plans'][key] = plan
 del packs
 return st

def v18d_run_small_m(x, layer, n, st):
 m = x.shape[0]
 if not 1 <= m <= v18d_SM_MAX_M:
  return None
 sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', v18d_v18d_native_mt(m)))
 if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != v18d_SM_LAYER_K[layer]) or (not x.is_cuda):
  return None
 x = x.contiguous()
 if x.data_ptr() % 16:
  return None
 out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
 if v18d_v18d_native_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
  return None
 return out
V18S_CUDA_TEMPLATE = _zsrc('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')
V18S_CPP = '\nint v18s_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
V18S_ESC_CAP = 32

def v18s_cuda_source(instances):
 cases = []
 for (i, inst) in enumerate(instances):
  args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
  cases.append(f'    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
 return V18S_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def v18s_build(instances, name='v18s_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
 return load_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda, functions=['v18s_launch', 'v18s_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def v18s_units(w):
 (n, k) = w.shape
 bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
 v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
 return v.reshape(n // 16, k // 64, 32, 32)

def v18s_to_i16(v):
 return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def v18s_to_i32(v):
 return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def v18s_bf16_bytes(v):
 (r, c) = v.shape[:2]
 words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
 return words.view(torch.uint8).view(r, c, 2048)

def v18s_c12_bytes(v):
 return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def v18s_c12_unpack(data, base, esc):
 return b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc)

def v18s_table(base):
 return b2q_tt_v10_m32x_m32x_table(base)

def v18s_pack_format(w, fmt):
 if fmt in (2, 3):
  hi = w.to(torch.bfloat16)
  lo = (w - hi.float()).to(torch.bfloat16)
  mats = [v18s_units(hi), v18s_units(lo)]
 else:
  mats = [v18s_units(w)]
 if fmt in (0, 2):
  data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
  return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
 (parts, tabs, escs) = ([], [], [])
 for (h, v) in enumerate(mats):
  (data, base, esc) = v18s_c12_bytes(v)
  if not torch.equal(v18s_c12_unpack(data, base, esc), v):
   return None
  parts.append(data)
  tabs.append(v18s_table(base))
  escs.append(esc + (torch.full_like(esc[0], h),))
 esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
 return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def v18s_geometry(R, C, rt, wr, wk, ku, s):
 if R % (wr * rt) or C != s * wk * ku:
  return None
 gx = R // (wr * rt)
 return (gx, wr * wk, gx * s * wr * wk)

def v18s_stream(pk, rt, wr, wk, ku, s):
 data = pk['data']
 (R, C, B) = data.shape
 geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
 if geo is None:
  return None
 (gx, nw, G) = geo
 dev = data.device
 si = torch.arange(ku, device=dev).view(ku, 1, 1)
 gw = torch.arange(G, device=dev).view(1, G, 1)
 j = torch.arange(rt, device=dev).view(1, 1, rt)
 (cta, warp) = (gw // nw, gw % nw)
 (by, bx) = (cta // gx, cta % gx)
 (wri, wki) = (warp % wr, warp // wr)
 tile = (bx * wr + wri) * rt + j
 chunk = (by * wk + wki) * ku + si
 idx = (tile * C + chunk).reshape(-1)
 wp = data.view(R * C, B)[idx].contiguous().view(-1)
 esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
 if pk['esc'] is not None and pk['esc'][0].numel():
  (er, ec, el, ev, x7, half) = pk['esc']
  (bx_e, rem) = (er // (wr * rt), er % (wr * rt))
  (wr_e, j_e) = (rem // rt, rem % rt)
  (by_e, rem2) = (ec // (wk * ku), ec % (wk * ku))
  (wk_e, s_e) = (rem2 // ku, rem2 % ku)
  g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
  seq = s_e * rt + j_e
  if int(seq.max()) >= (1 << 14) - 1:
   return None
  counts = torch.bincount(g_e, minlength=G)
  if int(counts.max()) > V18S_ESC_CAP:
   return None
  order = torch.argsort(g_e * (1 << 14) + seq)
  (g_s, seq_s) = (g_e[order], seq[order])
  rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
  reg = ev[order] // 2 + 16 * half[order]
  entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
  esc[g_s, rank] = v18s_to_i32(entry)
 return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
 if sp['fmt'] == 4:
  return ext.v18s_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
 (wr, wk, ku, s) = cfg
 (m, k) = x.shape
 return ext.v18s_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def v18s_rows(layer, m, n_full):
 return 1024 if layer == 'unpooler' and m <= 8 else n_full

def v18s_bf16_twin(inst):
 return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
 (n, k) = exact.shape
 mm = rows_per_launch
 dev = exact.device
 out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
 got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
 eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
 for i in range(k // mm):
  if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
   return False
 return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
 (n, k) = w_rows.shape
 is_router = w_rows.dtype == torch.float32
 tol = 0.0001 if is_router else 0.01
 gen = torch.Generator(device=w_rows.device).manual_seed(seed)
 for m in ms:
  x = torch.randn((m, k), generator=gen, device=w_rows.device)
  x[m - 1].zero_()
  x[m - 1, (seed * 131 + m) % k] = 16.0
  x = x.to(torch.bfloat16)
  ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
  out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
  if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
   return False
  diff = out.double() - ref
  rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
  if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
   return False
 return True

def v18s_exact(w_rows, fmt):
 if fmt in (2, 3):
  hi = w_rows.to(torch.bfloat16)
  return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
 return w_rows

def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
 m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
 rows = v18s_rows(layer, m_hi, w.shape[0])
 fmt = inst[0]
 pkey = (layer, rows, fmt)
 if pkey not in packs:
  packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
 if packs[pkey] is None:
  return None
 skey = pkey + (inst[2],) + tuple(cfg)
 if skey not in streams:
  streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
 sp = streams[skey]
 if sp is None:
  return None
 vkey = ('verified', inst) + skey
 if vkey not in streams:
  w_rows = w[:rows].contiguous()
  lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
  fixed_m = inst[7] if len(inst) > 7 else 0
  ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
  streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
 if not streams[vkey]:
  return None
 return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
V18_ROUTES = [{'m': 1, 'layer': 'qkv_gate', 'name': 'decodediet_qkv1', 'module': 'v18'}, {'m': 1, 'layer': 'pooler', 'name': 'p32_s1_o6', 'module': 'v18'}, {'m': 8, 'layer': 'o', 'name': 'warps_finalist_o8', 'module': 'v18'}]
V18_SMALL = {('pooler', 1): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1)), ('o', 8): ((1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))}

def prepare_v18_extra(weights):
 qkv = v18d_prepare_small_m({'qkv_gate': weights['qkv_gate']}, table_m={}, table_mt={('qkv_gate', 1): v18d_SM_TABLE_MT['qkv_gate', 1]}, name='v18_decoder_qkv1')
 instances = list(dict.fromkeys((inst for (original, cfg) in V18_SMALL.values() for inst in (original, v18s_bf16_twin(original)))))
 ext = v18s_build(instances, 'v18_small_selected')
 ids = {inst: i for (i, inst) in enumerate(instances)}
 (packs, streams, plans) = ({}, {}, {})
 for ((layer, m), (inst, cfg)) in V18_SMALL.items():
  key = (layer, 'm', m)
  plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, inst, cfg)
  if plan is None:
   plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, v18s_bf16_twin(inst), cfg)
  if plan is not None:
   plans[layer, m] = plan
 return {'qkv': qkv, 'ext': ext, 'plans': plans}

def run_v18(x, layer, n, st):
 if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
  return None
 (m, k) = x.shape
 if (layer, m) == ('qkv_gate', 1):
  return v18d_run_small_m(x, layer, n, st['qkv'])
 plan = st['plans'].get((layer, m))
 if plan is None or n != plan['rows'] or k != (8192 if layer == 'pooler' else 1024) or (x.device != plan['wp'].device):
  return None
 out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
 rc = v18s_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
 if rc:
  raise RuntimeError(f'v18 selected launch failed: {rc}')
 return out
_v18_base_prepare = _achita_dense_prepare
_v18_base_kernel = _achita_dense_kernel

def _achita_dense_prepare(weights: _AchitaDenseWeights) -> dict:
 state = _v18_base_prepare(weights)
 state['v18'] = prepare_v18_extra(weights)
 torch.cuda.synchronize(weights['qkv_gate'].device)
 return state

def _achita_dense_kernel(data: _AchitaDenseInputs, state: dict) -> torch.Tensor:
 out = run_v18(data['x'], data['config']['layer'], data['config']['n'], state['v18'])
 return _v18_base_kernel(data, state) if out is None else out
SELECTED_ROUTES = [r for r in SELECTED_ROUTES if (r['m'], r['layer']) not in {(x['m'], x['layer']) for x in V18_ROUTES}] + V18_ROUTES
dd_decodediet_tt_mw_CONFIGS[32, 'unpooler'] = ((32, 64, 2, 2, 2, True, 2), 2)
_achita_wg_base_prepare = _achita_dense_prepare
_achita_wg_base_kernel = _achita_dense_kernel

def _achita_wg_time(fn, flush, stream, repeats=13):
 fn()
 fn()
 graph = torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph, stream=stream):
  fn()
 samples = []
 for _ in range(repeats):
  flush.sum()
  begin = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  begin.record()
  graph.replay()
  end.record()
  end.synchronize()
  samples.append(begin.elapsed_time(end) * 1000)
 return statistics.median(samples)

def _achita_dense_prepare(weights):
 state = _achita_wg_base_prepare(weights)
 state['achita_wg_choices'] = {}
 return state

def _achita_dense_kernel(data, state):
 (x, cfg) = (data['x'], data['config'])
 key = (x.shape[0], cfg['layer'], cfg['n'], x.shape[1])
 choice = state['achita_wg_choices'].get(key)
 if choice is not None:
  y = dd_decodediet_tt_mw_run_m32w(x, cfg['layer'], cfg['n'], choice)
  if y is not None:
   return y
 return _achita_wg_base_kernel(data, state)
"SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit\na55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:\n\n  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)\n  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,\n                                   with the interleaved RoPE fused in on sliding layers\n  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache\n                                   (cast, then the store_cache JIT kernel at the step's rows)\n  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers\n  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)\n  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)\n  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);\n                                   it also writes the FP32 router input up to 8 tokens\n  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs\n                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a\n                                   side stream, overlapping the pooler\n  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)\n  pooler                           bf16_gemm_dispatch, replicated on every GPU\n  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)\n  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),\n                                   silu_and_mul, Triton post-expert norm weighted sum\n                                   (MuseSpark1xFp4Experts._forward_marlin); then the\n                                   all-reduce (identity here)\n  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then\n                                   the all-gather (identity here: zeros in the other columns);\n                                   above that, the whole weight (bf16_gemm_dispatch)\n  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention\n                                   norm (JIT)\n\nDiffers from SGLang:\n  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the\n    inputs; SGLang builds them once per step for all layers.\n  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."
from typing import TypedDict
import types
import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import MuseSpark1xNormPreservingResidualConnection, MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

class Weights(TypedDict):
 qkv_gate: torch.Tensor
 o: torch.Tensor
 attn_residual_gate: torch.Tensor
 pre_ffn_norm: torch.Tensor
 router: torch.Tensor
 router_bias: torch.Tensor
 pooler: torch.Tensor
 pre_expert_norm: torch.Tensor
 w13: torch.Tensor
 w13_sf: torch.Tensor
 w13_gs: torch.Tensor
 w2: torch.Tensor
 w2_sf: torch.Tensor
 w2_gs: torch.Tensor
 post_expert_norm: torch.Tensor
 unpooler: torch.Tensor
 post_ffn_norm: torch.Tensor
 ffn_residual_gate: torch.Tensor
 next_pre_attn_norm: torch.Tensor

class Config(TypedDict):
 batch: int
 q_len: int
 tokens: int
 hidden: int
 num_q_heads: int
 num_kv_heads: int
 head_dim: int
 page_size: int
 window: int
 rope_theta: float
 qk_scale_factor: float
 sm_scale: float
 kv_scale: float
 n_experts: int
 top_k: int
 local_experts: int

class Inputs(TypedDict):
 stream: torch.Tensor
 attn_input: torch.Tensor
 k_cache: torch.Tensor
 v_cache: torch.Tensor
 positions: torch.Tensor
 cache_loc: torch.Tensor
 cache_seqlens: torch.Tensor
 cu_seqlens_q: torch.Tensor
 scheduler_metadata: torch.Tensor | None
 k_scale: torch.Tensor
 v_scale: torch.Tensor
 page_table: torch.Tensor
 seq_lens: torch.Tensor
 local_start: torch.Tensor
 config: Config
HIDDEN = 8192
RMS_EPS = 1e-05
POST_NORM_EPS = 1e-08
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500000.0

def _gemm(x, w):
 return _bf16_gemm_dispatch_impl(x, w, None)

def _param(w):
 return nn.Parameter(w.contiguous(), requires_grad=False)

def _norm(cls, w, size, eps, **kw):
 norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
 if w is not None:
  norm.weight = _param(w)
  norm.precompute_effective_weight()
 return norm

def _residual(w):
 gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
 gate.gate = _param(w)
 gate.precompute_coefficients()
 return gate

def _marlin_bank(weights):
 return None

def prepare(weights):
 device = weights['qkv_gate'].device
 router = weights['router']
 hi = router.to(torch.bfloat16)
 lo = (router - hi.float()).to(torch.bfloat16)
 inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64)
 freqs = torch.einsum('i,j -> ij', torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
 pre_ffn = _norm(MuseSpark1xRMSNorm, weights['pre_ffn_norm'], HIDDEN, RMS_EPS)
 next_pre_attn = _norm(MuseSpark1xRMSNorm, weights['next_pre_attn_norm'], HIDDEN, RMS_EPS)
 post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True, post_norm_gain_center_type='one', residual_id=1)
 post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights['post_ffn_norm'], HIDDEN, POST_NORM_EPS, is_post_norm=True, post_norm_gain_center_type='one', residual_id=2)
 post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights['post_expert_norm'], D_EXPERT, POST_NORM_EPS, is_post_norm=True, post_norm_gain_center_type='one', residual_id=2)
 state = {'dense': {name: weights[name].contiguous() for name in ('qkv_gate', 'o', 'pooler', 'unpooler')}, 'router': router.contiguous(), 'router_hi_t': hi.t().contiguous(), 'router_lo_t': lo.t().contiguous(), 'routing': FusedMuseSparkRouting(expert_bias=weights['router_bias'].float().contiguous(), route_scale=1.0, router_output_multiplier=0.1767766952966369, router_score_division_eps=1e-15), 'cos_sin': torch.cat((freqs.cos(), freqs.sin()), dim=-1), 'qk_norm': FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS), 'o_norm_gate': FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64), 'attn_boundary': FusedPostNormResidualPreNorm(post_attn, _residual(weights['attn_residual_gate']), pre_ffn), 'ffn_boundary': FusedPostNormResidualPreNorm(post_ffn, _residual(weights['ffn_residual_gate']), next_pre_attn), 'pre_expert_norm': _norm(MuseSpark1xRMSNorm, weights['pre_expert_norm'], D_EXPERT, RMS_EPS), 'post_expert_w': post_expert._get_effective_weight().detach().to(torch.bfloat16), 'bank': _marlin_bank(weights), 'side_stream': torch.cuda.Stream(device), '_modules': (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert)}
 del freqs
 torch.cuda.synchronize()
 return state

def _experts(x, topk_weights, topk_ids, state):
 bank = state['bank']
 (num_tokens, size_k) = x.shape
 size_n = bank['size_n']
 gemm1_n = 2 * size_n
 local_ids = bank['expert_map'][topk_ids].contiguous()
 topk_weights = topk_weights.to(torch.float32).contiguous()
 block_size_m = 8 if num_tokens <= 64 else 16
 (sorted_ids, block_experts, n_padded) = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
 cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
 cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
 cache1 = cache13[:num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
 cache3 = cache13[:num_tokens * TOP_K * size_k].view(-1, size_k)
 common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
 ws = bank['workspace']
 cache1 = moe_wna16_marlin_gemm(x, cache1, bank['w13'], None, bank['w13_s'], bank['w13_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
 silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
 y = moe_wna16_marlin_gemm(cache2, cache3, bank['w2'], None, bank['w2_s'], bank['w2_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=1, size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
 return fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)

def kernel_fn(data, state):
 cfg = data['config']
 (batch, length, tokens) = (cfg['batch'], cfg['q_len'], cfg['tokens'])
 (hq, hkv, d) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
 dense = state['dense']
 qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)
 attention_data = dict(data, qkv=qkv.reshape(batch, length, -1))
 o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)
 attn_output = _achita_dense_project(o, 'o', state)
 split_router = tokens > ROUTER_FP32_MAX_TOKENS
 (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
 (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
 if 8 < tokens <= 32 and state['fpr'] is not None:
  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
 else:
  side_stream.wait_stream(main_stream)
  expert_input = _achita_dense_project(ffn_input, 'pooler', state)
  with torch.cuda.stream(side_stream):
   if not split_router and tokens > 0 and (state['rf'] is not None):
    (topk_weights, topk_ids) = _rf_route(router_input, state['rf'])
   else:
    if split_router:
     logits = _achita_wide_router(ffn_input, state)
     if logits is None:
      logits = torch.mm(ffn_input, state['router_hi_t'], out_dtype=torch.float32)
      logits += torch.mm(ffn_input, state['router_lo_t'], out_dtype=torch.float32)
    else:
     logits = F.linear(router_input, state['router'])
    (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
  main_stream.wait_stream(side_stream)
  if not torch.cuda.is_current_stream_capturing():
   for tensor in (topk_weights, topk_ids):
    tensor.record_stream(main_stream)
 expert_input, local_ids = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)
 expert_output = _experts(expert_input, topk_weights, local_ids, state)
 if tokens <= SLICED_UNPOOL_MAX_TOKENS:
  unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)
  next_attn_input, stream = _sfn_boundary(stream, unpooled, state)
 else:
  ffn_output = _achita_dense_project(expert_output, 'unpooler', state)
  next_attn_input, stream, _ = state['ffn_boundary'](stream, ffn_output, branch_dtype=torch.bfloat16)
 return (next_attn_input, stream, topk_ids)
import triton
import triton.language as tl

@triton.jit
def _achita_moe_small_finalize(Y, WEIGHT, NORM, IDS, OUT, H: tl.constexpr, K: tl.constexpr, EPS: tl.constexpr, BK: tl.constexpr):
 token = tl.program_id(0)
 expert = tl.arange(0, BK)
 dim = tl.arange(0, H)
 ids = tl.load(IDS + token * K + expert, mask=expert < K, other=-1)
 active = (expert < K) & (ids >= 0)
 values = tl.load(Y + (token * K + expert[:, None]) * H + dim[None, :], mask=active[:, None], other=0).to(tl.float32)
 norm = tl.load(NORM + dim).to(tl.float32)
 scaled = values * norm[None, :]
 inv = tl.rsqrt(tl.sum(scaled * scaled, 1) / H + EPS)
 weights = tl.load(WEIGHT + token * K + expert, mask=active, other=0)
 normalized = scaled * inv[:, None]
 result = tl.sum(normalized * weights[:, None], 0)
 tl.store(OUT + token * H + dim, result)
_achita_original_finalize = fused_post_expert_norm_weighted_sum

def fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=None, eps=1e-08, weight_first=False, slot_expert_ids=None):
 if y.shape[0] <= 16 and y.shape[2] == 4096 and (y.shape[1] == 8) and weight_first and (norm_weight is not None) and (slot_expert_ids is not None):
  result = torch.empty((y.shape[0], y.shape[2]), device=y.device, dtype=y.dtype)
  _achita_moe_small_finalize[y.shape[0],](y, topk_weights, norm_weight, slot_expert_ids, result, y.shape[2], y.shape[1], eps, 8, num_warps=8)
  return result
 return _achita_original_finalize(y, topk_weights, norm_weight=norm_weight, eps=eps, weight_first=weight_first, slot_expert_ids=slot_expert_ids)

def _achita_dense_project(x, layer, state, rows=None):
 w = state['dense'][layer]
 n = w.shape[0] if rows is None else rows
 data = {'x': x, 'config': {'m': x.shape[0], 'n': n, 'k': x.shape[1], 'layer': layer}}
 output = _achita_dense_kernel(data, state['achita_dense'])
 return _gemm(x, w[:n]) if output is None else output
_achita_layer_prepare = prepare

def prepare(weights):
 state = _achita_layer_prepare(weights)
 state['achita_dense'] = _achita_dense_prepare({key: weights[key] for key in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')})
 return state
import math
import triton
import triton.language as tl

@triton.jit
def _achita_qkv_preprocess(QKV, POS, LOC, KC, VC, KS, VS, QOUT, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, STRIDE: tl.constexpr, LOG_THETA: tl.constexpr, ROPE: tl.constexpr):
 token = tl.program_id(0)
 head = tl.program_id(1)
 dim = tl.arange(0, D)
 values = tl.load(QKV + token * STRIDE + head * D + dim).to(tl.float32)
 inv = tl.rsqrt(tl.sum(values * values, 0) / D + 1e-05)
 values = values * inv
 if ROPE:
  other = tl.load(QKV + token * STRIDE + head * D + (dim ^ 1)).to(tl.float32) * inv
  position = tl.load(POS + token).to(tl.float32)
  angle = position * tl.exp(-2.0 * (dim // 2) / D * LOG_THETA)
  values = values * tl.cos(angle) + tl.where(dim % 2 == 0, -other, other) * tl.sin(angle)
 values = values.to(tl.bfloat16).to(tl.float32)
 if head < HQ:
  tl.store(QOUT + (token * HQ + head) * D + dim, values)
 else:
  kv_head = head - HQ
  slot = tl.load(LOC + token)
  cache_offset = (slot * HKV + kv_head) * D + dim
  tl.store(KC + cache_offset, values / tl.load(KS))
  value = tl.load(QKV + token * STRIDE + (HQ + HKV + kv_head) * D + dim).to(tl.float32)
  tl.store(VC + cache_offset, value / tl.load(VS))

def _achita_decoder_attention(data, qkv):
 cfg = data['config']
 (batch, length, tokens) = (cfg['batch'], cfg['q_len'], cfg['tokens'])
 (hq, hkv, d) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
 q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
 _achita_qkv_preprocess[tokens, hq + hkv](qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], q, hq, hkv, d, qkv.stride(0), math.log(cfg['rope_theta']) if cfg['rope_theta'] else 0.0, bool(cfg['rope_theta']), num_warps=4)
 descale = {}
 if cfg['kv_scale'] != 1.0:
  descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv), v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
 out = flash_attn_with_kvcache(q=q, k_cache=data['k_cache'], v_cache=data['v_cache'], page_table=data['page_table'], cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=length, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'], causal=True, window_size=(cfg['window'] - 1, 0) if cfg['window'] else (-1, -1), scheduler_metadata=data['scheduler_metadata'], **descale)
 return out[0] if isinstance(out, tuple) else out
_achita_before_native_prepare = prepare

def prepare(weights):
 state = _achita_before_native_prepare(weights)
 expert_weights = {k: weights[k] for k in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
 expert_weights['post_norm_w'] = weights['post_expert_norm']
 state['native_moe'] = _native_moe_prepare(expert_weights)
 return state

def _experts(x, topk_weights, topk_ids, state):
 return _native_moe_kernel_fn({'hidden_states': x, 'topk_ids': topk_ids, 'topk_weights': topk_weights, 'config': {'local_start': 0, 'local_count': 32}}, state['native_moe'])
_fpr_CUDA = '#include <type_traits>\n\n#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n\n__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"\n               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));\n}\n\n__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {\n  // c ? b : a, bitwise\n  uint32_t d;\n  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));\n  return d;\n}\n\n// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw\n// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).\n// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).\n__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {\n  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;\n  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;\n  r[0] = prmt(lw0, ha, 0x5140);\n  r[1] = prmt(lw0, ha, 0x7362);\n  r[2] = prmt(lw1, hb, 0x5140);\n  r[3] = prmt(lw1, hb, 0x7362);\n}\n\ntemplate <typename T> __device__ __forceinline__ T cvt_out(float v);\ntemplate <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }\ntemplate <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }\n\n// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.\n// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low\n//   bytes of the lane\'s 32 weights of 4 k-steps, part 2 = their 32 codes.\n// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).\n// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).\n// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).\n// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).\n\n// ------------------------------------------------------------------ mbarrier / bulk copy helpers\n__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {\n  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");\n}\n__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {\n  while (!mbar_test(bar, parity)) {}\n}\n// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling\n__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {\n  uint32_t ok = 0;\n  do {\n    asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");\n  } while (!ok);\n}\n__device__ __forceinline__ void mbar_arrive(uint32_t bar) {\n  asm volatile("{\\n .reg .b64 st;\\n mbarrier.arrive.shared.b64 st, [%0];\\n}\\n" :: "r"(bar) : "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {\n  asm volatile("{\\n .reg .b64 st;\\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\\n}\\n" :: "r"(bar), "r"(bytes) : "memory");\n}\n// L2 policies (v86y = v86r + bf16_proj v131\'s hints): weight streams are read once per call -> evict_first; x tiles\n// are re-read by every CTA -> evict_last\n__device__ __forceinline__ uint64_t l2_pol_ef() {\n  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;\n}\n__device__ __forceinline__ uint64_t l2_pol_el() {\n  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;\n}\n__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");\n}\n// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)\n__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {\n  uint32_t ok = 0;\n  do {\n    asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\\n"\n                 " selp.u32 %0, 1, 0, p;\\n}\\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");\n  } while (!ok);\n}\n\n\n// cross-CTA spins are bounded (POD.md Oct 26 17:50) in the measured-free form (attention v298, POD.md 18:31): count\n// polls, no timer read per poll, and past 2^24 polls call an out-of-line trap (keeps the hot path\'s codegen unchanged)\n__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }\n__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {\n  for (unsigned it = 0;; ++it) {\n    if (*((volatile const int*)p) - target >= 0) return;\n    if (ns > 0) __nanosleep(ns);\n    if (it >= (1u << 24)) okc_spin_trap();\n  }\n}\n\n\n#include <cudaTypedefs.h>\n#include <cstring>\n__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"\n               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {\n  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());\n}\n__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {\n  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());\n}\n__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {\n  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"\n               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");\n}\nstatic PFN_cuTensorMapEncodeTiled_v12000 get_encode() {\n  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;\n  if (!fn) {\n    cudaDriverEntryPointQueryResult q;\n    void* p = nullptr;\n    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);\n    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");\n    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);\n  }\n  return fn;\n}\n\n// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]\nstatic CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {\n  CUtensorMap mp;\n  cuuint64_t dims[3] = {64, m, k / 64};\n  cuuint64_t strides[2] = {k * 2, 128};\n  cuuint32_t box[3] = {64, box_rows, units};\n  cuuint32_t estr[3] = {1, 1, 1};\n  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,\n                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,\n                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);\n  return mp;\n}\n\nstatic CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {\n  CUtensorMap m;\n  cuuint64_t dims[2] = {cols, rows};\n  cuuint64_t strides[1] = {cols * 2};\n  cuuint32_t box[2] = {64, box_rows};\n  cuuint32_t estr[2] = {1, 1};\n  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,\n                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,\n                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);\n  return m;\n}\n\n\n\n#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))\n// PDL (programmatic dependent launch) for the sk2 launches: 0 = off (plain launches); N > 0 = launch with programmatic\n// stream serialization and issue the first N weight stages before griddepcontrol.wait (set by sk2_set_pdl)\nstatic int g_sk2_pdl = 0;\nvoid sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n#define SK2_DIAG 0\n// ============================================================================ sk2 (m <= 32): swapped operands\n// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),\n// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole\n// K: no cross-CTA reduction.  Warps split the CTA\'s rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).\n// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA\'s weights.\ntemplate <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,\n          bool PK, bool FUSE>\n__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)\nsk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,\n           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,\n           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,\n           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,\n           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {\n  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds\n  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };\n  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;\n  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);\n  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;\n  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;\n  if (trace && threadIdx.x == 0) tr[0] = t_start;\n  constexpr int NW = NWR * WSK;\n  constexpr int NC = NW * 32;\n  constexpr int R = NWR * NT * 8;\n  constexpr int UPS = WSK * UPW;\n  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);\n  constexpr int WBY = NW * UPW * NT * TU;\n  constexpr int XT = TOKR * 128;\n  constexpr int XBY = UPS * XT;\n  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;\n  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out\n  constexpr int TOKS = 16 * MT;\n  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores\n  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates\n  extern __shared__ __align__(1024) unsigned char smem_raw[];\n  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);\n  const uint32_t sbase = (sraw + 1023u) & ~1023u;\n  unsigned char* gbase = smem_raw + (sbase - sraw);\n  const uint32_t bars = sbase + NSTAGE * SST;\n  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;\n  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages\n  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;\n  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");\n  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);\n  constexpr int CSZ = (TOKS * CP + 3) & ~3;\n  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums\n  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes\n  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)\n  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)\n  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M\n  // read zrow); else the full TOKR-row tile with compile-time addressing\n  const int Mb = PK ? M : TOKR;\n  const uint32_t XU = (uint32_t)Mb * 128u;\n  const uint32_t XBYr = UPS * XU;\n  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed\n\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0\n  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;\n  const int tst = K / (UPS * 64);                      // stages over all of K\n  const int gs0 = (crank * tst) / CL;                  // this CTA\'s first stage (uneven split for CL = 3)\n  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count\n  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)\n  const uint8_t* wsrc = Ws + (size_t)cta * WBY;\n  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;\n  // FUSE (pooler + router in one launch): after its pooler stages every CTA streams one router piece: 8 router rows\n  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router\'s own stage-major hi|lo stream\n  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring\n  constexpr int NSTR = FUSE ? 4 : 0;\n  constexpr int WBYR = NW * 2048;\n  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");\n  const int rt = cta >> 2, rq = cta & 3;\n  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;\n  const int nstt = nst + NSTR;\n  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA\'s own stages)\n  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;\n\n  if (tid == 0) {\n    // stage 0 first of all (the first bytes\' latency is the ramp of the whole kernel): the weight copy needs only\n    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster\n    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier\n    // is initialized and fenced before the first copy.\n    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");\n    mbar_init(full0, 1);\n    if constexpr (CL == 1)   // PDL: the first P stages\' weights go out before the wait; their barriers first\n      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);\n    if constexpr (CL > 1) {\n      mbar_init(empty0, NW + (COMP ? NESC : 0));\n      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }\n      mbar_init(pbar, 1);\n      mbar_init(cbar, 32 * NESC);\n      mbar_init(qbar, 1);\n      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    // weights never depend on the previous kernel: under PDL (pdl > 0) they stream while it finishes; x (its output)\n    // only after griddepcontrol.wait (a no-op when the launch has no programmatic dependency)\n    for (int st = 0; st < P; ++st) {\n      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);\n      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);\n    }\n    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (!(dgv & 4))\n      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n    if (COMP) {   // this CTA\'s escape table\n      if constexpr (CL == 1) {\n        mbar_init(pbar, 1);\n        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n      }\n      mbar_expect_tx(pbar, pmax * 8);\n      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);\n    }\n    if constexpr (CL == 1) {\n      mbar_init(empty0, NW + (COMP ? NESC : 0));\n      for (int i = 1; i < NSTAGE; ++i) {\n        if (i >= P) mbar_init(full0 + 8 * i, 1);\n        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));\n      }\n      if (!COMP) mbar_init(pbar, 1);\n      mbar_init(cbar, 32 * NESC);\n      mbar_init(qbar, 1);\n      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    }\n  }\n  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;\n  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0\'s qbar initialized before\n  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores\n  __syncthreads();\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here\n  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above\n\n  if (warp == NW) {\n    // producer warp: lane 0 refills the ring\n    if (lane == 0) {\n      for (int st = P; st < nstt; ++st) {\n        const int slot = st % NSTAGE;\n        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);\n        if (trace) tr[16 + 3 * st] = gtime();\n        const uint32_t s0 = sbase + slot * SST;\n        if (!FUSE || st < nst) {\n          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n        } else {   // router piece: global router stage rq * 4 + j of tile rt\n          const int gsr = rq * NSTR + (st - nst);\n          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);\n          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);\n          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);\n        }\n      }\n    }\n    __syncwarp();\n    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\n    return;\n  }\n  if (warp >= NW + 1) {\n    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the\n    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,\n    // so corr[t][row] accumulates with no atomics; each warp has its own copy\n    const int ew = warp - (NW + 1);\n    if (COMP) {\n      float* cr = corr + ew * CSZ;\n      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;\n      mbar_wait_sleep(pbar, 0);\n      const int cnt = ptab_s[0].x;\n      constexpr int KST = UPS * 64;\n      int pe = 1;\n      for (int st = 0; st < nstt; ++st) {\n        const int slot = st % NSTAGE;\n        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)\n          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // every escape warp hands every stage back\n          continue;\n        }\n        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\n        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY\n        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot\n        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a\n        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).\n        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if ((st % NESC) != ew) {   // another warp\'s stage: skip its escapes, hand the slot back\n          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;\n          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n          continue;\n        }\n        const unsigned char* xs = gbase + slot * SST;\n        while (pe <= cnt) {\n          const int2 e = ptab_s[pe];\n          const int kk = e.x & 0xFFFF;\n          if (kk >= kend) break;\n          if (lane < M && kk >= kbeg) {\n            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;\n            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(\n                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);\n            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);\n          }\n          ++pe;\n        }\n        __syncwarp();\n        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n      }\n    }\n    mbar_arrive(cbar);\n    __syncwarp();\n    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\n    return;\n  }\n\n  // ---------------- consumers\n  const int wr = warp / WSK, wk = warp % WSK;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int q = lane >> 3;\n  const int chi = q >> 1;\n  int trow[MT];\n#pragma unroll\n  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);\n  float acc[NACC][MT][NT][4];\n#pragma unroll\n  for (int c = 0; c < NACC; ++c)\n#pragma unroll\n    for (int mt = 0; mt < MT; ++mt)\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;\n\n  for (int st = 0; st < nst; ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n    const uint32_t s0 = sbase + slot * SST;\n    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }\n#pragma unroll\n    for (int uu = 0; uu < UPW; ++uu) {\n      const int ui = wk * UPW + uu;\n      const uint32_t xt = s0 + ui * XU;\n      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;\n      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)\n      uint4 lw[NT];\n      uint2 cw[NT];\n      uint32_t b[NT][8];\n      uint32_t bl[HL ? NT : 1][8];\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt) {\n        if constexpr (COMP) {\n          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));\n          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));\n        } else {\n          const uint32_t a0 = wb + nt * TU + lane * 32;\n          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));\n          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));\n          if constexpr (HL) {\n            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));\n            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));\n          }\n        }\n      }\n      uint32_t a[4][MT][4];\n      if (dgv & 16) {\n#pragma unroll\n        for (int j = 0; j < 4; ++j)\n#pragma unroll\n          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;\n      } else {\n#pragma unroll\n        for (int j = 0; j < 4; ++j)\n#pragma unroll\n          for (int mt = 0; mt < MT; ++mt)\n            ldsm_x4((PK && trow[mt] >= M) ? zrow\n                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),\n                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);\n      }\n      // the stage\'s shared memory is in registers now: hand the slot back before decoding and multiplying\n      if (UPW == 1 && uu == 0) {\n        __syncwarp();\n        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n      }\n      // 2) decode\n      if constexpr (COMP) {\n#pragma unroll\n        for (int nt = 0; nt < NT; ++nt) {\n          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);\n          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);\n        }\n      }\n      // 3) tensor-core work: independent accumulators per k-step\n      if (dgv & 8) {   // diagnostic: replace the tensor-core instructions by ALU work\n        const int nh = 0;\n#pragma unroll\n        for (int j = 0; j < 4; ++j)\n#pragma unroll\n          for (int nt = 0; nt < NT; ++nt) {\n            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);\n            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);\n          }\n        continue;\n      }\n#pragma unroll\n      for (int j = 0; j < 4; ++j)\n#pragma unroll\n        for (int mt = 0; mt < MT; ++mt)\n#pragma unroll\n          for (int nt = 0; nt < NT; ++nt) {\n            constexpr int HM = HL ? 2 : 1;\n            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);\n            if constexpr (HL)\n              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);\n          }\n    }\n    if (UPW > 1) {\n      __syncwarp();\n      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n    }\n    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n  }\n  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue\n  if constexpr (FUSE) {\n    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators\n    float accr[2][MT][4];\n#pragma unroll\n    for (int h = 0; h < 2; ++h)\n#pragma unroll\n      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;\n    for (int st = nst; st < nstt; ++st) {\n      const int slot = st % NSTAGE;\n      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n      const uint32_t s0 = sbase + slot * SST;\n      const int ui = wk;\n      const uint32_t xt = s0 + ui * XU;\n      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;\n      uint32_t bh[8], bo[8];\n      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));\n      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));\n      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));\n      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));\n      uint32_t a[4][MT][4];\n#pragma unroll\n      for (int j = 0; j < 4; ++j)\n#pragma unroll\n        for (int mt = 0; mt < MT; ++mt)\n          ldsm_x4((PK && trow[mt] >= M) ? zrow\n                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),\n                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);\n      __syncwarp();\n      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n#pragma unroll\n      for (int j = 0; j < 4; ++j)\n#pragma unroll\n        for (int mt = 0; mt < MT; ++mt) {\n          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);\n          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);\n        }\n    }\n    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8]\n    if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warps are done with the ring too\n    named_bar(1, NC);\n    float* redr = reinterpret_cast<float*>(gbase);\n#pragma unroll\n    for (int mt = 0; mt < MT; ++mt) {\n      const int row = 2 * t4, tok = mt * 16 + g;\n      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];\n      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];\n      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];\n      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];\n    }\n    named_bar(1, NC);\n    // this CTA\'s quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)\n    // adds them in quarter order (deterministic) and writes the router output\n    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;\n    for (int i = tid; i < TOKS * 8; i += NC) {\n      float v = 0.f;\n#pragma unroll\n      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];\n      __stcg(pp + i, v);\n    }\n    named_bar(1, NC);\n    __shared__ int s_rlast;\n    if (tid == 0) {\n      int old;\n      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");\n      s_rlast = (old & 3) == 3;\n      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");\n    }\n    named_bar(1, NC);\n    if (s_rlast) {\n      const float* pr = part + (size_t)rt * 4 * 32 * 8;\n      for (int i = tid; i < M * 8; i += NC) {\n        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);\n        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;\n      }\n    }\n    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below\n  }\n  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done\n\n  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)\n  if (dgv & 1) {\n    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);\n    return;\n  }\n  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring\'s x tiles until it is done\n  if (CL > 1 && crank > 0) {\n    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0\'s qbar is initialized\n    // ranks > 0: every accumulator goes straight into rank 0\'s peer buffer (slot crank - 1) by st.async\n    // (completes rank 0\'s qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0\'s\n    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);\n    uint32_t peer_r, qbar_r;\n    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));\n    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));\n#pragma unroll\n    for (int mt = 0; mt < MT; ++mt)\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt) {\n        const int row = (wr * NT + nt) * 8 + 2 * t4;\n        const int tok = mt * 16 + g;\n        float v[4];\n#pragma unroll\n        for (int e = 0; e < 4; ++e) {\n          v[e] = acc[0][mt][nt][e];\n#pragma unroll\n          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];\n        }\n        if (COMP && wk == 0) {\n          v[0] += corr[tok * CP + row];\n          v[1] += corr[tok * CP + row + 1];\n          if (tok + 8 < 32) {\n            v[2] += corr[(tok + 8) * CP + row];\n            v[3] += corr[(tok + 8) * CP + row + 1];\n          }\n        }\n        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);\n        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"\n                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");\n        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"\n                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");\n      }\n    if (trace && tid == 0) tr[15] = gtime();\n    return;\n  }\n  named_bar(1, NC);\n  if (trace && tid == 0) tr[9] = gtime();\n#pragma unroll\n  for (int mt = 0; mt < MT; ++mt)\n#pragma unroll\n    for (int nt = 0; nt < NT; ++nt) {\n      const int row = (wr * NT + nt) * 8 + 2 * t4;\n      const int tok = mt * 16 + g;\n      float* r0 = red + (wk * TOKS + tok) * RP + row;\n      float v[4];\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        v[e] = acc[0][mt][nt][e];\n#pragma unroll\n        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];\n      }\n      if (trace && mt == 0 && nt == 0) {\n        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));\n        if (tid == 0) tr[12] = gtime();\n      }\n      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);\n      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);\n    }\n  if (trace && tid == 0) tr[11] = gtime();\n  named_bar(1, NC);\n  if (trace && tid == 0) tr[10] = gtime();\n  if (trace && tid == 0) tr[13] = gtime();\n  if constexpr (CL > 1) {\n    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\n    mbar_wait_cluster(qbar, 0);\n  }\n  if (trace && tid == 0) tr[14] = gtime();\n  constexpr int R4 = R / 4;\n  for (int idx = tid; idx < R4 * M; idx += NC) {\n    const int tt = idx / R4, row = (idx - tt * R4) * 4;\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    if constexpr (COMP) {\n#pragma unroll\n      for (int c = 0; c < NESC; ++c) {\n        v.x += corr[c * CSZ + tt * CP + row];\n        v.y += corr[c * CSZ + tt * CP + row + 1];\n        v.z += corr[c * CSZ + tt * CP + row + 2];\n        v.w += corr[c * CSZ + tt * CP + row + 3];\n      }\n    }\n#pragma unroll\n    for (int w = 0; w < WSK; ++w) {\n      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);\n      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;\n#pragma unroll\n      for (int c = 0; c < CL - 1; ++c) {\n        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);\n        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;\n      }\n    }\n    OutT* o = out + (size_t)tt * N + cta * R + row;\n    if constexpr (sizeof(OutT) == 4) {\n      *reinterpret_cast<float4*>(o) = v;\n    } else {\n      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);\n      uint2 pk;\n      pk.x = *reinterpret_cast<uint32_t*>(&lo);\n      pk.y = *reinterpret_cast<uint32_t*>(&hi);\n      *reinterpret_cast<uint2*>(o) = pk;\n    }\n  }\n  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {   // every CTA: SM id, start and end time\n    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));\n    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();\n  }\n}\n\ntemplate <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,\n          bool FUSE = false>\nstatic void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,\n                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,\n                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {\n  constexpr int NW = NWR * WSK;\n  constexpr int R = NWR * NT * 8;\n  constexpr int UPS = WSK * UPW;\n  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);\n  constexpr int WBY = NW * UPW * NT * TU;\n  constexpr int XBY = UPS * TOKR * 128;\n  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;\n  constexpr int TOKS = 16 * MT;\n  constexpr int RP = R + ((40 - (R % 32)) % 32);\n  constexpr int CP = R + 1;\n  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;\n  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +\n                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;\n  auto go = [&](auto pkc) {\n    constexpr bool PK = decltype(pkc)::value;\n    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;\n    static bool attr_done = false;\n    if (!attr_done) {\n      int dev = 0, optin = 0;\n      cudaGetDevice(&dev);\n      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n      cudaFuncAttributes fa;\n      cudaFuncGetAttributes(&fa, kern);\n      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);\n      attr_done = true;\n    }\n    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);\n    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());\n    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());\n    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());\n    const int pdl = g_sk2_pdl;\n    if (CL > 1 || pdl > 0) {\n      cudaLaunchConfig_t lc = {};\n      lc.gridDim = dim3(N / R * CL);\n      lc.blockDim = dim3((NW + 1 + NESC) * 32);\n      lc.dynamicSmemBytes = smem;\n      lc.stream = at::cuda::getCurrentCUDAStream();\n      cudaLaunchAttribute at_[2];\n      int na = 0;\n      if (CL > 1) {\n        at_[na].id = cudaLaunchAttributeClusterDimension;\n        at_[na].val.clusterDim.x = CL;\n        at_[na].val.clusterDim.y = 1;\n        at_[na].val.clusterDim.z = 1;\n        ++na;\n      }\n      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes\n        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n        at_[na].val.programmaticStreamSerializationAllowed = 1;\n        ++na;\n      }\n      lc.attrs = at_;\n      lc.numAttrs = na;\n      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,\n                                        wsr, outr, part, rcnt, pdl));\n    } else {\n      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,\n                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,\n                                                                               wsr, outr, part, rcnt, 0);\n      C10_CUDA_KERNEL_LAUNCH_CHECK();\n    }\n  };\n  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});\n}\n\nvoid sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,\n          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {\n  if (tokr == 16) { launch_sk2<1, 16, 4, 1, 8, 1, 5, true, false,\n        2, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr,\n        reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(),\n        rcnt.data_ptr<int>()); return; }\nif (tokr == 32) { launch_sk2<2, 32, 4, 1, 8, 1, 3, true, false,\n        2, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr,\n        reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(),\n        rcnt.data_ptr<int>()); return; }\n  TORCH_CHECK(false, "sk2f: no config");\n}\n\nvoid sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\n         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {\n  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n  \n  TORCH_CHECK(false, "sk2: no config");\n}\n'
_fpr_CPP = '#include <torch/extension.h>\n\nvoid skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,\n                torch::Tensor ptab, int64_t pmax, int64_t base4,\n                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,\n                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);\nvoid gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,\n            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);\nvoid wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,\n          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);\nvoid gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,\n        int64_t cl, int64_t nstage);\nvoid sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,\n         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);\nvoid sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,\n          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);\nvoid sk2_set_pdl(int64_t n);\n'
_fpr_GEOMETRIES = {16: 5, 32: 3}

def _fpr_bfp_sk2_tileunits(w):
 (n, k) = w.shape
 t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
 return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)

def _fpr_bfp_sk2_compress(w):
 (n, k) = w.shape
 tu = _fpr_bfp_sk2_tileunits(w)
 bits = tu.contiguous().view(torch.int16).to(torch.int32) & 65535
 hi = bits >> 8
 low = bits & 255
 e71 = hi & 127
 sign = hi >> 7
 hist = torch.bincount(e71.flatten(), minlength=128)
 win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
 base = int(win.argmax())
 o = (e71 - base).clamp(0, 7)
 code = sign << 3 | o
 esc = (e71 < base) | (e71 > base + 7)
 if int(esc.sum()) > n * k // 200:
  return None
 (T, U) = (n // 8, k // 64)
 c8 = code.to(torch.uint8)
 w0 = c8[..., 0:4] | c8[..., 4:8] << 4
 w1 = c8[..., 8:12] | c8[..., 12:16] << 4
 codes = torch.cat([w0, w1], dim=-1)
 packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
 idx = esc.nonzero()
 dec_bits = ((sign << 7 | base + o) << 8 | low)[esc]
 wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
 wtrue = tu[esc].float()
 delta = wtrue - wdec
 if not bool((wdec + delta == wtrue).all()):
  return None
 (lane, v) = (idx[:, 2], idx[:, 3])
 rows = idx[:, 0] * 8 + (lane >> 2)
 cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * (v >> 1 & 1) + 2 * (lane & 3) + (v & 1)
 return {'packed': packed, 'base4': base * 16843009, 'rows': rows, 'cols': cols, 'delta': delta.contiguous()}

def _fpr_bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
 dev = tub.device
 U = tub.shape[1]
 ups = WSK * UPW
 ar = lambda v: torch.arange(v, device=dev)
 s = ar(U // ups).view(-1, 1, 1, 1, 1)
 c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
 w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
 uu = ar(UPW).view(1, 1, 1, -1, 1)
 nt = ar(NT).view(1, 1, 1, 1, -1)
 tile = c * (R // 8) + w // WSK * NT + nt
 unit = s * ups + w % WSK * UPW + uu
 return tub[tile, unit].contiguous()

def _fpr_bfp_sk2_table(comp, n_cfg, R):
 dev = comp['rows'].device
 (rows, cols, delta) = (comp['rows'], comp['cols'], comp['delta'])
 keep = rows < n_cfg
 (rows, cols, delta) = (rows[keep], cols[keep], delta[keep])
 ncta = n_cfg // R
 cta = rows // R
 counts = torch.bincount(cta, minlength=ncta)
 pmax = int(counts.max().item()) + 1 if rows.numel() else 1
 pmax += pmax % 2
 order = torch.argsort(cta.long() * 65536 + cols.long())
 (cta, rows, cols, delta) = (cta[order], rows[order], cols[order], delta[order])
 start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
 start[1:] = torch.cumsum(counts, 0)
 slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
 tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
 tab[:, 0, 0] = counts.to(torch.int32)
 tab[cta, slot, 0] = (rows % R << 16 | cols).to(torch.int32)
 tab[cta, slot, 1] = delta.view(torch.int32)
 return (tab.contiguous(), pmax)

def _fpr_bfp__sk2_tokr(m):
 return 8 if m <= 8 else 16 if m <= 16 else 32

def _fpr_bfp_fused_pooler_router(x, state):
 m = x.shape[0]
 assert m <= 32 and state.get('sm90') and ('pooler' in state['sk2']) and ('router' in state['sk2'])
 (ci, stream, tab, pmax, b4, n_rows) = state['sk2']['pooler']
 rstream = state['sk2']['router'][1]
 if 'fused_part' not in state:
  state['fused_part'] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
  state['fused_cnt'] = torch.zeros(32, dtype=torch.int32, device=x.device)
 out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
 outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
 ext = _fpr_get_ext()
 ext.sk2_set_pdl(_fpr_PDL)
 ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state['fused_part'], state['fused_cnt'], m, _fpr_bfp__sk2_tokr(m))
 return (out, outr)
from torch.utils.cpp_extension import load_inline as _fpr_load_inline
_fpr_ext = None
_fpr_PDL = 0

def _fpr_get_ext():
 global _fpr_ext
 if _fpr_ext is None:
  (major, minor) = torch.cuda.get_device_capability()
  arch = f'{major}{minor}a'
  _fpr_ext = _fpr_load_inline(name='okc_fpr_f68f77c0d29cc56b_' + arch, cpp_sources=_fpr_CPP, cuda_sources=_fpr_CUDA, functions=['sk2f', 'sk2_set_pdl'], extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}'], no_implicit_headers=True, verbose=False)
  _fpr_ext.sk2_set_pdl(0)
 return _fpr_ext

def _fpr_prepare(weights):
 if torch.cuda.get_device_capability()[0] != 9:
  return None
 w = weights['pooler'].contiguous()
 bits = w.view(torch.int16)
 if bool((bits == -32768).any()) or not bool(torch.isfinite(w).all()):
  return None
 comp = _fpr_bfp_sk2_compress(w)
 if comp is None:
  return None
 (tab, pmax) = _fpr_bfp_sk2_table(comp, 4096, 32)
 for (tokr, nstage) in _fpr_GEOMETRIES.items():
  toks = 32 if tokr == 32 else 16
  sst = (8 * tokr * 128 + 8 * 4 * 768 + 1023) // 1024 * 1024
  nesc = min(nstage, 4)
  smem = 1024 + nstage * sst + 16 * nstage + 32 + nesc * ((toks * 33 + 3) // 4 * 4) * 4 + pmax * 8 + 128
  if smem > 232448:
   return None
 ps = _fpr_bfp_sk2_stream(comp['packed'], 4096, 32, 4, 1, 8, 1)
 router = weights['router']
 hi = router.to(torch.bfloat16)
 lo = (router - hi.float()).to(torch.bfloat16)
 (T, U) = (32, 128)
 tub = torch.cat([_fpr_bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024), _fpr_bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
 rs = _fpr_bfp_sk2_stream(tub, 256, 8, 1, 1, 8, 1)
 state = {'sm90': True, 'sk2': {'pooler': (0, ps, tab, pmax, comp['base4'], 4096), 'router': (0, rs, None, 0, 0, 256)}, 'fused_part': torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=w.device), 'fused_cnt': torch.zeros(32, dtype=torch.int32, device=w.device)}
 _fpr_get_ext()
 return state
_fpr_original_prepare = prepare

def prepare(weights):
 state = _fpr_original_prepare(weights)
 state['fpr'] = _fpr_prepare(weights)
 return state
from torch.utils.cpp_extension import load_inline as _rf_load_inline
_RF_CPP = '\n#include <torch/extension.h>\nvoid run(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor parts,\n         torch::Tensor counter, torch::Tensor outw, torch::Tensor outi);\n'
_RF_CUDA = "\n#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_runtime.h>\n#include <math.h>\n#include <limits.h>\n\n// H200: 16 row tiles * 8 K splits = 128 CTAs, approximately one SM wave.\n// Each warp handles two router rows, sharing their weights across all tokens.\ntemplate<int TM>\n__global__ void route_fp32(const float* __restrict__ x, const float* __restrict__ w,\n                          const float* __restrict__ bias, volatile float* parts,\n                          unsigned int* counter, float* outw, int* outi, int M) {\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int nb = blockIdx.x & 15, ks = blockIdx.x >> 4;\n  const int row0 = nb * 16 + warp * 2, row1 = row0 + 1;\n  float a[TM], b[TM];\n#pragma unroll\n  for (int m = 0; m < TM; ++m) a[m] = b[m] = 0.f;\n#pragma unroll\n  for (int j = 0; j < 32; ++j) {\n    const int k = ks * 1024 + j * 32 + lane;\n    const float w0 = w[(size_t)row0 * 8192 + k], w1 = w[(size_t)row1 * 8192 + k];\n#pragma unroll\n    for (int m = 0; m < TM; ++m) {\n      if (m < M) {\n        const float v = x[(size_t)m * 8192 + k];\n        a[m] = __fmaf_rn(v, w0, a[m]);\n        b[m] = __fmaf_rn(v, w1, b[m]);\n      }\n    }\n  }\n#pragma unroll\n  for (int m = 0; m < TM; ++m) {\n    if (m < M) {\n#pragma unroll\n      for (int d = 16; d > 0; d >>= 1) {\n        a[m] += __shfl_down_sync(0xffffffffu, a[m], d);\n        b[m] += __shfl_down_sync(0xffffffffu, b[m], d);\n      }\n      if (lane == 0) {\n        parts[((size_t)ks * 8 + m) * 256 + row0] = a[m];\n        parts[((size_t)ks * 8 + m) * 256 + row1] = b[m];\n      }\n    }\n  }\n  // Publish all eight producer warps' partials before this CTA's completion.\n  // Fenced stores + volatile partials follow CUDA's last-block reduction pattern:\n  // docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html\n  if (lane == 0) __threadfence();\n  __syncthreads();\n  __shared__ int last;\n  if (tid == 0) {\n    unsigned int old = atomicAdd(counter, 1u);\n    last = ((old & 127u) == 127u);\n  }\n  __syncthreads();\n  if (!last || warp >= M) return;\n  __threadfence();\n  // The last CTA reduces every split, then routes one token per warp. The\n  // counter advances 128 per invocation; its low seven bits safely cycle even\n  // across uint32 wrap. All consumed partials are overwritten on every call.\n  float score[8], sel[8];\n#pragma unroll\n  for (int j = 0; j < 8; ++j) {\n    const int e = lane + 32 * j;\n    float v = 0.f;\n#pragma unroll\n    for (int s = 0; s < 8; ++s) v += parts[((size_t)s * 8 + warp) * 256 + e];\n    const float z = __fmul_rn(v, 0.1767766952966369f);\n    score[j] = 1.f / (1.f + expf(-z));\n    sel[j] = score[j] + bias[e];\n  }\n  float chosen_score[8];\n  int chosen_id[8];\n  float sum = 0.f;\n#pragma unroll\n  for (int r = 0; r < 8; ++r) {\n    float best = -INFINITY, raw = 0.f;\n    int id = INT_MAX;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int e = lane + j * 32;\n      if (sel[j] > best || (sel[j] == best && e < id)) {\n        best = sel[j]; raw = score[j]; id = e;\n      }\n    }\n#pragma unroll\n    for (int d = 16; d > 0; d >>= 1) {\n      const float ob = __shfl_xor_sync(0xffffffffu, best, d);\n      const float os = __shfl_xor_sync(0xffffffffu, raw, d);\n      const int oi = __shfl_xor_sync(0xffffffffu, id, d);\n      if (ob > best || (ob == best && oi < id)) { best = ob; raw = os; id = oi; }\n    }\n    chosen_score[r] = raw; chosen_id[r] = id; sum += raw;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) if (lane + j * 32 == id) sel[j] = -INFINITY;\n  }\n  const float inv = 1.f / (sum + 1e-15f);\n#pragma unroll\n  for (int r = 0; r < 8; ++r) {\n    if (lane == r) {\n      outw[warp * 8 + r] = chosen_score[r] * inv;\n      outi[warp * 8 + r] = chosen_id[r];\n    }\n  }\n}\n\nvoid run(at::Tensor x, at::Tensor w, at::Tensor bias, at::Tensor parts,\n         at::Tensor counter, at::Tensor outw, at::Tensor outi) {\n  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat);\n  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kFloat);\n  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8 && x.size(1) == 8192);\n  TORCH_CHECK(w.dim() == 2 && w.size(0) == 256 && w.size(1) == 8192);\n  const int M = (int)x.size(0);\n  auto stream = at::cuda::getCurrentCUDAStream();\n  auto* p = (volatile float*)parts.data_ptr<float>();\n  auto* c = reinterpret_cast<unsigned int*>(counter.data_ptr<int>());\n  if (M == 1) route_fp32<1><<<128, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),\n      p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M);\n  else route_fp32<8><<<128, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),\n      p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M);\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n"
_rf_ext = None

def _rf_get_ext():
 global _rf_ext
 if _rf_ext is None:
  _rf_ext = _rf_load_inline(name='okc_router_fp32_d5a88ed70badd431_90a', cpp_sources=_RF_CPP, cuda_sources=_RF_CUDA, functions=['run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, verbose=False)
 return _rf_ext

def _rf_route(x, state):
 x = x.contiguous()
 outw = torch.empty((x.shape[0], 8), dtype=torch.float32, device=x.device)
 outi = torch.empty((x.shape[0], 8), dtype=torch.int32, device=x.device)
 state['ext'].run(x, state['weight'], state['bias'], state['parts'], state['counter'], outw, outi)
 return (outw, outi)
_rf_old_prepare = prepare

def prepare(weights):
 state = _rf_old_prepare(weights)
 dev = weights['router'].device
 state['rf'] = None
 if torch.cuda.get_device_capability(dev)[0] == 9:
  state['rf'] = {'ext': _rf_get_ext(), 'weight': state['router'], 'bias': weights['router_bias'].float().contiguous(), 'parts': torch.empty(8 * 8 * 256, dtype=torch.float32, device=dev), 'counter': torch.zeros(1, dtype=torch.int32, device=dev)}
 return state

"""Fuse pre-expert RMSNorm and global-to-local route rotation."""
import triton as _pn_triton
import triton.language as _pn_tl


@_pn_triton.jit
def _pn_inputs_kernel(X, Gain, Ids, Start, Out, LocalIds,
                      XStride: _pn_tl.constexpr, IdStride: _pn_tl.constexpr,
                      H: _pn_tl.constexpr, Eps: _pn_tl.constexpr,
                      Block: _pn_tl.constexpr):
    row = _pn_tl.program_id(0)
    col = _pn_tl.arange(0, Block)
    x = _pn_tl.load(X + row * XStride + col, col < H, 0).to(_pn_tl.float32)
    gain = _pn_tl.load(Gain + col, col < H, 0).to(_pn_tl.float32)
    inv = _pn_tl.rsqrt(_pn_tl.sum(x * x, 0) / H + Eps)
    y = (x * inv) * gain
    _pn_tl.store(Out + row * H + col, y, col < H)
    slot = _pn_tl.arange(0, 8)
    ids = _pn_tl.load(Ids + row * IdStride + slot).to(_pn_tl.int32)
    start = _pn_tl.load(Start).to(_pn_tl.int32)
    # Positive modulus 256 is exactly a low-eight-bit mask, including negatives.
    _pn_tl.store(LocalIds + row * 8 + slot, (ids - start) & 255)


def _pn_expert_inputs(x, ids, local_start, state):
    rows, hidden = x.shape
    gain = state['pre_expert_norm']._get_effective_weight()
    assert x.dtype == gain.dtype == torch.bfloat16 and hidden == 4096
    out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    if rows:
        _pn_inputs_kernel[(rows,)](
            x, gain, ids, local_start, out, local_ids, x.stride(0), ids.stride(0),
            hidden, RMS_EPS, _pn_triton.next_power_of_2(hidden), num_warps=4,
            enable_fp_fusion=False)
    return out, local_ids


def _achita_wide_router(x, state):
    # Both existing arms use dynamic M, masked stores and TMA boundary checks.
    # Keep the full FP32 small-token router and the 9..32 fusion unchanged.
    if x.shape[0] <= 32 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    rw = state['achita_dense'].get('bundle', {}).get('rw', {})
    if not rw.get('enabled') or x.device != rw['weight'].device:
        return None
    return rw_run_routerwide_candidate(x, rw, 128 if x.shape[0] <= 128 else 256)

"""Fuse implicit unpooler zero padding into the small-token FFN boundary.

Adapted from SGLang's Apache-2.0 fused_norm_residual at pinned revision
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f; the complete Apache license above applies.
"""
import triton as _sfn_triton
import triton.language as _sfn_tl


@_sfn_triton.jit
def _sfn_kernel(Branch, Stream, PostWeight, PreWeight, Alpha, Beta, Output, NewStream,
                BranchStride: _sfn_tl.constexpr, StreamStride: _sfn_tl.constexpr,
                H: _sfn_tl.constexpr, Slice: _sfn_tl.constexpr,
                PostEps: _sfn_tl.constexpr, PreEps: _sfn_tl.constexpr):
    row = _sfn_tl.program_id(0)
    col = _sfn_tl.arange(0, H)
    branch = _sfn_tl.load(Branch + row * BranchStride + col, col < Slice, 0).to(_sfn_tl.float32)
    post_weight = _sfn_tl.load(PostWeight + col).to(_sfn_tl.float32)
    stream = _sfn_tl.load(Stream + row * StreamStride + col).to(_sfn_tl.float32)
    alpha = _sfn_tl.load(Alpha + col).to(_sfn_tl.float32)
    beta = _sfn_tl.load(Beta + col).to(_sfn_tl.float32)
    pre_weight = _sfn_tl.load(PreWeight + col).to(_sfn_tl.float32)
    post_input = (branch * post_weight).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)
    # The implicit zero tail is part of the 8192-column RMSNorm domain.
    post_inv = _sfn_tl.rsqrt(_sfn_tl.sum(post_input * post_input, 0) / H + PostEps)
    norm_branch = (post_input * post_inv).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)
    updated = alpha * stream + beta * norm_branch
    _sfn_tl.store(NewStream + row * H + col, updated)
    # Default pre_norm_dtype=None: keep the residual input in FP32 here.
    pre_inv = _sfn_tl.rsqrt(_sfn_tl.sum(updated * updated, 0) / H + PreEps)
    _sfn_tl.store(Output + row * H + col, (updated * pre_inv) * pre_weight)


def _sfn_boundary(stream, branch, state):
    cached = state['sliced_ffn_boundary']
    if (not branch.shape[0] or cached is None or not branch.is_contiguous()
            or not stream.is_contiguous()):
        padded = F.pad(branch, (0, HIDDEN - UNPOOL_SLICE))
        output, updated, _ = state['ffn_boundary'](stream, padded, branch_dtype=torch.bfloat16)
        return output, updated
    output = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    updated = torch.empty_like(stream)
    _sfn_kernel[(branch.shape[0],)](
        branch, stream, cached['post_weight'], cached['pre_weight'], cached['alpha'], cached['beta'],
        output, updated, branch.stride(0), stream.stride(0), HIDDEN, UNPOOL_SLICE,
        cached['post_eps'], cached['pre_eps'], num_warps=8, enable_fp_fusion=False)
    return output, updated


_sfn_original_prepare = prepare


def prepare(weights):
    state = _sfn_original_prepare(weights)
    boundary = state['ffn_boundary']
    state['sliced_ffn_boundary'] = None
    if (boundary._post_norm_is_input_scale and boundary._use_residual_gate
            and boundary._post_norm.has_weight and boundary._pre_norm.has_weight
            and boundary._residual_add.gate.numel() == HIDDEN):
        boundary._ensure_cached_weights(torch.bfloat16)
        state['sliced_ffn_boundary'] = {
            'post_weight': boundary._cc_post_weight, 'pre_weight': boundary._cc_pre_weight,
            'alpha': boundary._cc_alpha, 'beta': boundary._cc_beta,
            'post_eps': boundary._post_norm.eps, 'pre_eps': boundary._pre_norm.eps}
    return state

# Additional exact FP4 placement: Copyright 2026 Lee Penkman, lee101/openkernels
# c565e373d34575c5f7df862b05f8be7619078b0f, Apache-2.0; full license retained above.
