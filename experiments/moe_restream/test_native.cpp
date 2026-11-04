// Actual extracted index helpers compiled as CPU C++; no CUDA claims.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using std::min; using std::max;
static_assert(__cplusplus >= 202002L, "native preflight must use the shared C++20 dialect");
constexpr int JMAX = 88, NW = 16, NRB = 2, RED_W = 4, EMAX = 3072;
struct Dim { unsigned x; } blockIdx, gridDim;
struct Params {};
#define __device__
#define __forceinline__ inline
#define __noinline__
#include "metadata.hpp"

int main() {
  uint64_t outputs = 0, owner_checks = 0, contributors = 0;
  for (int count = 1; count <= 256; ++count) {
    Smem S{}; S.njF = 1; S.jC[0] = count;
    wgs_prefix(S, 1);
    const int R = rs_rows(count);
    int done[16] = {};
    for (int pb = 0; pb < 2; ++pb) {
      std::vector<uint8_t> visits(4096 * count, 0);
      const int units = (pb ? 64 : 128) / R;
      for (int u = 0; u < units; ++u) {
        assert(rs_job(S, pb, u) == 0);
        for (int wg = 0; wg < 4; ++wg) {
          const int c = rs_count(S, 0, wg), ntbase = rs_ntbase(S, 0, wg);
          if (!c) continue;
          if (!pb) { ++done[rs_done_index(S, 0, u)]; ++contributors; }
          for (int wp = 0; wp < 4; ++wp) {
            const int r16 = rs_r16(S, 0, u, wg, wp);
            for (int lane = 0; lane < 32; ++lane) {
              const int g = lane >> 2, t = lane & 3;
              for (int nt = 0; nt < (c + 7) / 8; ++nt) {
                for (int h = 0; h < 2; ++h) {
                  const int tk = (ntbase + nt) * 8 + 2 * t + h;
                  if (tk >= count) continue;
                  const int col = pb ? r16 * 16 + g : r16 * 8 + g;
                  assert(col >= 0 && col < 4096);
                  assert(++visits[tk * 4096 + col] == 1);
                  ++outputs;
                  if (pb) {
                    assert(col + 8 < 4096);
                    assert(++visits[tk * 4096 + col + 8] == 1);
                    ++outputs;
                  }
                }
              }
            }
          }
        }
      }
      for (auto v : visits) assert(v == 1);
    }
    for (int st = 0; st < 16; ++st) assert(done[st] == 8 * ((count + 63) / 64));
  }
  std::mt19937 rng(20261106);
  for (int trial = 0; trial < 1000; ++trial) {
    Smem S{};
    int counts[32] = {}, routes = rng() % 2049;
    for (int i = 0; i < routes; ++i) {
      int e = rng() % 32;
      if (counts[e] < 256) ++counts[e];
    }
    if (trial % 3 == 0) { std::fill(counts, counts + 32, 0); counts[trial % 32] = 65 + rng() % 192; }
    if (trial < 8) std::copy(actual_counts[trial], actual_counts[trial] + 32, counts);
    for (int c : counts) if (c > 8) S.jC[S.njF++] = c;
    for (int c : counts) if (c > 0 && c <= 8) S.jC[S.njF + S.njS++] = c;
    const int nj = S.njF + S.njS;
    if (!nj) {
      gridDim.x = 132; blockIdx.x = 0;
      wgs_ranges(Params{}, S);
      for (int v : S.wgr) assert(v == 0);
      continue;
    }
    wgs_prefix(S, nj);
    for (int G : {1, 17, 132, 160}) {
      gridDim.x = G;
      for (int pb = 0; pb < 2; ++pb) {
        const auto total = pb ? S.wtotB : S.wtot;
        assert((uint64_t)total * G < (1ull << 32));
        int prev = 0;
        for (int c = 0; c < G; ++c) {
          int a, b;
          wgs_range(S, nj, pb, c, G, a, b);
          assert(a == prev && a <= b); prev = b;
          if (trial < 8) {
            const int sp = wgs_split(pb, a, b);
            std::vector<int> producer, consumer;
            for (int i = 0, gs = sp; i < b - a; ++i, ++gs) {
              if (gs == b) gs = a;
              producer.push_back(gs);
            }
            for (int gs = sp, ge = b;;) {
              if (gs >= ge) { if (ge == b && sp > a) { gs = a; ge = sp; } else break; }
              int end = min(32, (gs & 31) + ge - gs), n = end - (gs & 31);
              for (int k = 0; k < n; ++k) consumer.push_back(gs + k);
              gs += n;
            }
            assert(producer == consumer);
            std::sort(producer.begin(), producer.end());
            for (int k = 0; k < b - a; ++k) assert(producer[k] == a + k);
          }
          // All split-unit first-stage queries must map to an interval owner.
          for (int u : {a / 32, max(a, b - 1) / 32}) {
            if (u >= S.wup[nj] * (pb ? 1 : 2)) continue;
            const int s = u * 32, j = rs_job(S, pb, u);
            assert(j >= 0 && j < nj);
            const int owner = wgs_cta_of(S, pb, s, G);
            int begin, end; wgs_range(S, nj, pb, owner, G, begin, end);
            assert(begin <= s && s < end);
            ++owner_checks;
          }
        }
        assert(prev == (S.wup[nj] << (6 - pb)));
      }
    }
  }
  const size_t smem = ((sizeof(Smem) + 127) / 128) * 128 + 3 * 18432 + 2 * 65536;
  assert(smem <= 227328);
  // Existing storage/reset covers the new semantic remapping, not new flags.
  static_assert(32 * 16 <= 88 * 16 && 32 * 16 <= 136 * 16, "flag capacity");
  std::printf("PASS outputs=%llu owner_checks=%llu contributors=%llu smem=%zu sizeofSmem=%zu\n",
              (unsigned long long)outputs, (unsigned long long)owner_checks,
              (unsigned long long)contributors, smem, sizeof(Smem));
}
