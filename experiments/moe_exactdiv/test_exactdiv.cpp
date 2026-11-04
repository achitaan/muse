// Native CPU correctness model, not a CUDA compile or a performance benchmark.
// helpers.hpp is generated from the submitted CUDA index functions verbatim,
// except for host attribute stubs and correction-count observation counters.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

int approximation_ulps = 0;
uint64_t division_calls = 0, correction_steps = 0;
unsigned max_corrections = 0;

inline float __fdividef(float n, float d) {
  // Native FP32 division plus adversarial error, beyond the +/-2-ulp range
  // normally relevant to device approximate division. Input conversions also
  // round to FP32, exactly as in the submitted function.
  float q = n / d;
  if (q != 0.f) {
    uint32_t bits;
    std::memcpy(&bits, &q, sizeof(bits));
    bits += approximation_ulps;
    std::memcpy(&q, &bits, sizeof(bits));
  }
  return q;
}

inline uint32_t __float2uint_rz(float q) {
  assert(std::isfinite(q) && q >= 0.f && q < 4294967296.f);
  return static_cast<uint32_t>(q);
}

struct Dim { uint32_t x; } blockIdx, gridDim;
constexpr int JMAX = 88, WGS_SLOTS = 160;
struct Params {};
struct Smem {
  int jC[JMAX] = {}, wcp[JMAX + 1] = {}, wcpB[JMAX + 1] = {};
  uint32_t wtot = 0, wtotB = 0;
  int njF = 0, njS = 0, wgr[4] = {};
};

#define __device__
#define __forceinline__ inline
#define __noinline__
#include "helpers.hpp"

uint64_t quotient_cases = 0, range_cases = 0, owner_cases = 0, zero_cases = 0;

void check_division(uint32_t n, uint32_t d) {
  assert(d > 0);
  // Independent expected value: exact native unsigned integer division.
  const uint32_t want = n / d;
  assert(want < (1u << 24));
  const auto got = candidate::wgs_div_u32(n, d);
  if (got != want) {
    std::fprintf(stderr, "division n=%u d=%u ulps=%d got=%u expected=%u\n",
                 n, d, approximation_ulps, got, want);
    std::exit(1);
  }
  ++quotient_cases;
}

void check_plan(const std::vector<int>& counts, const std::vector<int>& grids,
                std::mt19937& rng) {
  const int nj = static_cast<int>(counts.size());
  Smem ref, got;
  ref.njF = got.njF = nj;
  for (int j = 0; j < nj; ++j) ref.jC[j] = got.jC[j] = counts[j];
  reference::wgs_prefix(ref, nj);
  candidate::wgs_prefix(got, nj);
  assert(ref.wtot == got.wtot && ref.wtotB == got.wtotB);
  for (int j = 0; j <= nj; ++j)
    assert(ref.wcp[j] == got.wcp[j] && ref.wcpB[j] == got.wcpB[j]);

  for (int pb = 0; pb < 2; ++pb) {
    const uint32_t tot = pb ? ref.wtotB : ref.wtot;
    for (int G : grids) {
      gridDim.x = G;
      for (int c = 0; c < G; ++c) {
        blockIdx.x = c;
        int a0, a1, b0, b1;
        reference::wgs_range(ref, nj, pb, c, G, a0, a1);
        candidate::wgs_range(got, nj, pb, c, G, b0, b1);
        if (a0 != b0 || a1 != b1) {
          std::fprintf(stderr, "range nj=%d G=%d pb=%d c=%d ulps=%d: [%d,%d) != [%d,%d)\n",
                       nj, G, pb, c, approximation_ulps, a0, a1, b0, b1);
          std::exit(1);
        }
        ++range_cases;
        // Actual split-unit callback sites plus a random unit: compare the
        // original function, the new function and interval containment.
        const int units = nj << (5 - pb);
        const int queries[] = {a0 & ~15, (a1 > a0 ? a1 - 1 : a0) & ~15,
                               static_cast<int>(rng() % units) * 16};
        for (int s : queries) {
          if (s >= (nj << (9 - pb))) continue;
          const int old_owner = reference::wgs_cta_of(ref, pb, s, G);
          const int new_owner = candidate::wgs_cta_of(got, pb, s, G);
          if (old_owner != new_owner) {
            std::fprintf(stderr, "owner nj=%d G=%d pb=%d s=%d ulps=%d got=%d expected=%d\n",
                         nj, G, pb, s, approximation_ulps, new_owner, old_owner);
            std::exit(1);
          }
          int begin, end;
          reference::wgs_range(ref, nj, pb, new_owner, G, begin, end);
          assert(begin <= s && s < end);
          ++owner_cases;
        }
      }
      // The full device wrapper must produce the exact same four ranges.
      blockIdx.x = rng() % G;
      candidate::wgs_ranges(Params{}, got);
      reference::wgs_range(ref, nj, 0, blockIdx.x, G, ref.wgr[0], ref.wgr[1]);
      reference::wgs_range(ref, nj, 1, blockIdx.x, G, ref.wgr[2], ref.wgr[3]);
      for (int k = 0; k < 4; ++k) assert(ref.wgr[k] == got.wgr[k]);
      assert(static_cast<uint64_t>(tot) * G < (1ull << 32));
    }
  }
}

int main() {
  std::mt19937 rng(20261104);
  constexpr uint32_t MAX_TOTAL = JMAX * 512 * 255;
  static_assert(MAX_TOTAL < (1u << 24), "finite representable scheduler quotient");
  std::vector<int> all_grids;
  for (int g = 1; g <= WGS_SLOTS; ++g) all_grids.push_back(g);
  const std::vector<int> selected_grids = {1, 2, 3, 31, 32, 33, 64, 128, 131, 132, 133, 159, 160};

  for (int error : {-4, -2, -1, 0, 1, 2, 4}) {
    approximation_ulps = error;
    // Exhaustive small quotients, including n=0 and exact/inexact division.
    for (uint32_t n = 0; n <= 4096; ++n)
      for (uint32_t d = 1; d <= 256; ++d) check_division(n, d);

    // The three scheduler domains, not arbitrary integer combinations: range
    // quotient <= total cost, stage quotient <= 511, owner quotient <= G-1.
    for (int i = 0; i < 100000; ++i) {
      const uint32_t total = 1 + rng() % MAX_TOTAL;
      const uint32_t G = 1 + rng() % WGS_SLOTS;
      const uint32_t c = rng() % (G + 1);
      check_division(total * c, G);
      const uint32_t cost = 1 + rng() % 255;
      check_division(rng() % (512 * cost), cost);
      const uint32_t xe = 1 + rng() % total;
      check_division(xe * G - 1, total);
    }

    // Integer boundaries at maximum representable job/CTA cost, including
    // every endpoint of one full WGS unit and all eight n-tile counts.
    for (uint32_t G = 1; G <= WGS_SLOTS; ++G) {
      for (uint32_t total : {1u, 512u * 112u, MAX_TOTAL - 1u, MAX_TOTAL}) {
        for (uint32_t c = 0; c <= G; ++c) check_division(total * c, G);
        for (uint32_t xe : {1u, total / 2 + 1, total}) check_division(xe * G - 1, total);
      }
    }

    check_plan({1}, all_grids, rng);
    check_plan({64}, all_grids, rng);
    check_plan(std::vector<int>(JMAX, 64), selected_grids, rng);
    check_plan({1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64}, all_grids, rng);
    for (int i = 0; i < 80; ++i) {
      const int nj = 1 + rng() % JMAX;
      std::vector<int> counts(nj);
      for (auto& count : counts) count = 1 + rng() % 64;
      check_plan(counts, selected_grids, rng);
    }

    for (int G : all_grids) {
      gridDim.x = G;
      for (int c = 0; c < G; ++c) {
        blockIdx.x = c;
        Smem empty;
        empty.wtot = empty.wtotB = 1234;
        std::fill(std::begin(empty.wgr), std::end(empty.wgr), -777);
        candidate::wgs_ranges(Params{}, empty);
        assert(empty.wtot == 0 && empty.wtotB == 0);
        for (int v : empty.wgr) assert(v == 0);
        ++zero_cases;
      }
    }
  }

  // For the actual scheduler domain and even +/-4-ulp approximate division,
  // excessive correction iterations would fail the intended performance gate.
  assert(max_corrections <= 8);
  std::printf("PASS EXACTDIV=%d quotient_cases=%llu range_cases=%llu owner_cases=%llu zero_cases=%llu "
              "instrumented_divisions=%llu correction_steps=%llu max_corrections=%u\n",
              OKC_EXACTDIV, static_cast<unsigned long long>(quotient_cases),
              static_cast<unsigned long long>(range_cases), static_cast<unsigned long long>(owner_cases),
              static_cast<unsigned long long>(zero_cases), static_cast<unsigned long long>(division_calls),
              static_cast<unsigned long long>(correction_steps), max_corrections);
}
