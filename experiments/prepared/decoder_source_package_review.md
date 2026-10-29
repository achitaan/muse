# Decode source packaging and preparation review

Authorization: local implementation and CPU verification only. GPU checks,
profiling jobs and submissions remain unauthorized. Work stays in the main
chat with the selected model. This packet changes no active or verified file.

## Exact sources and result

Parent: `best/decoder_verified.py`,
`d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988`,
989,018 bytes. Its historical check is `9560c915e85f4f50` at 1.990224880x.
That result belongs to D9, not this package.

Package: `bc528949d73aa09ae9a330d030cadd5654f9ff7d20d48e8ff26bbeed96ae1c43`,
893,369 bytes. It saves 95,649 bytes and leaves 106,631 bytes below the
1,000,000-byte cap. It has no GPU evaluation or performance result.

Frozen source and evidence:
`results/decoder-source-package-20261029/bc528949d73aa09ae9a330d030cadd5654f9ff7d20d48e8ff26bbeed96ae1c43/`.

## Transformation and CPU verification

The builder encodes only `_native_moe_CUDA_SRC` as base64 of zlib-compressed
UTF-8. It moves the existing `_zsrc` loader and its existing standard-library
imports before this new use. It adds no dependency or kernel. All numerical
boundaries, dispatch, fallback paths, preparation functions and licenses stay
in the package. No weight or activation is compressed by this change.

The independent static verifier establishes:

- Decoding recovers all 139,873 original CUDA bytes, SHA256
  `1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`.
- Expanding the literal and restoring the loader's location recovers D9's
  entire module AST, including every preparation and runtime function.
- Reversing the three source edits recovers every D9 byte, including notices.
- Mutations to a decoded CUDA byte, the loader body and the FP32-router token
  boundary are each rejected at their corresponding equivalence assertions.
- The problem header and source cap remain valid. No submission code is
  executed; the audit imports only Python's standard library.

These are source-equivalence checks, not GPU validation. No CUDA compilation,
graph replay, numerical GPU probe, service query or quota use occurred.

CPU reproduction, from the repository root:

```text
python experiments/prepared/build_decoder_source_package.py
python experiments/prepared/verify_decoder_source_package.py --freeze
python experiments/prepared/inspect_decoder_preparation.py
python experiments/prepared/inspect_decoder_preparation.py --tracked-evidence-only
```

Frozen evidence refuses replacement by different bytes. Compression was built
with zlib 1.2.11; another compressor version may emit different encoded bytes.
Use the frozen exact package for any later authorized evaluation.

## Preparation evidence and remaining gate

The AST map records 13 preparation functions in dependency order and the
existing dense pool of 21 jobs with four workers. This is a static call map,
not a measured GPU timeline. The two jobs using `tt_v10_m32x_prepare_m32x`
have different weights/options, so their shared function name does not prove
duplicate work. The final WG sweep is already disabled in D9.

`preparation-review.json` is the current portable report. It binds the exact
tracked `parent-result.json` and `parent-launch.json` to D9 and job9560c915.
The original local-cache `preparation-map.json` remains preserved. The
tracked-evidence mode reproduces the review without an ignored cache or API
access; the builder and static verifier run first to restore `.tools` files.

The exact cached D9 check records 256.896945766 seconds of preparation. The
historical difference from its 300-second limit is 43.103054234 seconds. The
package removes no preparation work and has no new setup measurement. Treat
that difference as historical evidence, not guaranteed integration headroom.

The standalone Dense D1 add-on is 49,699 bytes. Adding that raw increment to
this package would total 943,068 bytes before adapting its wrapper and state,
leaving 56,932 bytes. This establishes source capacity for that increment;
it does not establish a legal or ready Decode integration.

D1 still requires one whole-layer preparation budget, correct Dense-state
integration, nearby-shape performance evidence and a worthwhile whole-layer
benefit. Its standalone clock gates cannot independently protect the complete
Decode preparation chain. Preserve every currently reachable fallback/build
until source and dispatch equivalence establish a valid simplification.

Do not promote or score this packaging-only source for speed. Keep D9 and all
other verified incumbents unchanged while preparing a justified next candidate.
