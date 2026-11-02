# achitaan: ZipServ-style fixed-position BF16 experiment

## Check and decision gate

- Base: `bf16_v2.py` (scored 1.5886, per caller).
- Statistics-only source: `bf16_stats.py`, commit `7a1f0f6`.
- Statistics CHECK: `459b74a599b148d4`.
- Result: **passed**, check speedup 1.5594226, no flags. **Diagnostic failed**:
  the API and saved log contain only the final 16000 characters, and the base
  tuner's later output displaced every early `ZS` line. No measured coverage
  or packed-size decision is available from this result. Do not infer one from
  the unrelated legacy `STATS_E` lines that survived.
- Fixed retry source (not yet resubmitted): run diagnostics after parent
  prepare, compact floats, keep the complete ZS block below 14500 characters,
  and actually pack/bitwise-roundtrip pooler to report all buffer bytes.
  One additional diagnostic CHECK needs caller approval before continuing.
- Initial requested gate was <=11.16 bits/weight (7% below 12). Council
  refinement supersedes it: top-seven coverage >=95% **and actual packed
  buffers <=11.4 bits/weight**, including offsets, padding and escape payloads.
- Start the decoder on **pooler m=1/8 only**. Do not extend to qkv/unpooler
  unless pooler passes the existing correctness and ABBA/confirmation gates.
- Per-matrix and local-table lower bounds are printed separately from the
  naive per-32-weight uint32-index model. The latter costs one extra bit/weight
  and cannot pass the size gate even with zero escapes.

## Existing route structure

`bf16_v2.py`'s `_TABLE` is heterogeneous, not one GEMV family:

| Layer | m=1 | m=8 | m=16 | m=32 |
|---|---|---|---|---|
| qkv_gate | vjeux | chenwainuo | vjeux | chenwainuo |
| pooler | vjeux | chenwainuo | chenwainuo | vjeux |
| unpooler | vjeux | chenwainuo | vjeux | vjeux |

Vjeux uses `cgemv`/`sk2`; chenwainuo has register-direct `gv` and `rq`.
The existing DT arm interface already builds independent `gv` instances and
can compare them with the actual selected default, regardless of its family.
Use **FMT=6 in the existing gv kernel**, not a separate GEMV implementation.
Keep the original default dispatch, and all its existing gated DT choices.

## Fixed layout, initial implementation

`zs_codec.py` implements and independently roundtrips a proposed layout:

- Unit: existing `[4 K16 steps, 32 lanes, 8 values]` BF16 fragments.
- Group: all 32 values belonging to one lane of that unit.
- Byte plane: 7 mantissa bits plus sign, in the existing two uint4 load layout.
- Three uint32 bitplanes per group; bits at fixed value positions.
- Matrix table: eight exponent bytes; codes 1..7 selected by frequency, zero
  reserved for a full-BF16 escape. No float conversion of weight bits.
- Escapes: full uint16 words in unit/lane/position order.
- Index: one uint32 base per 1024-weight unit and one terminal sentinel.
  Each lane gets its group base via a warp prefix sum of escape-mask popcounts;
  an escape's within-group rank is one popcount of its mask prefix.

Total size before alignment is

`11 + 16*fallback + 32/1024 + 96/weight_count` bits/weight.

Thus the refined <=11.4 gate needs **97.6953125%** coverage in the large-matrix
limit. The initial <=11.16 gate needed 99.1953125%. Neither is implied by 95%.
Offsets and escapes occupy an actual combined byte array padded to 16 bytes;
the byte plane, bitplanes and eight-byte table are counted too. The packer's
`print_packed_statistics` emits effective bytes/bpw and the full component
breakdown in a `ZS` line from those actual arrays, after exact roundtrip.
The already-submitted statistics job cannot contain this later refinement;
its payload-only numbers would only be a screening lower bound, not admission.

The reference tests cover all 65536 BF16 encodings (including NaN payloads),
no-escape data, boundary escapes, mixed special values, and seeded random data.
These are codec/layout checks, not GPU performance or real-weight validation.

## Integration if the statistics qualify

1. Extend `GL` and `gv` with `ZS = FMT == 6`, `NL=2`, `NA=1`, no old c12
   escape walk and no Codec E LUT/ring. Retain the original X load, MMA,
   accumulator, and reduction order.
2. Use `W` for byte planes, `W2` for the three uint32 planes. Extend the
   register prefetch ring with three plane words per lane/unit. Reuse the
   existing escape argument for offsets followed by the uint16 escape array;
   table remains two register uint32 values.
3. Form each lane mask as `~(p0|p1|p2)` and use a five-stage warp scan once
   per 32-weight group, not a variable-length per-weight decode chain. Use
   fixed-position exponent selections with register byte-permute lookups;
   only escaping positions load the side array. Inspect compiled instruction
   count before claiming the requested ~3-5 integer operations per weight:
   unpacking actual bitplanes into tensor-core fragments has a nonzero cost
   that cannot be inferred from a payload compression ratio.
4. Register only pooler m=1/8 and same-geometry raw BF16 twins in `_DTGVI`;
   make the real
   weight pack/unpack bit comparison mandatory in prepare. Match twin math
   bitwise on normal/outlier/one-hot inputs using `_dt_twin`.
5. Compare against the actual incumbent per cell through `_dt_cell`; require
   both median/paired ABBA ratios <=0.988 in both passes plus fresh-input
   confirmation. Never choose an arm on estimated bandwidth alone.
6. Print actual packed effective bpw and reject >11.4 before any ZS candidate
   launch. Extend beyond pooler only after a measured pooler win.
7. Materialize readable `bf16_zs.py`, AST-check with Python 3.12, verify LF,
   zero forbidden-source tokens and <900KB, then submit one CHECK.

Source concept: HPMLL/ZipServ_ASPLOS26, Apache-2.0;
arxiv.org/html/2603.17435. The initial codec is independently written; no
external source or compressed source code is embedded.
