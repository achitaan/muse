from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
s = s.replace("- **Dense top-50 cutoff (snapshot, not refreshed):** `1.5613x` \u2192 needs about +2.19% overall",
              "- **Dense top-50 cutoff (refreshed 2026-10-30 22:05 UTC):** `1.5698x` (we are rank 229/1497) \u2192 needs about +2.75% overall")
row = ("| P1 | 2026-10-30 | Optimizer loop pass 1 (PROGRESS.md H1): qkv_gate M1/M8/M16 wave balance. Profile shows qkv M1/M8 at ~3.0 TB/s "
       "effective vs pooler M1 3.7 TB/s; incumbent v18d `(1,4,8,4)` = 576 CTAs (4.36/SM). Same v18d native c12 kernel and pack, geometries "
       "`(1,4,4,8)` `(1,8,2,8)` `(1,2,8,8)` `(1,8,4,4)` `(1,16,8,1)` `(2,4,4,8)` `(1,4,2,16)`; one module per geometry. Never tested before "
       "(S1 covered only the sliced unpooler). Gates: v18d_v18d_native_plan (bit-exact pack, one-hot, FP64 random) + official checker; "
       "ABBA 16 pairs, median >= 1.03 and wins >= 75%; markers P1_USE/P1_SELECT/P1_KEEP | pending | pending | | logged |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
print('ok', s.count('1.5698'))
