"""Build decoder_r2f_a3.py: the scored R2f file plus the Attention board's A3 tuning (decode 16 sliding: n_splits 8)."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = ("_native_attn__TUNED = {(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1), "
       "(1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)}")
NEW = ("_native_attn__TUNED = {(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1), "
       "(1, 1, False, 1): dict(nc=8, coop=1, n_splits=252), (16, 1, True, 1): dict(n_splits=8, nc=1)}")

src = (HERE / 'decoder_r2f.py').read_bytes().decode('utf-8')
assert hashlib.sha256(src.encode()).hexdigest().startswith('75fa2a73'), 'must start from the scored R2f file'
assert src.count(OLD) == 1
out = src.replace(OLD, NEW)
ast.parse(out)
(HERE / 'decoder_r2f_a3.py').write_bytes(out.encode('utf-8'))
print(len(out.encode()), hashlib.sha256(out.encode()).hexdigest())
