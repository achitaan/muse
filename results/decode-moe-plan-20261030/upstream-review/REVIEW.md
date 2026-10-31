# Public upstream revision review

Reviewed 2026-10-31 UTC. Read-only source research; no GPU work, candidate edits or source imports.

The retained MoE header pins `vjeux/okc-2026` revision `f35a41ee1c13e316dabbf772830d47b23b453393`. An anonymous public repository inspection and `git ls-remote https://github.com/vjeux/okc-2026.git HEAD` returned that same full revision. There is no newer upstream HEAD delta to import from this repository. This does not claim that our modified retained MoE is byte-identical to upstream, or rule out research elsewhere.

Sources: [public repository](https://github.com/vjeux/okc-2026), [pinned revision](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393), retained `best/moe_verified.py` attribution header.

The connected GitHub tool required an unavailable connection identifier. Public API retrieval was unavailable through the web tool; Python's default certificate verification also failed. Neither authentication nor TLS verification was bypassed. The existing Git client completed the public metadata read with normal verification.

Decision: no implementation or GPU check justified by this upstream lookup. Keep the frozen M1 diagnostic unchanged.
