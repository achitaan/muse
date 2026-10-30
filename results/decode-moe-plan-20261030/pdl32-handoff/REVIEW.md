# M32 sliding handoff integration: CPU draft

`decoder_pdl32_handoff_draft.py` is source57f3accbe89aa8ec21e330ea4232f268f242fff84fae72e1ac67995f0cd72bc1,956942 bytes. Its first940518 bytes are exactly the retained c15df9d3 Decode source. It is unsubmitted and not admitted to the final Decode check.

## Source-bound evidence

O32 qualified in H200 check ec1ea26451494f02: finite bytes match changing normal/zero/outlier inputs; synthetic chains save4.096/4.224us in the two arm orders. The legacy host factory left the private kernel at49152 maximum dynamic bytes and zero resident blocks. Its isolated factory set196736 bytes and one resident block;42 registers,zero local/static shared. The exact ELF binding type was not inspected.

U32 qualified in check3264689ee9974ef9: bytes match; synthetic chains save3.712/3.936us. H200 reported90 registers,zero local bytes and132 active clusters. Both measurements used an optimistic one-CTA5000ns producer,not the real layer. Their savings are not additive measured layer gains.

The frozen O32 and U32 sources are exact substrings of the tested09b371d0 projection bundle. The new source encloses each in a unique namespace after its includes,then adds typed host aliases and a resource query. Removing those namespace openings/closings exactly reconstructs each original source. All device arithmetic and synchronization text is unchanged. Namespacing also isolates host template statics and map-function caches.

## Fixed implementation

Only B32,L1,M32 sliding may select the bundle. O32 and U32 are registered together,with no component sweep. Full context,other sizes/layouts/dtypes,and sliced-unpool rows retain the original dispatch. No MoE,Attention,router,norm,PDL-flag,weight or scratch change is introduced. The generic layer continues to execute the parent function; only its two eligible projection calls receive isolated kernels through a separate candidate-state dictionary.

The retained MoE already issues an outgoing scheduling signal before its final barrier/reduction. Its false PDL host flag controls the incoming launch attribute. CUDA permits an ordinary primary launch followed by an incoming-PDL secondary launch; concurrency remains opportunistic. U32 therefore requires no expert-code or flag change. The actual overlap window remains unmeasured. [NVIDIA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

## Admission gates

The extension compiles in a background thread and must be joined before probing/capture. Actual H200 opt-in,register/local-memory and occupancy queries must pass. Both leaf kernels must pass finite byte comparisons on changing normal/zero/outlier inputs. The complete parent and candidate layers must then pass the retained shared gate: normalized output,residual,routing and entire K/V-cache bytes on three changing graph inputs,and more than3% improvement in each cold-L2 graph arm order. Selection starts false and is set only after all these gates pass. No new diagnostic starts after274s; the retained timing gate stops at278s.

The dictionaries share the parent's sequential-scratch contract. They contain no cached routing/input decisions. No compiler thread may remain active during capture. Eager warm-up occurs before graph capture. A fresh preparation's300s limit and actual H200 integration remain unverified.

## CPU verification and limits

Source/namespace inverses,explicit Pybind header,18 dispatcher cases and17 projection fallback cases pass. Namespaced SM90 device proxies compile: O32 uses42 registers,U32 uses88;both zero stack/spills. SASS confirms eight O32 bulk weight copies before its wait,and eight U32 independent loads before its wait;dependent activation TMA follows each wait. Prior H200 U32 used90 registers,so CPU counts do not replace the runtime gates.

Historical M32 sliding raw graph time165.680us implies a3% threshold of4.970us. The two isolated ceilings establish a plausible experiment,not a forecast or admission proof. Real predecessor tails,overlap contention,whole-layer numerical replay,selected-path execution and preparation headroom are still unresolved. The final Decode check remains reserved; no scoring/publication follows from this CPU draft.

The published retained result remains2.075466170x. No current rank or score improvement was measured by this packet.
