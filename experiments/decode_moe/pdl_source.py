"""Exact, bounded PDL twins of the R2f-selected projection sources."""
import hashlib


def edit_once(text, old, new, edits):
    assert text.count(old) == 1, (old[:100], text.count(old))
    edits.append((old,new))
    return text.replace(old,new)


def twin(source, prefix, inst, *, fused=False, form='inline'):
    """Only the fixed packed, direct-x, noncluster instances are eligible here.

    The wait remains on every dependent route; the extra R2f top-k block never
    enters the ordinary weight preamble. Original math and store code is kept.
    """
    assert inst[0] == 1 and inst[1:3] == (1,1)
    assert inst[4:6] == (0,0) and inst[13] == 1 and inst[14] & 2
    edits=[];s=source
    if '@@CASES@@' in s:s=edit_once(s,'@@CASES@@','',edits)
    # The dependent activation uses the coherent L2 path after synchronization.
    s=edit_once(s,'ld.global.nc.v4.u32','ld.global.cg.v4.u32',edits)
    for anchor in (f'if constexpr (C12 && !RO) my_e = {prefix}_lde(esc + (size_t)gw * 32 + lane);',):
        assert s.count(anchor)==2
        s=s.replace(anchor,'// PDL: escape word is loaded by the independent preamble.')
        edits.append((anchor,'// PDL: escape word is loaded by the independent preamble.'))
    if fused:
        old=(f'if (blockIdx.x == 128) {{ {prefix}_r2_topk(ra, warp, lane); return; }}\n'
             f'      {prefix}_r2_router(ra, warp, lane);')
        new=(f'if (blockIdx.x == 128) {{ asm volatile("griddepcontrol.wait;" ::: "memory"); '
             f'{prefix}_r2_topk(ra, warp, lane); return; }}\n'
             '      // PDL: router input loads follow the independent weight preamble.')
        s=edit_once(s,old,new,edits)
        # FP32 input reads must not use a stale noncoherent cache on replay.
        old='r2x[i] = ra.x[(size_t)(i >> 10) * 8192 + ks * 1024 + (i & 1023)]'
        new=('asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(r2x[i]) : '
             '"l"(ra.x + (size_t)(i >> 10) * 8192 + ks * 1024 + (i & 1023)) : "memory")')
        s=edit_once(s,old,new,edits)
    st,rt,nl=inst[3],inst[2],3
    locations=[(a,b,c,d) for a in range(st) for b in range(rt) for c in range(nl) for d in range(4)]
    if form=='inline':
        outputs=[f'"=&r"(w[{a}][{b}][{c}].{"xyzw"[d]})' for a,b,c,d in locations]
        outputs.append('"=&r"(my_e)')
        nout=len(outputs)
        addresses=[f'"l"(wq + (size_t){a} * sstride + {b*nl+c} * 32)' for a in range(st) for b in range(rt) for c in range(nl)]
        addresses.append('"l"(esc + (size_t)gw * 32 + lane)')
        asm=[]
        for j in range(len(addresses)-1):
            asm.append('ld.global.nc.L1::no_allocate.v4.s32 {'+','.join('%'+str(j*4+d) for d in range(4))+'}, [%'+str(nout+j)+'];')
        asm.append(f'ld.global.nc.u32 %{nout-1}, [%{nout+len(addresses)-1}];')
        asm.append('griddepcontrol.wait;')
        preamble=('static_assert(ST == '+str(st)+' && RT == 1 && C12 && (OPT & 2), "fixed PDL preamble");\n'
                  '    asm volatile("'+ ' '.join(asm)+'"\n      : '+', '.join(outputs)+'\n      : '+', '.join(addresses)+' : "memory");')
    elif form=='call':
        helper=(f'__device__ __noinline__ void {prefix}_pdl_wait(uint32_t checksum) {{\n'
                '  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");\n}\n')
        s=edit_once(s,'// Never taken, but ptxas cannot prove it:',helper+'\n// Never taken, but ptxas cannot prove it:',edits)
        refs=[f'(uint32_t)w[{a}][{b}][{c}].{"xyzw"[d]}' for a,b,c,d in locations]
        preamble=(f'#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n'
                  f'    my_e = {prefix}_lde(esc + (size_t)gw * 32 + lane);\n'
                  f'    {prefix}_pdl_wait(my_e ^ '+ ' ^ '.join(refs)+');')
    else:raise ValueError(form)
    if fused:preamble+=f'\n    {prefix}_r2_router(ra, warp, lane);'
    s=edit_once(s,'#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);',preamble,edits)
    # Preserve the cluster attribute when a future eligible launcher has one;
    # all four currently selected twins use RED0/S1.
    s=edit_once(s,'cudaLaunchAttribute at[1];','cudaLaunchAttribute at[2];',edits)
    anchor='  if (query == 1) {'
    extra=('  const int pdl_attr = lc.numAttrs;\n'
           '  at[pdl_attr].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n'
           '  at[pdl_attr].val.programmaticStreamSerializationAllowed = 1;\n'
           '  lc.attrs = at; lc.numAttrs = pdl_attr + 1;\n')
    s=edit_once(s,anchor,extra+anchor,edits)
    inverse=s
    for a,b in reversed(edits):
        assert inverse.count(b)==(2 if a.startswith('if constexpr (C12') else 1)
        inverse=inverse.replace(b,a)
    assert inverse==source
    return s,dict(parent_CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  CUDA_sha256=hashlib.sha256(s.encode()).hexdigest(),form=form,fused=fused,
                  instance=inst,exact_inverse=True,edits=[dict(old=a,new=b) for a,b in edits])
