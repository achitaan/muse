"""Source-only changes to pinned FS1; defaults retain the exact FS1 math."""
def diagnostic_source(src):
    def rep(old,new):
        nonlocal src
        assert src.count(old)==1,(old[:100],src.count(old))
        src=src.replace(old,new)
    prefix=r'''
#ifndef FS_TL
#define FS_TL 0
#endif
#ifndef FS_NOB
#define FS_NOB 0
#endif
#ifndef FS_NOFLUSH
#define FS_NOFLUSH 0
#endif
#ifndef FS_NOA
#define FS_NOA 0
#endif
#ifndef FS_NOACC
#define FS_NOACC 0
#endif
#if FS_NOA
__device__ __forceinline__ uint4 fsd_gload(const uint4* p){
  uint4 v;asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
__device__ __forceinline__ uint4 fsd_sload(const void* p){
  uint4 v;asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"((unsigned)__cvta_generic_to_shared(p)):"memory");return v;
}
#endif
#if FS_TL
// One lane/warp writes private160-word trace:11 header fields,64 A/B
// pairs at16, up to8 segment start/end pairs at144. No new CTA barriers.
__device__ __forceinline__ unsigned long long fsd_clock(){
  unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
#define FS_STAMP(i) do {if(lane==0) p.trace[((size_t)blockIdx.x*9+warp)*160+(i)]=fsd_clock();}while(0)
#define FS_META(i,v) do {if(lane==0) p.trace[((size_t)blockIdx.x*9+warp)*160+(i)]=(v);}while(0)
#else
#define FS_STAMP(i) do {}while(0)
#define FS_META(i,v) do {}while(0)
#endif
'''
    rep('namespace fs {',prefix+'\nnamespace fs {')
    rep('  int M,lo,zero;', '  int M,lo,zero;\n#if FS_TL\n  unsigned long long* trace;\n#endif')
    rep('  asm volatile("griddepcontrol.wait;":::"memory");',
        '  FS_STAMP(0);\n  asm volatile("griddepcontrol.wait;":::"memory");')
    rep('  __syncthreads();\n  if(warp==8)',
        '  __syncthreads();\n  FS_STAMP(1);FS_META(7,S.b-S.a);FS_META(8,S.a);FS_META(9,S.b);\n  if(warp==8)')
    rep('    return;\n  }','    FS_STAMP(6);\n    return;\n  }')
    rep('  int task=S.a,seq=0;','  int task=S.a,seq=0;\n#if FS_TL\n  int segment=0;\n#endif')
    rep('for(int c=0;c<8;++c) {x[2*c]=__ldg(src+c*8);x[2*c+1]=__ldg(src+c*8+1);}',
        'for(int c=0;c<8;++c) {\n#if FS_NOA\n        x[2*c]=fsd_gload(src+c*8);x[2*c+1]=fsd_gload(src+c*8+1);\n#else\n        x[2*c]=__ldg(src+c*8);x[2*c+1]=__ldg(src+c*8+1);\n#endif\n      }')
    rep('        const uint4 sc=*reinterpret_cast<const uint4*>(slot+32768+warp*512+lane*16);',
        '#if FS_NOA\n        const uint4 sc=fsd_sload(slot+32768+warp*512+lane*16);\n#else\n        const uint4 sc=*reinterpret_cast<const uint4*>(slot+32768+warp*512+lane*16);\n#endif')
    rep('          const uint4 w=*reinterpret_cast<const uint4*>(slot+(warp*8+c)*512+lane*16);',
        '#if FS_NOA\n          const uint4 w=fsd_sload(slot+(warp*8+c)*512+lane*16);\n#else\n          const uint4 w=*reinterpret_cast<const uint4*>(slot+(warp*8+c)*512+lane*16);\n#endif')
    rep('        int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);',
        '        int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);\n        if(seq==0) {FS_STAMP(2);}')
    rep('          for(int k=0;k<4;++k) {\n            uint32_t r0,r1,r2,r3;dq4',
        '          for(int k=0;k<4;++k) {\n#if FS_NOA\n            // Volatile asm consumes ALL original A/x loads; no dequant or MMA.\n            asm volatile(""::"r"(ww[k]),"r"(sw),"r"(xx[2*k]),"r"(xx[2*k+1]):"memory");\n#else\n            uint32_t r0,r1,r2,r3;dq4')
    rep('xx[2*k],xx[2*k+1]);\n          }','xx[2*k],xx[2*k+1]);\n#endif\n          }')
    rep('            float2 z=achita_marlin_out2(h?s.y:s.x,h?s.w:s.z,p.ga[e]*TWO62);',
        '#if FS_NOA\n            S.inter[2*t+h][half*8+g]=__float2bfloat16_rn(0.f);\n#else\n            float2 z=achita_marlin_out2(h?s.y:s.x,h?s.w:s.z,p.ga[e]*TWO62);')
    rep('S.inter[2*t+h][half*8+g]=__float2bfloat16_rn(2*t+h<cnt?__fdividef(z.x,1.f+__expf(-z.x))*z.y:0.f);',
        'float v=2*t+h<cnt?__fdividef(z.x,1.f+__expf(-z.x))*z.y:0.f;\n            S.inter[2*t+h][half*8+g]=__float2bfloat16_rn(v);\n#if FS_NOB\n            asm volatile("st.volatile.shared.f32 [%0],%1;"::"r"(su(&S.red[0][lane][h])),"f"(v):"memory");\n#endif\n#endif')
    rep('\n      int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);',
        '\n      if(task-S.a<64) {FS_STAMP(16+2*(task-S.a));}\n      int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);')
    rep('      const uint32_t b0=', '#if !FS_NOB\n#if FS_NOACC\n      float bcheck=0.f;\n#endif\n      const uint32_t b0=')
    rep('        if(t<2) {', '#if FS_NOACC\n        // One register checksum retains all B outputs; publish once per task.\n        bcheck+=d[0]+d[1]+d[2]+d[3];\n#else\n        if(t<2) {')
    rep('        if constexpr(HIGH) {\n          // Rebalance', '#endif\n        if constexpr(HIGH) {\n          // Rebalance')
    rep('      __syncwarp();\n      if(lane==0)arrive(S.empty+sl);',
        '#if FS_NOACC\n      asm volatile("st.volatile.shared.f32 [%0],%1;"::"r"(su(&S.red[warp][lane][0])),"f"(bcheck):"memory");\n#endif\n#endif // !FS_NOB\n      __syncwarp();\n      if(lane==0)arrive(S.empty+sl);\n      if(task-S.a<64) {FS_STAMP(17+2*(task-S.a));}')
    rep('    for(int i=tid*4;i<min(cnt,4)*HD;i+=NT*4) {',
        '    FS_STAMP(3);\n#if FS_TL\n    if(segment<8) {FS_STAMP(144+2*segment);}\n#endif\n    for(int i=tid*4;i<min(cnt,4)*HD;i+=NT*4) {')
    rep('      asm volatile("red.global.add.v4.f32 [%0],{%1,%2,%3,%4};"::"l"(dst),"f"(z.x),"f"(z.y),"f"(z.z),"f"(z.w):"memory");',
        '#if FS_NOFLUSH\n      asm volatile("st.volatile.shared.v4.f32 [%0],{%1,%2,%3,%4};"::"r"(su(S.red[warp][lane])),"f"(z.x),"f"(z.y),"f"(z.z),"f"(z.w):"memory");\n#elif !FS_NOB\n      asm volatile("red.global.add.v4.f32 [%0],{%1,%2,%3,%4};"::"l"(dst),"f"(z.x),"f"(z.y),"f"(z.z),"f"(z.w):"memory");\n#endif')
    rep('          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst),"f"(upper[z][0]):"memory");',
        '#if FS_NOFLUSH\n          asm volatile("st.volatile.shared.v2.f32 [%0],{%1,%2};"::"r"(su(S.red[warp][lane])),"f"(upper[z][0]),"f"(upper[z][1]):"memory");\n#elif !FS_NOB\n          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst),"f"(upper[z][0]):"memory");')
    rep('          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst+8),"f"(upper[z][1]):"memory");',
        '          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst+8),"f"(upper[z][1]):"memory");\n#endif')
    rep('    CSYNC();\n  }\n  asm volatile("griddepcontrol.launch_dependents;',
        '    CSYNC();\n    FS_STAMP(4);\n#if FS_TL\n    if(segment<8) {FS_STAMP(145+2*segment);}\n    ++segment;FS_META(10,segment);\n#endif\n  }\n  asm volatile("griddepcontrol.launch_dependents;')
    rep('  if((int)blockIdx.x<p.M) {','  FS_STAMP(5);\n  if((int)blockIdx.x<p.M) {')
    rep('\n}\n} // namespace fs', '\n  FS_STAMP(6);\n}\n} // namespace fs')
    rep('  static bool attr=false;static int smcount=0;',
        '#if FS_TL\n  TORCH_CHECK(scratch.size()>2 && scratch[2].numel()>=grid*9*160,"FS trace capacity");\n  p.trace=(unsigned long long*)scratch[2].data_ptr();\n#endif\n  static bool attr=false;static int smcount=0;')
    return src
