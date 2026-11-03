"""Equivalent KPERM3 scale-lifetime spelling; all four pairs once per stage."""
def predecode(src):
    a='    const uint2 stage_sc = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);'
    assert src.count(a)==1
    src=src.replace(a,a+'''
    const uint32_t scale0 = __byte_perm(stage_sc.x, 0u, 0x4140u) * k16c + 0x5C005C00u;
    const uint32_t scale1 = __byte_perm(stage_sc.x, 0u, 0x4342u) * k16c + 0x5C005C00u;
    const uint32_t scale2 = __byte_perm(stage_sc.y, 0u, 0x4140u) * k16c + 0x5C005C00u;
    const uint32_t scale3 = __byte_perm(stage_sc.y, 0u, 0x4342u) * k16c + 0x5C005C00u;''')
    a='''#if OKC_KPERM == 3
      const uint2 scg = stage_sc;
#else
      const uint2 scg = scn;
#endif'''
    assert src.count(a)==1
    src=src.replace(a,'''#if OKC_KPERM == 0
      const uint2 scg = scn;
#endif''')
    a='''          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;'''
    assert src.count(a)==1
    src=src.replace(a,'''#if OKC_KPERM == 3
          const uint32_t sgp = js == 0 ? scale0 : js == 1 ? scale1 : js == 2 ? scale2 : scale3;
#else
'''+a+'''
#endif''')
    return src
