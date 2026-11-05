# Lossless router-stream arm, ported from loop/dec-pooler 0714254.
_DP_FRT, _DP_EXT, _DP_ERROR = _gle_frt, None, []


def _dp_source():
    source = _gle_source()
    old = '''          for (int r = 0; r < 8; ++r)
            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);'''
    new = '''          // [router stage j][128 CTAs][8 rows][512 FP32 columns].
          // One 16-KiB transaction, identical destination bytes and consumers.
          const size_t wp = ((size_t)((kb & 2047) >> 9) * 128 + cta) * 4096;
          bulk_g2s(s0, reinterpret_cast<const uint8_t*>(ep.w32 + wp), 16384, full0 + 8 * slot);'''
    if source.count(old) != 1:
        raise RuntimeError('pool source anchor not unique')
    return source.replace(old, new)


def _dp_build():
    global _DP_EXT
    try:
        source = _dp_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace('sk2_kernel', 'sk2_kernel_poolpack_' + tag)
                  .replace('launch_sk2', 'launch_sk2_poolpack_' + tag)
                  .replace('void sk2(', 'void sk2_poolpack_' + tag + '(')
                  .replace('  sk2(Ws,', '  sk2_poolpack_' + tag + '(Ws,')
                  .replace('void sk2f(', 'void sk2f_poolpack_' + tag + '(')
                  .replace('  sk2f(Ws,', '  sk2f_poolpack_' + tag + '(Ws,'))
        cpp = '''void gle_pr(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
 torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t);
void sk2_set_pdl(int64_t);'''
        _DP_EXT = _kvpre_load(load_inline, 'decode_pooler_pack_' + tag,
            '#include <torch/extension.h>\n' + cpp, cuda_sources=source,
            functions=['gle_pr', 'sk2_set_pdl'], no_implicit_headers=True,
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)
    except Exception as ex:
        _DP_ERROR.append(f'{type(ex).__name__}: {str(ex)[-500:]}')


def _dp_pack(weight):
    return weight.view(32, 8, 4, 4, 512).permute(3, 0, 2, 1, 4).contiguous().view(256, 8192)


def _dp_unpack(weight):
    return weight.view(4, 32, 4, 8, 512).permute(1, 3, 2, 0, 4).contiguous().view(256, 8192)


def _gle_frt(xb, x32, state):
    active = state.get('_dp_active', False)
    if not active:
        result = _DP_FRT(xb, x32, state)
    else:
        bst, m = state['bfp'], xb.shape[0]
        _, stream, tab, pmax, b4, _ = bst['sk2']['pooler']
        if 'fused_part' not in bst:
            bst['fused_part'] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=xb.device)
            bst['fused_cnt'] = torch.zeros(32, dtype=torch.int32, device=xb.device)
        pooled = torch.empty((m, 4096), device=xb.device, dtype=torch.bfloat16)
        logits = torch.empty((m, 256), device=xb.device, dtype=torch.float32)
        _bfp_pdl(m, 'pool')
        _DP_EXT.sk2_set_pdl(bfp_SK2_PDL)
        _DP_EXT.gle_pr(stream, bst['sk2']['router'][1], xb, pooled, logits, tab, pmax, b4,
                       bst['fused_part'], bst['fused_cnt'], m, x32, state['_dp_weight'], int(GLE_ILV))
        result = pooled, logits
    observer = getattr(_KP_TLS, 'observe', None)
    if observer is not None:
        observer['pool_used'] = active
    return result
