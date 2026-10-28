"""One four-tile WGS module; shared weights, independent scratch, no tuning."""
import time as _wgs32_time


def _wgs32_cuda_source(source):
    assert source.count('#define OKC_FT_T 64') == 1
    source = source.replace('#define OKC_FT_T 64', '#define OKC_FT_T 32', 1)
    for begin, end in (
        ('template <> struct WG<5> {', '// one 36 KB k-stage of a fat unit'),
        ('template <> struct WGSMMA<5> {', '// two producer warps:'),
    ):
        start = source.index(begin)
        stop = source.index(end, start)
        source = source[:start] + source[stop:]
    for function in ('fat_compute_wg', 'fat_compute'):
        start = source.index('          case 5: ' + function + '<5>')
        stop = source.index('\n        }', start)
        source = source[:start] + '          default: __trap(); break;' + source[stop:]
    start = source.index('      case 5: wgs_seg<PB, 5>')
    stop = source.index('\n    }', start)
    source = source[:start] + '      default: __trap(); break;' + source[stop:]
    return source


ALT_SRCS = dict(ALT_SRCS)
ALT_SRCS['wgs32'] = _wgs32_cuda_source(CUDA_SRC)
_wgs32_original_prepare = prepare


def prepare(weights):
    started = _wgs32_time.monotonic()
    state = _wgs32_original_prepare(weights)
    state['wgs32'] = None
    elapsed = _wgs32_time.monotonic() - started
    print('MOE_WGS32_PREPARE_BASE_S', elapsed, flush=True)
    if (state['sfx'] and '-DOKC_WGS=1' in tuple(_main_defines(state['sfx']))
            and torch.cuda.get_device_capability(weights['w13'].device) == (9, 0)
            and elapsed < 235):
        definitions = tuple(_main_defines(state['sfx'])) + state['sfx']
        incumbent = _ext(definitions)
        smaller = _ext(definitions + ('@wgs32',))
        state['wgs32'] = {'ext': smaller, 'scratch': _make_scratch(weights['w13'].device)}
        print('MOE_WGS32_RESOURCES', tuple(incumbent.okc_resources()),
              tuple(smaller.okc_resources()), flush=True)
    print('MOE_WGS32_ENABLED', state['wgs32'] is not None, flush=True)
    return state
