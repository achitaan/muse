"""Unique-byte/compute roofline for the supplied 646c489f50b34d66 measurements.

RESTART event times include CUDA graph overhead. The task says approximately
4 us are subtracted by the judge; report both denominators, not just one.
These are effective bandwidths, NOT hardware-counter measurements of HBM.
"""
SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024),
          "pooler": (4096, 8192), "unpooler": (8192, 4096)}
EVENT_US = {128: (15.936, 10.704, 20.880, 21.248),
            256: (21.296, 12.528, 28.288, 28.096)}


def rows():
    for m, times in EVENT_US.items():
        for (layer, (n, k)), event_us in zip(SHAPES.items(), times):
            flops = 2 * m * n * k
            io_bytes = 2 * m * (n + k)
            c12 = 1.5 * n * k + io_bytes
            raw = 2 * n * k + io_bytes
            compressed = m == 128 and layer != "o"
            actual = c12 if compressed else raw
            work_us = event_us - 4.0
            yield dict(m=m, layer=layer, route="c12-rq" if compressed else "raw-BF16",
                       event_us=event_us, work_us=work_us,
                       event_tflops=flops / event_us / 1e6, tflops=flops / work_us / 1e6,
                       event_tbps=actual / event_us / 1e6, tbps=actual / work_us / 1e6,
                       raw_floor=max(raw / 4.8e6, flops / 989e6),
                       c12_floor=max(c12 / 4.8e6, flops / 989e6),
                       regime="bandwidth-favored" if actual / 4.8e6 >= flops / 989e6 else "compute-favored")


if __name__ == "__main__":
    print("| M | layer | route | event us | event TF/s | effective TB/s | ~work us | ~work TF/s | ~work TB/s | raw/c12 floors us | roofline regime |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows():
        print("| {m} | {layer} | {route} | {event_us:.3f} | {event_tflops:.1f} | {event_tbps:.3f} | "
              "{work_us:.3f} | {tflops:.1f} | {tbps:.3f} | {raw_floor:.2f}/{c12_floor:.2f} | {regime} |".format(**r))
