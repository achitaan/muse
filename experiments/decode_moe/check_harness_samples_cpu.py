"""Verify event units, paired order and optional raw observation retention without CUDA."""
from types import SimpleNamespace
from harness import HARNESS_SRC


def check(record):
    milliseconds = {'parent': [.001, .003, .002], 'candidate': [.0008, .0028, .0018]}
    counts = {'parent': 0, 'candidate': 0}
    order = []
    state = SimpleNamespace(capturing=None, stamp=None, flushes=0)

    class Graph:
        def replay(self):
            index = counts[self.label]
            counts[self.label] += 1
            state.stamp = (self.label, index)
            order.append(self.label)

    class Context:
        def __init__(self, graph): self.graph = graph
        def __enter__(self): state.capturing = self.graph
        def __exit__(self, *args): state.capturing = None

    class Event:
        def __init__(self, **kwargs): pass
        def record(self): self.stamp = state.stamp
        def elapsed_time(self, end):
            label, index = end.stamp
            return milliseconds[label][index]

    class Flush:
        def zero_(self): state.flushes += 1

    def fn(label):
        if state.capturing is not None: state.capturing.label = label

    cuda = SimpleNamespace(synchronize=lambda: None, CUDAGraph=Graph, graph=Context, Event=Event)
    torch = SimpleNamespace(cuda=cuda, empty=lambda *args, **kwargs: Flush(), uint8='uint8')
    scope = {'torch': torch}
    exec(HARNESS_SRC, scope)
    samples = {'existing': [99.]}
    result = scope['_okh_time']({'parent': lambda: fn('parent'), 'candidate': lambda: fn('candidate')},
                                'CPU-mock', reps=3, **({'samples': samples} if record else {}))
    assert result == {'parent': 2., 'candidate': 1.8}
    assert order == ['parent', 'candidate'] * 3 and state.flushes == 6
    if record:
        assert samples == {'existing': [99.], 'parent': [1., 3., 2.], 'candidate': [.8, 2.8, 1.8]}
    else:
        assert samples == {'existing': [99.]}


if __name__ == '__main__':
    check(False)
    check(True)
    print('CPU timer-contract checks passed:ms to us,paired order,cold flush count,raw samples,legacy median result')
