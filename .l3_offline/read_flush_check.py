"""CPU-only protocol check of the user's moediag13 READ-flush pattern."""
import ast,contextlib,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT.parent/'decode-next/experiments/boards/diag/moediag13.py'
t=path.read_text();n=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='bench')
source=ast.get_source_segment(t,n)
assert 'torch.amax(flush)' in source
assert 'flush.zero_' not in source and 'flush.add_' not in source
events=[]
class Event:
    def __init__(self,**kw):pass
    def record(self):events.append('event')
    def synchronize(self):events.append('event_sync')
    def elapsed_time(self,other):return .01
class Graph:
    def replay(self):events.append('replay')
class CUDA:
    Event=Event;CUDAGraph=Graph
    @staticmethod
    def synchronize():events.append('sync')
    @staticmethod
    @contextlib.contextmanager
    def graph(g):events.append('capture');yield;events.append('captured')
def amax(flush):events.append('read_flush');return object()
ns=dict(torch=types.SimpleNamespace(cuda=CUDA,amax=amax),_FS=[],statistics=__import__('statistics'),error=lambda *a:None)
exec(compile(source,'<moediag13 bench>','exec'),ns)
assert ns['bench'](lambda:events.append('call'),object(),n=40)==(10.,10.)
assert events[events.index('read_flush'):]==['read_flush','event','replay','event','event_sync']*40
assert len(ns['_FS'])==40
print('PASS40 READ flushes before start events, reductions retained; no GPU timing or cache claim')
