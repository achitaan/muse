from pathlib import Path
import ast, base64, zlib, hashlib
from build_source import ROOT
parent=ROOT/'experiments/boards/moe/moe_b13.py'
candidate=ROOT/'experiments/boards/moe/moe_fs1.py'
raw=candidate.read_bytes()
md=(ROOT/'experiments/boards/diag/moediag4.py').read_text()
nodes={n.name:n for n in ast.parse(md).body if isinstance(n,ast.FunctionDef)}
helpers='\n\n'.join(ast.get_source_segment(md,nodes[k]) for k in ('generate_weights','generate_input','bench'))
text='''"""DIAGNOSTIC ONLY: pinned moe_b13 TINY/TINY2 vs FS1 hybrid.
Judge-like R1/R8, synthetic M2/M4 and ZERO8;40 cold-L2 graph replays,
both case/variant orders. Uses random local weights (same distribution, not
the full256-expert generator offsets). Separate three-TU thin-host bundle.
No local GPU results are implied by offline compilation or CPU tests.
"""
import base64,hashlib,math,os,queue,statistics,threading,time,types,zlib
import torch
'''
text+='BASE_SHA='+repr(hashlib.sha256(parent.read_bytes()).hexdigest())+'\n'
text+='CANDIDATE_SHA='+repr(hashlib.sha256(raw).hexdigest())+'\n'
text+='CANDIDATE_B85='+repr(base64.b85encode(zlib.compress(raw,9)).decode())+'\n'
text+=helpers+'\n\n'+(ROOT/'.fs_offline/fsdiag_body.py').read_text()
ast.parse(text)
out=ROOT/'experiments/boards/diag/fsdiag.py';out.write_bytes(text.encode())
print(out,len(text.encode()))
