"""Adapt the already checked moediag7 driver oracle to L2 and larger fixups."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
t=(ROOT/'.l1_offline/test_diag.py').read_text()
t=t.replace("ROOT/'.l1_offline/diag'", "ROOT/'.l2_offline/diag'")
t=t.replace('moediag_l1.py','moediag_l2.py').replace('moe_l1.py','moe_l2.py').replace("'L1'","'L2'")
t=t.replace('L1_BASE','L2_BASE').replace('moe_b13/L1','moe_b13/L2')
t=t.replace("assert 'setmaxnreg' not in bodies[0] and 'setmaxnreg' in bodies[1]", "assert all('setmaxnreg' not in s for s in bodies)\n    assert 'm64n96k16' in bodies[1] and 'm64n96k16' not in bodies[0]")
t=t.replace("(md,'over',lambda:False)", "(md,'over',lambda:False),(md,'_l2_narrow_rows',lambda *args:torch.ones((1,4096)))")
t=t.replace("# Preserve inherited40-repetition protocol", """# Independent check that per-expert row selection excludes65+ token experts.
c=[1,65,2]+[0]*29
buf=torch.arange(3072+sum(c)*4096,dtype=torch.float32)
ys=buf[3072:].view(-1,4096)
assert torch.equal(md._l2_narrow_rows({'scratch':[None,buf]},c,3072),torch.cat((ys[:1],ys[66:68])))
print('PASS per-route narrow-output selection excludes native-wide expert rows')
# Preserve inherited40-repetition protocol""")
# Validate the larger diagnostic scratch contract before actual GPU allocation.
t=t.replace("print('PASS6 requested", "assert fake['WGS_FIXW']==6144 and fake['WGS_FIXF']==2*160*4*6144+2*160*4+136*16\nprint('PASS6 requested")
exec(compile(t,str(ROOT/'.l2_offline/test_diag.py'),'exec'))
