import ast,json,numpy as np,torch
from pathlib import Path
source=ast.parse(Path('candidates/active.py').read_text())
prefix='b2q_tt_v10_m32x_m32x_'
names={prefix+'c12_bytes',prefix+'c12_unpack',prefix+'table',prefix+'to_i32'}
names|={p+'_'+s for p in ['v18s','tt_v10_sm2_smallm2','tt_v12_smtrace1_smtrace','tt_v13_smtrace8_smtrace'] for s in ['c12_bytes','c12_unpack','table']}
selected=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
ns={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),'codec-only','exec'),ns)
def prmt(a,b,sel):
 out=np.zeros_like(a,dtype=np.uint32)
 for i in range(4):
  q=(sel>>(4*i))&7
  out|=(((a if q<4 else b)>>((q%4)*8))&255)<<(8*i)
 return out
def gpu_decoder_model(data,base):
 p=data.numpy().reshape(*data.shape[:2],3,32,16)
 low=np.concatenate((p[:,:,0],p[:,:,1]),axis=-1).copy().reshape(*data.shape[:2],32,4,8).view(np.uint32)
 c=p[:,:,2].copy().view(np.uint32)
 rotated=(c>>4)|(c<<28)
 t0=np.uint32(base*16843009)
 h0=(c&np.uint32(0x87878787))+t0;h1=(rotated&np.uint32(0x87878787))+t0
 regs=np.stack((prmt(low[...,0],h0,0x5140),prmt(low[...,0],h0,0x7362),prmt(low[...,1],h1,0x5140),prmt(low[...,1],h1,0x7362)),axis=-1)
 return torch.from_numpy(regs.view(np.uint16).astype(np.int32)).reshape(*data.shape[:2],32,32)
torch.manual_seed(20261028)
allbits=torch.arange(65536,dtype=torch.int32).reshape(1,64,32,32)
random=(torch.randn(1,128,32,32)*0.01).to(torch.bfloat16).view(torch.int16).to(torch.int32)&65535
reports=[]
for py_prefix in ('v18s','tt_v10_sm2_smallm2','tt_v12_smtrace1_smtrace','tt_v13_smtrace8_smtrace'):
 for v in (allbits,random):
  data,base,esc=ns[py_prefix+'_c12_bytes'](v)
  restored=ns[py_prefix+'_c12_unpack'](data,base,esc)
  assert torch.equal(restored,v)
  decoded=gpu_decoder_model(data,base)
  er,ec,el,ev,xor7=esc
  decoded[er,ec,el,ev]^=xor7.to(torch.int32)<<8
  assert torch.equal(decoded,v)
 reports.append({'family':py_prefix,'all_65536_patterns_exact':True,'random_values_exact':random.numel()})
print(json.dumps(reports))
