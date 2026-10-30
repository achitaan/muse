"""Positive norm-addend/FTZ oracle; CPU only,not CUDA validation."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parents[2]/'results/decode-moe-plan-20261030/norm-group'

def sum32(a):
    return np.cumsum(a,dtype=np.float32)[-1]if len(a)else np.float32(0)

def grouped(a,flush):
    jobs=len(a);v=a.copy()
    if flush:v[(v.view(np.uint32)&0x7f800000)==0]=0
    contributions=[[]for _ in range(jobs)]
    for c in range(132):
        lo,hi=jobs*256*c//132,jobs*256*(c+1)//132
        for j in range(jobs):
            x,y=max(lo,j*256)-j*256,min(hi,(j+1)*256)-j*256
            if x<y:
                part=sum32(v[j,x:y])
                # Original global atomic also flushes its input at publication.
                if (part.view(np.uint32)&0x7f800000)==0:part=np.float32(0)
                contributions[j].append(part)
    return np.array([sum32(np.asarray(x[::-1],dtype=np.float32))for x in contributions])

def main():
    maximum=0.;cases=0
    for jobs in (1,2,4,8,16,32):
        for seed in (17,37,71):
            rng=np.random.default_rng(seed)
            for pattern in ('normal','wide','subnormal','zero'):
                if pattern=='normal':a=(rng.random((jobs,256))*4).astype(np.float32)
                elif pattern=='wide':a=np.exp2(rng.integers(-140,21,(jobs,256))).astype(np.float32)
                elif pattern=='subnormal':a=np.full((jobs,256),np.float32(2.**-130))
                else:a=np.zeros((jobs,256),dtype=np.float32)
                ftz=a.copy();ftz[(ftz.view(np.uint32)&0x7f800000)==0]=0
                old=np.cumsum(ftz,axis=1,dtype=np.float32)[:,-1];new=grouped(a,True)
                # Oracle is the mathematical sum of original FTZ addends.
                exact=ftz.astype(np.float64).sum(axis=1)
                denom=max(float(np.linalg.norm(exact)),1e-30)
                error=float(np.linalg.norm(new.astype(np.float64)-old.astype(np.float64))/denom)
                assert error<=1e-5
                maximum=max(maximum,error);cases+=1
                if pattern in ('zero','subnormal'):assert np.array_equal(new,np.zeros_like(new))
    tiny=np.full((32,256),np.float32(2.**-130))
    assert np.all(grouped(tiny,True)==0)and np.any(grouped(tiny,False)>0)
    record=dict(CUDA_sha256=hashlib.sha256((OUT/'candidate.cu').read_bytes()).hexdigest(),cases=cases,
      maximum_relative_change=maximum,missing_FTZ_mutant_detected=True,GPU_calls=0,
      scope='Norm addends only;positive FP32 grouping with original FTZ input semantics. No kernel ownership,race,GEMM or output validation.')
    (OUT/'numerical-cpu-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))

if __name__=='__main__':main()
