"""Untimed diagnostic views of retained WGS live SwiGLU data.

Call only outside the timed path, after synchronization, with a fresh host
copy of the probe's routes. This module never modifies a kernel or scratch.
"""

import torch


def jobs_from_routes(ids,lo=0,nloc=32):
    assert 0<nloc<=32 and len(ids)<=384
    groups=[[]for _ in range(nloc)]
    for token,row in enumerate(ids):
        assert len(row)==8
        for slot,e in enumerate(row):
            if lo<=e<lo+nloc:groups[e-lo].append(token*8+slot)
    starts=[];offset=0
    for entries in groups:starts.append(offset);offset+=len(entries)
    fat=[];skinny=[]
    for expert,entries in enumerate(groups):
        count=len(entries)
        if count>8:
            fat.extend((expert,starts[expert]+r,min(64,count-r))for r in range(0,count,64))
        elif count:skinny.append((expert,starts[expert],count))
    jobs=[];padded=0
    for expert,start,count in fat+skinny:
        jobs.append(dict(expert=expert,start=start,count=count,padded_start=padded))
        padded+=(count+7)//8*8
    assert len(jobs)<=88 and padded<=3776 and offset<=3072
    return jobs,offset,padded


def canonical_xb(packed,jobs,n,hidden=4096):
    """Extract only valid rows in expert-major route order, ignoring padding."""
    assert hidden%256==0
    out=torch.empty((n,hidden),dtype=packed.dtype,device=packed.device)
    col=torch.arange(hidden,dtype=torch.int64,device=packed.device)
    for job in jobs:
        count,start=job['count'],job['start'];ntt=(count+7)//8
        row=torch.arange(count,dtype=torch.int64,device=packed.device)[:,None]
        # Retained wgs_epi_tile: stage,token tile,column quad,token,column.
        offsets=(job['padded_start']*hidden+((col//256)*ntt+row//8)*2048
                 +((col%256)//8)*64+(row%8)*8+col%8)
        assert int(offsets.max())<packed.numel()
        out[start:start+count]=packed[offsets]
    return out


def snapshot_wgs_inter(state,host_ids,lo=0,nloc=32):
    """Single completed launch only; original parent allocation constants."""
    jobs,n,padded=jobs_from_routes(host_ids,lo,nloc)
    hidden,emax,xrows=4096,3072,3776
    begin=(emax+xrows)*hidden
    packed=state['scratch'][2][begin:begin+padded*hidden]
    return canonical_xb(packed,jobs,n,hidden)
