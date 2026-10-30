"""Download public upstream data for inspection; never import or execute it."""

import ast
import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/public-source-refresh'


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Muse-source-review','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=25)as response:return response.read()


def main():
    repository='vjeux/okc-2026'
    metadata=json.loads(get('https://api.github.com/repos/'+repository+'/commits/main'))
    sha=metadata['sha'];assert re.fullmatch('[0-9a-f]{40}',sha)
    folder=OUT/sha;folder.mkdir(parents=True,exist_ok=True)
    records={}
    for name in ('muse_moe_w4a16.py','LICENSE'):
        url=f'https://raw.githubusercontent.com/{repository}/{sha}/{name}'
        raw=get(url);(folder/name).write_bytes(raw)
        records[name]=dict(url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    upstream=(folder/'muse_moe_w4a16.py').read_text(encoding='utf-8')
    local=(ROOT/'best/moe_verified.py').read_text(encoding='utf-8')
    def literal(source,name):
        return next(ast.literal_eval(n.value)for n in ast.parse(source).body
                    if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')==name)
    uc,lc=literal(upstream,'CUDA_SRC'),literal(local,'CUDA_SRC')
    (folder/'upstream-native.cu').write_bytes(uc.encode())
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository=repository,
        commit=sha,commit_date=metadata['commit']['committer']['date'],commit_url=metadata['html_url'],
        previous_upstream_pin='f35a41ee1c13e316dabbf772830d47b23b453393',files=records,
        upstream_CUDA_sha256=hashlib.sha256(uc.encode()).hexdigest(),retained_CUDA_sha256=hashlib.sha256(lc.encode()).hexdigest(),
        native_text_equal=uc==lc,imported_or_executed=False,GPU_jobs=0,retained_sources_changed=False)
    (OUT/'latest.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
