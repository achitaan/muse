"""Record source-backed parking decisions and refresh queue views, without GPU work."""
import json
from engine import ROOT, append_record, records


def main():
    audit=json.loads((ROOT/'source_audit.json').read_text())
    ledger=ROOT/'ledger'
    already=any(r['kind']=='source_audit' and r['payload']==audit for r in records(ledger).values())
    if not already:
        append_record(ledger,'source_audit',audit)
    candidates=json.loads((ROOT/'candidates.json').read_text())
    for decision in audit['decisions']:
        candidate=next(c for c in candidates if c['id']==decision['candidate'])
        candidate.update(research_status=decision['status'],decision_reason=decision['reason'],
                         next_experiment=decision['reopen_condition'],source_audit_id=audit['audit_id'])
        (ROOT/(candidate['id']+'.json')).write_text(json.dumps(candidate,indent=2)+'\n',encoding='utf8',newline='\n')
    (ROOT/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__': main()
