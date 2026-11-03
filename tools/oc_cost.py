import sqlite3, json, sys, collections
db = r'C:\Users\Achita\.local\share\opencode\opencode.db'
c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
root = 'ses_e76aeed06ffeOusqHDl9FlFtnJ'
par = {sid: p for sid, p in c.execute('select id, parent_id from session_v2')}
def in_tree(s):
    seen = 0
    while s and seen < 50:
        if s == root: return True
        s = par.get(s); seen += 1
    return False
# inspect one message structure
row = c.execute("select data from session_message where session_id=? and type like '%assist%' limit 1", (root,)).fetchone()
types = [t for (t,) in c.execute('select distinct type from session_message limit 20')]
print('types', types)
if row: print(str(row[0])[:600])
tot = collections.defaultdict(float); n = collections.Counter(); since = None
day = collections.defaultdict(float)
import datetime
for sid, t, data, tc in c.execute('select session_id, type, data, time_created from session_message'):
    if not in_tree(sid): continue
    try: d = json.loads(data)
    except Exception: continue
    cost = d.get('cost') if isinstance(d, dict) else None
    if cost is None and isinstance(d, dict):
        cost = (d.get('info') or {}).get('cost') if isinstance(d.get('info'), dict) else None
    if cost:
        m = d.get('modelID') or (d.get('model') or {}).get('modelID') if isinstance(d.get('model'), dict) else d.get('modelID')
        tot[str(m)] += float(cost); n[str(m)] += 1
        day[datetime.datetime.utcfromtimestamp(tc/1000).strftime('%m-%d')] += float(cost)
print('total', round(sum(tot.values()), 2))
for k, v in sorted(tot.items(), key=lambda x: -x[1]): print(k, round(v, 2), n[k])
print({k: round(v, 2) for k, v in sorted(day.items())})
