"""Spend of this opencode session tree (root + all subagents), read-only from the local opencode DB."""
import sqlite3, os, collections, sys
root = sys.argv[1] if len(sys.argv) > 1 else 'ses_e5babb74effescXXNsHeNFMNK4'
c = sqlite3.connect('file:' + os.path.expanduser('~/.local/share/opencode/opencode.db') + '?mode=ro', uri=True)
rows = {r[0]: r for r in c.execute('select id,parent_id,agent,model,cost from session_v2')}
kids = collections.defaultdict(list)
for i, r in rows.items(): kids[r[1]].append(i)
tree, st = [], [root]
while st:
    s = st.pop(); tree.append(s); st += kids.get(s, [])
print(f'sessions {len(tree)} (subagents {len(tree)-1}) total ${sum((rows[s][4] or 0) for s in tree):.2f}')
