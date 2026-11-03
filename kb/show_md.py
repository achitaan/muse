import json, sys
j = json.load(open(sys.argv[1], encoding='utf-8-sig'))
lt = j['result'].get('log_tail') or ''
line = [l for l in lt.split('\n') if l.startswith('MD SUMMARY LAST')][-1]
d = json.loads(line[line.index('results=') + 8:])
for x in d:
    print(json.dumps(x)[:2500])
    print()
