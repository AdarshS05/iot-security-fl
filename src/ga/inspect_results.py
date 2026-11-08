import json
import sys

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

print('results length:', len(d.get('results', [])))
if d.get('results'):
    r0 = d['results'][0]
    print('r0 keys:', list(r0.keys()))
    print('r0 full:', json.dumps(r0, indent=2))
