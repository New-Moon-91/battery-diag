#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""결과 디렉터리 → 요약 CSV/표. 예: python scripts/collect.py /home/data/batdiag/results/sweep_load"""
import json, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1]); rows = []
for d in sorted(root.glob('*/result.json')):
    r = json.loads(d.read_text())
    row = dict(job=d.parent.name, **r['cfg'], **r['summary'], gstar=r['gstar'])
    row.update({f'gap_{k}': v for k, v in r['gaps'].items()})
    if 'dcl' in r: row['gap_DCL'] = r['dcl']['gap']
    row.pop('REG', None); rows.append(row)
df = pd.DataFrame(rows)
out = root/'summary.csv'; df.to_csv(out, index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False)); print('\n→', out)
