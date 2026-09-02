#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_rcr.py 의 기록부가 lam 을 0.9 로 하드코딩한 것을 고친다 (2026-09-02).

    python3 fix_lam_label.py

계산 결과에는 영향이 없다. result.json 의 `cell.lam_eff` / `cell.load` **라벨만** 고친다.
이미 나온 결과는 `collect_lam.py` 로 정정 취합하면 되고, 재실행할 필요는 없다.
"""
import sys, shutil
from pathlib import Path
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()/'battery-diag'/'run_rcr.py'
s = p.read_text(encoding='utf-8')
old = """        rec = dict(cell=dict(W=W, Mcyc=CAP, CAP=I.CAP, NARR=NARR, lam_eff=0.9*NARR,
                             load=0.9*NARR/max(I.CAP, 1), Li=Li),"""
new = """        rec = dict(cell=dict(W=W, Mcyc=CAP, CAP=I.CAP, NARR=NARR, lam=a.lam,
                             lam_eff=a.lam*NARR,
                             load=a.lam*NARR/max(I.CAP, 1), Li=Li),"""
if 'lam_eff=a.lam*NARR' in s:
    print('이미 적용되어 있다.'); sys.exit(0)
if old not in s:
    sys.exit('실패: 기록부를 찾지 못했다. run_rcr.py 를 직접 확인하라.')
shutil.copy(p, str(p)+'.pre-lamlabel.bak')
p.write_text(s.replace(old, new), encoding='utf-8')
print(f'적용 완료: {p}  (백업 .pre-lamlabel.bak)')
