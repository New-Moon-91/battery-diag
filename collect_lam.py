#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lam 스윕 결과를 **올바른 load 로** 취합한다.

    python3 collect_lam.py [결과디렉터리]        # 기본 results/rcr2

왜 필요한가 — 2026-09-02 발견한 버그
  `run_rcr.py:main` 의 기록부가 lam 을 0.9 로 **하드코딩**하고 있다.

      lam_eff = 0.9*NARR
      load    = 0.9*NARR/CAP

  `--lam` 은 모형(Config)에는 제대로 들어가므로 **계산 결과 자체는 옳다.**
  틀리는 것은 result.json 의 `cell.lam_eff` / `cell.load` 라벨뿐이고,
  `scripts/report_rcr.py` 가 그 라벨을 읽는다. 따라서 그대로 취합하면
  lam 이 다른 셀들이 전부 같은 load 로 찍힌다 — 리튬 혼입과 같은 종류의 조용한 오염이다.

  올바른 lam 은 result.json 의 `summary.lam_eff` 에 들어 있다(Instance 가 직접 계산).
  이 스크립트는 그쪽을 쓴다. **재실행이 필요 없다.**
"""
import json, csv, sys, re
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/rcr2')
rows, bad = [], []
for f in sorted(root.glob('*/result.json')):
    e = json.loads(f.read_text(encoding='utf-8'))
    c, sm = e['cell'], e.get('summary', {})
    jid = e.get('jid') or f.parent.name
    CAP = c.get('CAP') or c.get('Mcyc')

    # 진짜 lam_eff — summary 우선, 없으면 jid 의 _lam 태그, 그것도 없으면 0.9
    lam_eff = sm.get('lam_eff')
    src = 'summary'
    if lam_eff is None:
        m = re.search(r'_lam([0-9.]+)', jid)
        lam = float(m.group(1)) if m else 0.9
        lam_eff = lam * c['NARR']; src = 'jid' if m else 'default0.9'
    load = lam_eff / max(CAP, 1)

    if abs(lam_eff - 0.9*c['NARR']) > 1e-9:
        bad.append((jid, c['lam_eff'], lam_eff, c['load'], load))

    for pol, m in e['results'].items():
        rows.append(dict(jid=jid, W=c['W'], CAP=CAP, NARR=c['NARR'],
                         lam=round(lam_eff/c['NARR'], 4), lam_eff=round(lam_eff, 4),
                         load=round(load, 4), Li=c['Li'], policy=pol,
                         U_kwh=m.get('U_kwh'), L_total_kwh=m.get('L_total_kwh'),
                         forced_rate=m.get('forced_rate'), rho=m.get('rho_required'),
                         dE_kg=m.get('dE_kg'), g=m.get('g'), gap_pct=m.get('gap_pct'),
                         lam_src=src, sec=e.get('sec')))

if not rows:
    sys.exit(f'{root} 에 result.json 이 없다.')

out = root/'lam_sweep_corrected.csv'
with out.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'{out}  ({len(rows)}행)')

if bad:
    print(f'\n★ 라벨이 틀렸던 셀 {len(bad)}개 — 아래 값으로 정정했다')
    print(f"  {'jid':38s} {'기록 lam_eff':>12} {'실제':>8} {'기록 load':>10} {'실제':>8}")
    for j, a1, a2, l1, l2 in bad:
        print(f"  {j:38s} {a1:12.4f} {a2:8.4f} {l1:10.4f} {l2:8.4f}")
else:
    print('\n라벨 불일치 없음 (lam=0.9 만 돌렸거나 이미 정정된 결과).')

# 요약
d = {}
for r in rows: d.setdefault((r['load'], r['jid']), {})[r['policy']] = r
print(f"\n{'부하':>8} {'lam':>6} {'λ_eff':>7} {'CAP':>4} {'U(S1)':>9} {'U(S2)':>9} {'S2−S1':>9}")
prev = None
for (load, jid), g in sorted(d.items()):
    if 'S1' in g and 'S2' in g and g['S1']['U_kwh'] is not None:
        u1, u2 = g['S1']['U_kwh'], g['S2']['U_kwh']
        dd = u2 - u1
        flip = ''
        if prev is not None and (prev[1] < 0) != (dd < 0): flip = '   ← 여기서 부호 전환'
        print(f"{load:8.4f} {g['S1']['lam']:6.3f} {g['S1']['lam_eff']:7.2f} "
              f"{g['S1']['CAP']:4d} {u1:9.2f} {u2:9.2f} {dd:+9.2f}{flip}")
        prev = (load, dd)
print("\n※ 부호가 바뀌는 두 부하 사이가 load** 다.")
