#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5 — 유형표(차종×용량) 산출.  data/types_w5.json 을 만든다.

    python scripts/build_types_w5.py

원자료 두 갈래를 합친다.

  구자료 `pool.csv`/`pass_soh.csv` (742건, 2022.3~2025.3)
      → n, q_P, cap, mu_S, sd_S.  **통과확률은 여기서만 나온다** — 신규 입찰자료는
        낙찰분만 담고 있어 유찰(=재사용 실패) 관측이 없기 때문이다.

  신규 입찰자료 `입찰_재사용__0819.xlsx`/`입찰_재활용__0819.xlsx` (273/172건)
      → 유형별 팩당 낙찰가 중앙값 → **실측 손익배수**.

실측 손익배수 (정합성 메모 §4.2):

    r_emp = q_P · (E[재사용 낙찰가] − V^S) / C_p

w4 까지는 분모를 C_p/q̄_P (전체 평균 0.5135) 로 고정해 유형별 통과확률 차이를
반영하지 못했다. 정밀검사는 통과확률 q_P 로만 재사용 매출을 얻으므로 q_P 는
분자에 유형별로 들어가야 한다.

E·V^S 는 **모형을 전혀 쓰지 않은** 실거래 중앙값(전기간)이다. 따라서 최적정책의
검사강도 서열과 대조할 때 독립 증거가 된다.

원자료가 없으면(gitignore 대상) 이 스크립트는 실행할 수 없다. 산출물
`data/types_w5.json` 은 추적되므로 원자료 없이도 모형은 돌아간다.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import (normalize_cap, fleet_from_pool, type_key,
                               reg_from_ratio, POOL_MIN, SIDE_MIN, CP)

REUSE_XLSX = 'data/입찰_재사용__0819.xlsx'
RECYC_XLSX = 'data/입찰_재활용__0819.xlsx'


def load_bids(root=ROOT):
    """신규 입찰자료 → (재사용, 재활용) — 유형키·팩당가·kWh당가를 붙여 돌려준다."""
    R = pd.read_excel(root/REUSE_XLSX)
    C = pd.read_excel(root/RECYC_XLSX)
    for d, lot in ((R, '묶음개수'), (C, '수량')):
        d['model'] = d['차종'].astype(str).str.strip()
        d['kwh'] = d['용량'].astype(float)
        d['key'] = [type_key(m, k) for m, k in zip(d.model, d.kwh)]
        d['date'] = pd.to_datetime(d['공고시작일시'])
        d['lot'] = d[lot]
        d['pack'] = d['낙찰금액'] / d['lot']          # 팩 1개당 낙찰가
        d['unit'] = d['pack'] / d['kwh']              # kWh당 낙찰가
    R['s'] = R['잔존가치'] / 100.0
    return R, C


def main():
    pool = pd.read_csv(ROOT/'data'/'pool.csv')
    ps = pd.read_csv(ROOT/'data'/'pass_soh.csv')
    pool, merges = normalize_cap(pool)
    for mg in merges:
        ps.loc[(ps.model == mg['model']) & (ps.kwh == mg['src']), 'kwh'] = mg['dst']
    FLEET = fleet_from_pool(pool, ps, by_capacity=True)

    R, C = load_bids()
    a = R.groupby('key').agg(n_re=('pack', 'size'), med_re=('pack', 'median'))
    b = C.groupby('key').agg(n_rc=('pack', 'size'), med_rc=('pack', 'median'))

    rows = []
    for t, (n, qP, cap, mu, sd) in FLEET.items():
        n_re = int(a.n_re.get(t, 0)); n_rc = int(b.n_rc.get(t, 0))
        med_re = float(a.med_re.get(t, np.nan)); med_rc = float(b.med_rc.get(t, np.nan))
        ok = (n >= POOL_MIN and n_re >= SIDE_MIN and n_rc >= SIDE_MIN)
        r = qP*(med_re-med_rc)/CP if (ok and np.isfinite(med_re) and np.isfinite(med_rc)) else None
        rows.append(dict(type=t, n=n, qP=qP, cap=cap, mu=mu, sd=sd,
                         n_re=n_re, n_rc=n_rc,
                         med_re=med_re if np.isfinite(med_re) else None,
                         med_rc=med_rc if np.isfinite(med_rc) else None,
                         ratio_emp=r, reg_emp=reg_from_ratio(r) if r is not None else None,
                         eligible=bool(ok and r is not None)))
    rows.sort(key=lambda x: (x['ratio_emp'] is None, x['ratio_emp'] or 0))

    out = dict(
        _source=dict(pool='data/pool.csv (742건, 2022.3~2025.3)',
                     bids=f'{REUSE_XLSX} (n={len(R)}) / {RECYC_XLSX} (n={len(C)})',
                     window='전기간 2022-01~2025-10'),
        _rule=dict(ratio_emp='q_P*(median 재사용 팩당 낙찰가 - median 재활용 팩당 낙찰가)/C_p',
                   Cp=CP, POOL_MIN=POOL_MIN, SIDE_MIN=SIDE_MIN,
                   reg='R1 <0.8, R3 0.8~1.5, R2 >=1.5'),
        _merges=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in m.items()} for m in merges],
        types=rows)
    f = ROOT/'data'/'types_w5.json'
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f"{'유형':<12}{'pool':>5}{'q_P':>6}{'용량':>6}{'μ':>7}{'σ':>6}"
          f"{'재/활':>8}{'실측배수':>9}{'등급':>5}")
    for x in rows:
        if not x['eligible']: continue
        print(f"{x['type']:<12}{x['n']:5d}{x['qP']:6.2f}{x['cap']:6.1f}{x['mu']:7.3f}"
              f"{x['sd']:6.3f}{str(x['n_re'])+'/'+str(x['n_rc']):>8}"
              f"{x['ratio_emp']:9.2f}{x['reg_emp']:>5}")
    print(f'\n적격 {sum(x["eligible"] for x in rows)}유형 / 전체 {len(rows)}   → {f}')
    print('병합:', merges)


if __name__ == '__main__':
    main()
