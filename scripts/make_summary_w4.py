#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4 [5] — results/w4/*.json → summary_w4.csv

    python scripts/make_summary_w4.py

한 줄이 W 하나다. 정확해·벤치마크 갭·분해·강제매각·정책구조·DCL 을 한 표에 모은다.
"""
import csv, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'w4'

FIELDS = ['W', 'nS', 'n_sa', 'nnz', 'csr_gb', 'stream', 'build_sec', 'solve_sec',
          'pi_iters', 'gstar', 'dgstar', 'B1', 'B2', 'INDEX', 'B_fast', 'B_fast_hold',
          'gap_B1', 'gap_B2', 'gap_INDEX', 'gap_B_fast', 'gap_B_fast_hold',
          'cost_full_inspection', 'cost_forced_disposal',
          'B_fast_thr', 'B_fast_hold_thr',
          'forced_units', 'forced_value', 'forced_prob',
          'forced_units_bfast', 'forced_value_bfast',
          'dcl_gap_mean', 'dcl_gap_sd', 'dcl_seeds', 'wall_sec']


def main():
    rows = []
    prev = None
    for f in sorted(OUT.glob('W*.json'), key=lambda p: int(p.stem[1:])):
        d = json.loads(f.read_text())
        g = d['gstar']
        r = dict(W=d['W'], nS=d['nS'], n_sa=d['n_sa'], nnz=d['nnz'],
                 csr_gb=round(d['csr_gb'], 3), stream=d['stream'],
                 build_sec=round(d['build_sec'], 1), solve_sec=round(d['solve_sec'], 1),
                 pi_iters=d['pi_iters'], gstar=round(g, 2),
                 dgstar=('' if prev is None else round(g-prev, 2)),
                 B_fast_thr=d['B_fast_thr'], B_fast_hold_thr=d.get('B_fast_hold_thr', ''),
                 wall_sec=round(d['wall_sec'], 1))
        prev = g
        for k in ('B1', 'B2', 'INDEX', 'B_fast', 'B_fast_hold'):
            r[k] = round(d['bench'][k], 2)
            r['gap_'+k] = round(d['gaps'][k], 4)
        for k in ('cost_full_inspection', 'cost_forced_disposal'):
            r[k] = round(d['decomp'][k], 4)
        for k in ('forced_units', 'forced_value', 'forced_prob'):
            r[k] = round(d['opt'][k], 6)
        r['forced_units_bfast'] = round(d['bfast']['forced_units'], 6)
        r['forced_value_bfast'] = round(d['bfast']['forced_value'], 2)
        gaps = [json.loads(p.read_text())['gap']
                for p in sorted(OUT.glob(f"dcl_W{d['W']}_carry_seed*.json"))]
        r['dcl_seeds'] = len(gaps)
        r['dcl_gap_mean'] = f'{st.mean(gaps):.3e}' if gaps else ''
        r['dcl_gap_sd'] = f'{st.stdev(gaps):.3e}' if len(gaps) > 1 else ''
        rows.append(r)

    with open(OUT/'summary_w4.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f'{OUT/"summary_w4.csv"}  ({len(rows)}행)')

    # 차종별 행동 비율표는 W 마다 따로 낸다 (긴 표라 CSV 를 나눈다).
    pr = []
    for f in sorted(OUT.glob('W*.json'), key=lambda p: int(p.stem[1:])):
        d = json.loads(f.read_text())
        for k, t in enumerate(d['opt']['types']):
            sell, fast, pu, hold = d['opt']['act_u']
            s_sell, s_ps, s_hold = d['opt']['act_s'][k]
            sell, fast, pu, hold = d['opt']['act_u'][k]
            handled = sell+fast+pu+hold
            pr.append(dict(W=d['W'], model=t,
                           reg_emp=d['summary']['REG_EMP'].get(t, ''),
                           ratio_emp=d['summary']['RATIO_EMP'].get(t, ''),
                           sell=round(sell, 6), fast=round(fast, 6), pu=round(pu, 6),
                           hold=round(hold, 6), scr_sell=round(s_sell, 6),
                           scr_ps=round(s_ps, 6), scr_hold=round(s_hold, 6),
                           handled=round(handled, 6),
                           fast_rate=round(fast/handled, 6) if handled else '',
                           prec_rate=round((pu+s_ps)/handled, 6) if handled else ''))
    with open(OUT/'policy_structure_w4.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pr[0])); w.writeheader(); w.writerows(pr)
    print(f'{OUT/"policy_structure_w4.csv"}  ({len(pr)}행)')


if __name__ == '__main__':
    main()
