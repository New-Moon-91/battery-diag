#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RCR 편 주 실험 — 규제 시나리오 x 진단용량 x 리튬 스윕.

    python run_rcr.py --out results/rcr --shard 0 --nshard 2

시나리오
  S0    무진단 — 전량 즉시 재활용
  S1    전수 성능평가 (2027 규제). 미검사 매각 금지. 용량 CAP 제약
  S1inf **용량 무제한 전수평가** — 규제가 의도대로 작동했을 때의 상한.
        S1inf 와 S1 의 격차가 곧 **진단 병목의 비용**이다. (Mcyc = W 로 구현)
  S2b   사전정보 선별 (신속검사 미사용)
  Bfast 전량 신속검사 강제 (원고 구조), 임계 0..SB-1 중 최선
  INDEX 근시안 EVSI 지표
  S2    정확해 최적정책 — torch 있을 때만

주의
  · prune=False, pu_all=True 가 필수다. 아니면 S1 이 표현되지 않는다
  · 가격은 2층(PriceRCR). 재활용 용량지수 1.2675 — 선형 근사 금지
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from battery_diag.data import fleet_w5, SEL_W5
from battery_diag.instance import Instance, Config
from battery_diag.price_rcr import PriceRCR
from battery_diag.build import build
from battery_diag import policies as pol, policies_rcr as prc, metrics_w as M

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--data', default='data')
ap.add_argument('--params', default='data/params_rcr_2layer.json')
ap.add_argument('--types', default='', help='쉼표 구분. 비우면 SEL_W5 4유형')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--workers', type=int, default=min(os.cpu_count() or 4, 24))
ap.add_argument('--cache', default=None)
ap.add_argument('--no-exact', action='store_true')
# 스윕 축
ap.add_argument('--W',    default='4,5,6')
ap.add_argument('--cap',  default='1,2,3')
ap.add_argument('--narr', default='1,2,3,4')      # lam_eff = 0.9 x NARR
ap.add_argument('--li',   default='35.8357')      # 리튬 스윕: 8.16,20,35.84,50,70.42
ap.add_argument('--no-s1inf', action='store_true')
# 민감도 축 — 재해결이 필요한 파라미터 (기본값은 Config 와 동일)
ap.add_argument('--phi', type=float, default=1.0,   help='미공개 결함 중 신속검사로 못 잡는 비율')
ap.add_argument('--Cf',  type=float, default=20_000.,  help='신속검사 변동비')
ap.add_argument('--Cp',  type=float, default=500_000., help='정밀검사 변동비')
ap.add_argument('--lam', type=float, default=0.9,
                help='도착 기회당 도착 확률. load = lam x NARR / CAP')
# 탄소·환산
ap.add_argument('--eps-new', type=float, default=74.0)
ap.add_argument('--eps-repurp', type=float, default=7.73)
ap.add_argument('--eps-diag', type=float, default=0.319)
ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--Lambda', type=float, default=107500.)
ap.add_argument('--ncenter', type=int, default=60)
ap.add_argument('--Dyear', type=int, default=245)
a = ap.parse_args()

F = lambda s, cast=float: [cast(x) for x in s.split(',') if x]
GRID = [g for i, g in enumerate(itertools.product(F(a.W,int), F(a.cap,int), F(a.narr,int), F(a.li)))
        if i % a.nshard == a.shard]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshard}: {len(GRID)}개 셀', flush=True)

SEL = a.types.split(',') if a.types else SEL_W5
FL = fleet_w5(a.data, SEL); tot = sum(v[0] for v in FL.values())
TYPES = {k: (v[1], v[2], v[3], v[4], v[0]/tot) for k, v in FL.items()}
EK = dict(eps_new=a.eps_new, eps_repurp=a.eps_repurp, eps_diag=a.eps_diag, beta=a.beta)

try:
    import torch
    from battery_diag.exact import ExactSolver
    HAVE = not a.no_exact
except ImportError:
    HAVE = False
    print('torch 없음 → 정확해(S2) 생략', flush=True)


def run_cell(W, CAP, NARR, Li, s1_only=False):
    price = PriceRCR.from_json(a.params, Li=Li)
    cfg = Config(lam=a.lam, NARR=NARR, Mcyc=CAP, W=W, prune=False, pu_all=True,
                 phi=a.phi, Cf=a.Cf, Cp=a.Cp)
    I = Instance(TYPES, price, cfg)
    arrays = build(I, workers=a.workers, cache_dir=a.cache, tag=','.join(SEL))
    POL = {'S1': (prc.s1_mandatory_precise(I), 1.0)}
    if not s1_only:
        POL.update({'S0': (pol.b1_sell_all(I), None),
                    'S2b': (pol.b2_no_screening(I), None),
                    'INDEX': (pol.index_myopic(I), None)})
        for thr in range(I.cfg.SB):
            POL[f'Bfast{thr}'] = (pol.b_fast(I, thr), None)
    gstar = None
    if HAVE and not s1_only:
        S = ExactSolver(arrays, device='cuda' if torch.cuda.is_available() else 'cpu')
        gstar, h, _ = S.solve()
        POL['S2'] = (S.greedy(h), None)
    res = {}
    for nm, (acts, rr) in POL.items():
        m = M.outcome_metrics(I, arrays, acts, rho_required=rr, **EK)
        m['q'] = M.required_throughput(m['rho_required'], a.Lambda, a.ncenter, a.Dyear)
        if gstar: m['gap_pct'] = 100*(gstar-m['g'])/gstar
        res[nm] = m
    return I, gstar, res


def main():
    for W, CAP, NARR, Li in GRID:
        _tag = ''
        if a.phi != 1.0:      _tag += f'_phi{a.phi:g}'
        if a.Cf  != 20_000.:  _tag += f'_Cf{a.Cf:g}'
        if a.Cp  != 500_000.: _tag += f'_Cp{a.Cp:g}'
        if a.lam != 0.9:      _tag += f'_lam{a.lam:g}'
        jid = f'W{W}_cap{CAP}_narr{NARR}_li{Li:g}{_tag}'
        d = out / jid
        if (d/'DONE.json').exists():
            print('skip', jid, flush=True); continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        I, gstar, res = run_cell(W, CAP, NARR, Li)
        # S1inf — 같은 W·λ·Li 에서 용량만 무제한 (규제가 의도대로 작동한 상한)
        if not a.no_s1inf and CAP < W:
            _, _, r2 = run_cell(W, W, NARR, Li, s1_only=True)
            res['S1inf'] = r2['S1']
        rec = dict(cell=dict(W=W, Mcyc=CAP, CAP=I.CAP, NARR=NARR, lam=a.lam,
                             lam_eff=a.lam*NARR,
                             load=a.lam*NARR/max(I.CAP, 1), Li=Li),
                   summary=I.summary(), gstar=gstar, eps=EK, results=res, sec=time.time()-t0)
        (d/'result.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
        (d/'DONE.json').write_text(json.dumps({'jid': jid, 'sec': rec['sec']}))
        s1, s2b = res['S1'], res.get('S2b', {})
        print(f"[OK] {jid} |S|={len(I.ST):,} {rec['sec']:.0f}s  "
              f"S1미이행={s1['forced_rate']:.3f} S1_U={s1['U_kwh']:.1f} "
              f"S2b_U={s2b.get('U_kwh', float('nan')):.1f}", flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
