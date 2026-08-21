#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w6 [4] — 규모 전이 실험.  작은 W 에서 학습한 정책이 큰 W 에서도 쓰이는가.

    python scripts/transfer_w6.py --train 4 5 --eval 4 5 6 --seeds 0 1 2
    python scripts/transfer_w6.py ... --invariant     # W 무관 정규화판

인코더가 순열불변 Deep Sets 이고 배터리 특징이 6차원 고정이라 **구조적으로는**
학습 W 와 다른 W 의 텐서를 그대로 먹일 수 있다 (슬롯 수 Um·Sm 만 달라진다).
문제는 정규화다 — 다음 셋이 W 로 나눈다.

  encode.state_tensors : ctx = [Σn/Um, |sc|/Sm, CAP/(Um+Sm), tot/(Um+Sm)]
  net.encode           : cnt = [Σmu/Um, Σms/Sm]
  net(carry)           : prevU4/Um, remU/Um, freeU/wcap  (및 선별축 대응물)

즉 같은 물리적 점유가 W 마다 다른 입력값이 된다. 학습 W 에서만 보던 구간 밖으로
나가므로 전이가 깨질 수 있다. `--invariant` 는 이 분모들을 고정 상수 WREF 로
바꿔 «절대 점유 대수» 를 그대로 보게 한다.

산출: results/w6/transfer_w6.csv  (학습W × 평가W × 시드)
"""
import argparse, csv, json, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
CACHE = os.environ.get('BATDIAG_CACHE', '/home/user/batdiag-cache')
OUT = ROOT/'results'/'w6'


def build(W, dev, log=print):
    """(인스턴스, 솔버, g*) — 캐시가 있으면 빌드는 사실상 공짜다."""
    import torch
    from battery_diag.data import SEL_W5 as SEL, fleet_w5, load_cached
    from battery_diag.instance import Instance, PriceW5, Config
    from battery_diag.build import build as build_ram
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol
    _, FS = load_cached(str(ROOT/'data'))
    F = fleet_w5(ROOT/'data', sel=SEL)
    tot = sum(F[t][0] for t in SEL)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in SEL}
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                 F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, PriceW5.from_json(ROOT/'data'/'params_w5.json'), cfg)
    t0 = time.time()
    stream = len(I.ST) >= 40000
    A = (build_stream(I, cache_dir=CACHE, tag=','.join(SEL)) if stream
         else build_ram(I, cache_dir=CACHE, tag=','.join(SEL)))
    S = StreamSolver(A, device=dev) if stream else ExactSolver(A, device=dev)
    # g* 는 w5 에서 이미 정확해로 풀어 뒀다. 다시 풀면 W=6 만 1,082초가 더 든다.
    # 캐시된 값을 쓰되 출처를 남긴다 (없으면 그때만 푼다).
    f = ROOT/'results'/'w5'/f'W{W}.json'
    if f.exists():
        g = float(json.loads(f.read_text(encoding='utf-8'))['gstar'])
        log(f'  [W={W}] nS={len(I.ST):,}  g*={g:,.2f} (results/w5/W{W}.json)'
            f'  로드 {time.time()-t0:.0f}s')
    else:
        g, _h, _ = (S.solve(acts0=pol.index_myopic(I)) if stream else S.solve())
        g = float(g)
        log(f'  [W={W}] nS={len(I.ST):,}  g*={g:,.2f} (신규 계산)  {time.time()-t0:.0f}s')
    return I, S, g


def main():
    import numpy as np, torch
    from battery_diag.dcl import run_dcl, greedy_policy
    from battery_diag.encode import state_tensors
    from battery_diag.ckpt import Checkpoint
    from battery_diag import policies as pol

    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', type=int, default=[4, 5])
    ap.add_argument('--eval', nargs='+', type=int, default=[4, 5, 6])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--invariant', action='store_true', help='W 무관 정규화 (WREF 고정)')
    ap.add_argument('--wref', type=float, default=8.0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if a.invariant:
        from battery_diag import encode as _enc, net as _net
        _enc.WREF = a.wref; _net.WREF = a.wref
        print(f'[W 무관 정규화] WREF={a.wref}')

    Ws = sorted(set(a.train) | set(a.eval))
    print('== 인스턴스·정확해 준비 ==')
    inst = {W: build(W, dev) for W in Ws}
    tens = {W: state_tensors(inst[W][0], device=dev) for W in Ws}

    rows = []
    for Wtr in a.train:
        I_tr, S_tr, g_tr = inst[Wtr]
        for sd in a.seeds:
            print(f'== 학습 W={Wtr} seed={sd} ==', flush=True)
            t0 = time.time()
            with tempfile.TemporaryDirectory() as td:
                net, gbest, hist = run_dcl(
                    I_tr, S_tr, pol.index_myopic(I_tr), Checkpoint(Path(td), 'tr'),
                    rounds=a.rounds, epochs=40, seed=sd, device=dev,
                    log=lambda *x: None, gstar=g_tr, carry=True)
            tsec = time.time()-t0
            net.eval()
            for Wev in a.eval:
                I_ev, S_ev, g_ev = inst[Wev]
                te = time.time()
                acts = greedy_policy(net, I_ev, tens[Wev])
                g_net = float(S_ev.evaluate(acts)[0])
                gap = 100*(g_ev-g_net)/g_ev
                rows.append(dict(train_W=Wtr, eval_W=Wev, seed=sd,
                                 same_scale=(Wtr == Wev), gstar=g_ev, g_net=g_net,
                                 gap_pct=gap, train_sec=round(tsec, 1),
                                 eval_sec=round(time.time()-te, 1),
                                 invariant=bool(a.invariant)))
                print(f'   학습W={Wtr} → 평가W={Wev} seed{sd}: 갭 {gap:8.4f}%'
                      f'  ({time.time()-te:.0f}s)', flush=True)
                _write(rows, a)
    _write(rows, a)
    print(f'\n→ {OUT}/transfer_w6{a.tag}.csv  ({len(rows)}행)')


def _write(rows, a):
    f = OUT/f'transfer_w6{a.tag}.csv'
    with open(f, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
