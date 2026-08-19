#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W-정식화 본 실험 — 정확해 + 벤치마크 + 강제매각 통계 + 정책구조.

    python scripts/run_w.py <W> [<W> ...]

각 W 마다 results/w_model/W{W}.json 을 남긴다. 이미 있으면 건너뛴다.

벤치마크 정의 (W-정식화판 — 기존 정식화와 **정의가 같다**).
  B1     전량 미검사 즉시매각. 창고에 아무것도 남기지 않는다.
  B2     선별 없음. 미검사에서 순가치 상위 CAP 건만 직행 정밀검사, 나머지 즉시매각.
  B_fast 원고 구조. 전량 신속검사 후 신호 thr 이상 상위 CAP 건만 정밀, 나머지 매각.
         신속검사는 미검사→선별완료 이동이라 총 점유가 불변이므로 W 가 이 정책을
         제약하지 않는다 — 그래서 "자리가 허용하는 한" 이라는 단서가 필요 없고,
         정의가 기존과 정확히 같다. thr 은 0..SB-1 중 최선을 고른다.
  INDEX  근시안 EVSI 지표. 자원이 분리돼 신속·정밀 결정이 독립이라는 근사.
이 넷 모두 W 를 직접 참조하지 않는다. W 는 도착 시 강제매각으로만 작용하므로
정책 정의를 바꿀 필요가 없다 — 기존 칸과의 갭 비교가 그래서 성립한다.
"""
import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

SEL = ['레이', '코나', 'SM3']
CACHE = os.environ.get('BATDIAG_CACHE', '/home/user/batdiag-cache')
OUT = Path(__file__).resolve().parents[1]/'results'/'w_model'


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol

    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)

    for W in [int(x) for x in sys.argv[1:]]:
        f = OUT/f'W{W}.json'
        if f.exists():
            print(f'[skip] W={W}', flush=True); continue
        t_all = time.time()
        cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                     F_E=FS['F_E'], F_U=FS['F_U'])
        I = Instance(types, price, cfg); nS = len(I.ST)

        # 규모 추정 먼저 — 스트리밍 경로가 걸리는지 확인하고 들어간다.
        import random
        rng = random.Random(0); smp = rng.sample(range(nS), min(120, nS))
        rows = nnz = 0
        for si in smp:
            st = I.ST[si]
            for a in I.actions(st):
                rows += 1; nnz += len(I.step_dist(st, a))
            I._acts.pop(st, None)
        nA_est = rows/len(smp)*nS
        nnz_est = nnz/len(smp)*nS
        gb = nnz_est*12/1e9
        stream = gb > float(os.environ.get('BATDIAG_STREAM_GB', 8))
        print(f'W={W}: nS={nS:,}  nA≈{nA_est:,.0f}  nnz≈{nnz_est:,.0f}  CSR≈{gb:.1f}GB  '
              f'→ {"스트리밍" if stream else "in-RAM"}', flush=True)

        t0 = time.time()
        if stream:
            A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL), log=print)
            t_build = time.time()-t0
            S = StreamSolver(A, device=dev)
            t0 = time.time(); gstar, h, it = S.solve(acts0=pol.index_myopic(I), log=print)
        else:
            A = build(I, cache_dir=CACHE, tag=','.join(SEL))
            t_build = time.time()-t0
            S = ExactSolver(A, device=dev)
            t0 = time.time(); gstar, h, it = S.solve()
        t_solve = time.time()-t0
        nnz_true = int(A['indptr'][-1]) if 'indptr' in A else int(len(A['probs']))
        n_sa = int(len(A['rsa']))
        print(f'  build {t_build:.1f}s  solve {t_solve:.1f}s ({it}회)  g*={gstar:,.2f}  '
              f'n_sa={n_sa:,} nnz={nnz_true:,}', flush=True)

        acts = S.greedy(h)
        stats_opt = policy_stats(I, S, acts)
        bench = {}
        for name, a in (('B1', pol.b1_sell_all(I)), ('B2', pol.b2_no_screening(I)),
                        ('INDEX', pol.index_myopic(I))):
            bench[name] = float(S.evaluate(a)[0])
        bf = {}
        bf_acts = {}
        for thr in range(cfg.SB):
            a = pol.b_fast(I, thr); bf_acts[thr] = a
            bf[thr] = float(S.evaluate(a)[0])
        thr_best = max(bf, key=bf.get)
        bench['B_fast'] = bf[thr_best]
        stats_bfast = policy_stats(I, S, bf_acts[thr_best])
        gaps = {k: 100*(gstar-v)/gstar for k, v in bench.items()}
        for k in ('B1', 'B2', 'INDEX', 'B_fast'):
            print(f'  {k:7s} {bench[k]:>13,.0f}  갭 {gaps[k]:6.3f}%', flush=True)
        print(f'  강제매각(최적) {stats_opt["forced_units"]:.4f}대/기간 '
              f'{stats_opt["forced_value"]:,.0f}원  발동확률 {100*stats_opt["forced_prob"]:.2f}%',
              flush=True)
        print(f'  강제매각(B_fast) {stats_bfast["forced_units"]:.4f}대/기간 '
              f'{stats_bfast["forced_value"]:,.0f}원', flush=True)

        res = dict(W=W, nS=nS, n_sa=n_sa, nnz=nnz_true, csr_gb=nnz_true*12/1e9,
                   stream=stream, build_sec=t_build, solve_sec=t_solve, pi_iters=it,
                   gstar=float(gstar), bench=bench, gaps=gaps, B_fast_thr=int(thr_best),
                   B_fast_all={str(k): v for k, v in bf.items()},
                   opt=stats_opt, bfast=stats_bfast,
                   summary=I.summary(), wall_sec=time.time()-t_all)
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
        print(f'  → {f}  (총 {time.time()-t_all:.0f}s)', flush=True)
        del S, A
        if dev == 'cuda': torch.cuda.empty_cache()


def policy_stats(I, S, acts):
    """정상분포 가중 강제매각 통계 + 차종별 행동 비율."""
    import numpy as np
    d = np.asarray(S.stationary(acts)); d = d/d.sum()
    nT = len(I.TY); nS = len(I.ST)
    f_units = np.zeros(nS); f_val = np.zeros(nS); f_prob = np.zeros(nS)
    au = np.zeros((nS, nT, 4)); asr = np.zeros((nS, nT, 3))
    for si, st in enumerate(I.ST):
        n, scr = st
        a = I.actions(st)[acts[si]]
        sv, fv, pu, ss, ps = a
        for k in range(nT):
            au[si, k] = [sv[k], fv[k], pu[k], n[k]-sv[k]-fv[k]-pu[k]]
        for j, x in enumerate(scr):
            k = I.TY.index(x[0])
            asr[si, k, 0 if j in ss else (1 if j in ps else 2)] += 1
        f_units[si], f_val[si], f_prob[si] = forced_from_state(I, st, a)
    return dict(forced_units=float(d @ f_units), forced_value=float(d @ f_val),
                forced_prob=float(d @ f_prob),
                act_u=np.einsum('s,ska->ka', d, au).tolist(),
                act_s=np.einsum('s,ska->ka', d, asr).tolist(),
                types=list(I.TY))


def forced_from_state(I, st, a):
    """이 (상태, 행동) 의 (기대 강제매각 대수, 가치, 발동확률).

    step_dist 와 같은 분기를 탄다. 신속검사 결과에 따라 행동 뒤 선별완료 개수가
    달라지고 그만큼 도착이 쓸 자리가 줄므로, 그 분기를 재현해야 정확하다.
    """
    n, scr = st; sv, fv, pu, ss, ps = a; c = I.cfg
    rem = sum(n[k]-sv[k]-fv[k]-pu[k] for k in range(len(I.TY)))
    m0 = len([j for j in range(len(scr)) if j not in ss and j not in ps])
    combos = [(1.0, m0)]                      # (확률, 행동 뒤 선별완료 개수)
    for k, t in enumerate(I.TY):
        q = I.QP[t]; pdet = (1-q)*I.P_DET; pb, _ = I.BP[t]
        for _ in range(fv[k]):
            nxt = []
            for p0, m in combos:
                nxt.append((p0*pdet, m))               # 결함 검출 → 재활용, 자리 안 씀
                for b in range(c.SB):
                    nxt.append((p0*(1-pdet)*pb[b], m+1))
            combos = nxt
    u = v = pr = 0.0
    for p0, m in combos:
        for pa, _aa, rev, ns in I._arr_slack(I.W - (rem + m)):
            w = p0*pa
            u += w*ns; v += w*rev
            if ns > 0: pr += w
    return u, v, pr


if __name__ == '__main__':
    main()
