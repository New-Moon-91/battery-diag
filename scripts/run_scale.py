#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scale1 관문 실험 — 실규모에서 휴리스틱 대 유체 LP 상계.

    python scripts/run_scale.py --fleet pool16 --lam 50 --W 1097 --M 10,20,30,40,50,60,80 \
        --T 4000 --burn 400 --seeds 0,1,2 --out results/scale/scale1_lam50.json

fleet
  w5      types_w5.json 의 적격 11유형 (CSV 불필요 — 컨테이너 시험용)
  pool16  pool.csv 차종×용량 전 유형 중 n ≥ POOL_MIN (SPEC §2 의 16유형, 서버 전용)
  sel4    w5 확정 4유형 (소형 대조용)

각 (M) 셀에 대해 LP 상계·μ·경로, 그리고 정책별 시드 평균 g 와 상계 대비 갭(%)을 적는다.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.instance import Instance, PriceW5, Config
from battery_diag.simulate import Simulator
from battery_diag import scale as S


def build_types(fleet: str, data=ROOT/'data'):
    from battery_diag.data import fleet_w5, SEL_W5, POOL_MIN
    if fleet == 'sel4':
        F = fleet_w5(data, sel=SEL_W5); sel = list(SEL_W5)
    elif fleet == 'w5':
        F = fleet_w5(data); sel = list(F)
    elif fleet == 'pool16':
        from battery_diag.data import load_cached
        F, _ = load_cached(data, by_capacity=True)
        sel = [t for t in F if F[t][0] >= POOL_MIN]
    else:
        raise ValueError(fleet)
    tot = sum(F[t][0] for t in sel)
    return {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in sel}


def failure_shares_or_default(data=ROOT/'data'):
    try:
        from battery_diag.data import load_cached
        _, FS = load_cached(data)
        return FS['F_E'], FS['F_U']
    except Exception:
        return 278/354, 62/354


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fleet', default='pool16')
    ap.add_argument('--lam', type=float, default=50.0, help='기간당 평균 도착 (포아송)')
    ap.add_argument('--W', type=int, default=1097)
    ap.add_argument('--M', default='10,20,30,40,50,60,80')
    ap.add_argument('--T', type=int, default=4000)
    ap.add_argument('--burn', type=int, default=400)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--kappa', default='1.0')
    ap.add_argument('--beta', default='', help='QUEUE 유한버퍼 배수 목록 (쉼표, 비우면 없음). 버퍼 = beta·CAP 대')
    ap.add_argument('--mu_scale', default='1.0',
                    help='QUEUE 의 경로 결정에 쓰는 그림자가격 배율 (쉼표). 1 미만이면 LP 의 '
                         '경계 유형을 FAST 로 끌어와 노는 용량을 채운다 (M=16 셀에서 용량 미사용 관측)')
    ap.add_argument('--arrivals', default='poisson', choices=('poisson', 'binomial'))
    ap.add_argument('--Cp', type=float, default=500_000)
    ap.add_argument('--out', default=str(ROOT/'results'/'scale'/'scale1.json'))
    a = ap.parse_args()

    types = build_types(a.fleet)
    FE, FU = failure_shares_or_default()
    price = PriceW5.from_json(ROOT/'data'/'params_w5.json')
    Ms = [int(x) for x in a.M.split(',')]; seeds = [int(x) for x in a.seeds.split(',')]
    kappas = [float(x) for x in a.kappa.split(',')]
    mu_scales = [float(x) for x in a.mu_scale.split(',')]
    betas = [float(x) for x in a.beta.split(',')] if a.beta else []
    # 포아송 평균 λ = lam·NARR 이 되도록 lam=1, NARR=λ (정수). 이항이면 NARR 슬롯·lam.
    if a.arrivals == 'poisson':
        lam, NARR = 1.0, int(round(a.lam))
    else:
        NARR = int(np.ceil(a.lam/0.9)); lam = a.lam/NARR
    lam_eff = lam*NARR
    out = dict(args=vars(a), types={t: list(v) for t, v in types.items()}, lam_eff=lam_eff,
               F_E=FE, F_U=FU, cells=[])
    print(f'fleet={a.fleet} |T|={len(types)} λ_eff={lam_eff} W={a.W} M={Ms} T={a.T} seeds={seeds}',
          flush=True)
    for M in Ms:
        t0 = time.time()
        cfg = Config(lam=lam, NARR=NARR, Cp=a.Cp, Cf=20_000, phi=1.0, Mcyc=M, W=a.W, F_E=FE, F_U=FU)
        I = Instance.lite(types, price, cfg)
        fb = S.fluid_bound(I, lam_eff, I.CAP)
        sim = Simulator(I, arrivals=a.arrivals)
        pols = {'B1': S.pol_b1(I), 'INDEX': S.pol_index(I),
                'LAGR': S.pol_lagr(I, fb['mu'], lam_eff)}
        for thr in range(cfg.SB):
            pols[f'B_fast{thr}'] = S.pol_b_fast(I, thr)
        for kp in kappas:
            for ms in mu_scales:
                tag = f'QUEUE{kp:g}' + (f'_m{ms:g}' if ms != 1.0 else '')
                pols[tag] = S.pol_lagr(I, fb['mu'], lam_eff, kappa=kp, mu_route=fb['mu']*ms)
            for bt in betas:
                pols[f'QUEUE{kp:g}_b{bt:g}'] = S.pol_lagr(I, fb['mu'], lam_eff, kappa=kp, beta=bt)
        cell = dict(M=M, CAP=I.CAP, Hslot=I.Hslot, load=lam_eff/max(I.CAP, 1), bound=fb['bound'],
                    mu=fb['mu'], lagr_min=fb['lagr_min'], prec_use_lp=fb['prec_use'],
                    route=fb['route'], REG=I.REG, DUMP=sorted(I.DUMP), policies={})
        for name, act in pols.items():
            runs = [S.evaluate(sim, act, T=a.T, seed=s, burn=a.burn) for s in seeds]
            gs = np.array([r['g'] for r in runs])
            cell['policies'][name] = dict(
                g=float(gs.mean()), g_sd=float(gs.std(ddof=1)) if len(gs) > 1 else 0.0,
                gap_pct=float(100*(1-gs.mean()/fb['bound'])),
                occ=float(np.mean([r['occ'] for r in runs])),
                prec_per=float(np.mean([r['prec_per'] for r in runs])),
                fast_per=float(np.mean([r['fast_per'] for r in runs])),
                forced_per=float(np.mean([r['forced_per'] for r in runs])),
                runs=runs)
            print(f'  M={M:3d} CAP={I.CAP:3d} {name:9s} g={gs.mean():14,.0f} '
                  f'gap={cell["policies"][name]["gap_pct"]:7.3f}%  '
                  f'occ={cell["policies"][name]["occ"]:7.1f} prec={cell["policies"][name]["prec_per"]:6.2f} '
                  f'forced={cell["policies"][name]["forced_per"]:6.2f}', flush=True)
        # B_fast 최선 thr
        bf = {k: v['g'] for k, v in cell['policies'].items() if k.startswith('B_fast')}
        cell['B_fast_best'] = max(bf, key=bf.get)
        cell['sec'] = time.time()-t0
        print(f'  M={M} LP={fb["bound"]:,.0f} μ={fb["mu"]:,.0f} route={fb["route"]} ({cell["sec"]:.0f}s)',
              flush=True)
        out['cells'].append(cell)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('done', a.out)


if __name__ == '__main__':
    main()
