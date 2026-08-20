#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5 [4] — 실증 검증.  모형이 예측한 것이 실거래에 나타나는가.

    python scripts/validate_w5.py [--policy results/w5/W7.json]

w4 판과 달리 **원자료 전수**를 쓴다 (재사용 273건 / 재활용 172건, 2022-01~2025-10).
w4 에서는 원자료가 이 머신에 없어 TASK 가 준 손익배수에서 중앙 낙찰가를 되풀었고
4개 차종밖에 내지 못했다 (BLOCKED_w4.md). 그 제약이 풀렸다.

  §1 가격 예측 검증 — 건별 예측 vs 실거래. v5 / w4 / w5 파라미터 대비.
  §2 배율 검증     — 모형 함의 재사용/재활용 배율 vs 실측, 기간별.
  §3 구조 검증     — 실측 손익배수 서열 vs 최적정책 검사강도 서열 (--policy 필요).
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import types_w5, SEL_W5, CP, reg_from_ratio
from battery_diag.instance import PriceParams, PriceW5
sys.path.insert(0, str(ROOT/'scripts'))
from build_types_w5 import load_bids

W24 = '2024-01-01'


def med(x):
    return float(np.median(np.asarray(x, float)))


def bias_tab(pred, act):
    """예측/실측 → (중앙편의%, 중앙절대오차%)"""
    e = np.asarray(pred, float)/np.asarray(act, float) - 1.0
    return 100*med(e), 100*med(np.abs(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='')
    ap.add_argument('--out', default=str(ROOT/'results'/'w5'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    R, C = load_bids()
    T = types_w5()
    pv5 = PriceParams.from_json(ROOT/'data'/'params_v5.json')
    pw4 = PriceParams.from_json(ROOT/'data'/'params.json')
    pw5 = PriceW5.from_json(ROOT/'data'/'params_w5.json')
    pw5m = PriceW5.from_json(ROOT/'data'/'params_w5.json'); pw5m.smear = False

    # ---------------- §1 가격 예측 ----------------
    R = R.copy()
    R['p_v5'] = [float(pv5.Erev(s, c)) for s, c in zip(R.s, R.kwh)]
    R['p_w4'] = [float(pw4.Erev(s, c)) for s, c in zip(R.s, R.kwh)]
    R['p_w5'] = [float(pw5.Erev(s, c)) for s, c in zip(R.s, R.kwh)]      # 정상상태 Li=Li_ref
    R['p_w5m'] = [float(pw5m.Erev(s, c)) for s, c in zip(R.s, R.kwh)]
    # 시점 리튬을 넣은 판 — 회귀 자체의 적합도 (정상상태 가정을 뺀 상한)
    R['p_w5t'] = np.exp(pw5.reuse_c0 + pw5.reuse_cap*np.log(R.kwh) + pw5.reuse_s*np.log(R.s)
                        + pw5.reuse_li*np.log(R['리튬']) + 0.5*pw5.reuse_sd**2)
    C = C.copy()
    C['v_w4'] = pw4.p_rc*C.kwh
    C['v_w5'] = [float(pw5.VS(c)) for c in C.kwh]
    C['v_w5t'] = np.exp(pw5.recyc_c0 + pw5.recyc_cap*np.log(C.kwh)
                        + pw5.recyc_li*np.log(C['리튬']) + 0.5*pw5.recyc_sd**2)

    print('=' * 78)
    print('§1 재사용 낙찰가 예측 검증  — 건별 예측 vs 실거래 (팩당, 원)')
    print('=' * 78)
    lines1 = []
    for wname, sub in (('전기간 (n=%d)' % len(R), R),
                       ('2024~  (n=%d)' % (R.date >= W24).sum(), R[R.date >= W24])):
        print(f'\n[{wname}]')
        print(f"{'파라미터':<22}{'중앙편의%':>10}{'중앙절대오차%':>13}")
        for lbl, col in (('v5 (구)', 'p_v5'), ('w4 (재캘리브)', 'p_w4'),
                         ('w5 정상상태 Li=고정', 'p_w5'), ('w5 중앙값판(smear=F)', 'p_w5m'),
                         ('w5 시점리튬', 'p_w5t')):
            b, m = bias_tab(sub[col], sub['pack'])
            print(f'{lbl:<22}{b:>10.1f}{m:>13.1f}')
            lines1.append(dict(window=wname, params=lbl, bias=b, mdae=m, n=len(sub)))

    print('\n[차종×용량별 — 전기간, w5 정상상태]')
    print(f"{'유형':<12}{'n':>4}{'실측중앙':>12}{'예측w5':>12}{'편의%':>8}"
          f"{'예측w4':>12}{'편의%':>8}")
    rows_t = []
    for t, g in R.groupby('key'):
        if len(g) < 3: continue
        a_ = med(g['pack']); p5 = med(g['p_w5']); p4 = med(g['p_w4'])
        print(f'{t:<12}{len(g):4d}{a_:12,.0f}{p5:12,.0f}{100*(p5/a_-1):8.1f}'
              f'{p4:12,.0f}{100*(p4/a_-1):8.1f}')
        rows_t.append(dict(type=t, n=len(g), med_act=a_, med_w5=p5,
                           bias_w5=100*(p5/a_-1), med_w4=p4, bias_w4=100*(p4/a_-1)))
    print(f"{'— 유형별 편의 중앙':<12}{'':>4}{'':>12}{'':>12}"
          f"{med([r['bias_w5'] for r in rows_t]):8.1f}{'':>12}"
          f"{med([r['bias_w4'] for r in rows_t]):8.1f}")

    # 기댓값 검증 — MDP 가 쓰는 것은 E[P] 이므로 **평균 대 평균**으로 봐야 한다.
    # 위의 중앙편의는 기댓값 예측기를 중앙값에 대본 것이라 구조적으로 +로 뜬다
    # (로그정규에서 평균/중앙값 = exp(σ²/2) = 재사용 1.134 · 재활용 1.143).
    print('\n[기댓값 검증 — 예측 평균 / 실측 평균, 시점리튬 조건부]')
    print(f"{'채널':<10}{'실측평균':>14}{'smear=T':>14}{'비':>7}{'smear=F':>14}{'비':>7}")
    lines_mean = []
    for nm, dd, ln, sd in (
            ('재사용', R, (pw5.reuse_c0 + pw5.reuse_cap*np.log(R.kwh)
                        + pw5.reuse_s*np.log(R.s) + pw5.reuse_li*np.log(R['리튬'])),
             pw5.reuse_sd),
            ('재활용', C, (pw5.recyc_c0 + pw5.recyc_cap*np.log(C.kwh)
                        + pw5.recyc_li*np.log(C['리튬'])), pw5.recyc_sd)):
        mT = float(np.exp(ln + 0.5*sd**2).mean()); mF = float(np.exp(ln).mean())
        act = float(dd['pack'].mean())
        print(f'{nm:<10}{act:14,.0f}{mT:14,.0f}{mT/act:7.3f}{mF:14,.0f}{mF/act:7.3f}')
        lines_mean.append(dict(channel=nm, act_mean=act, pred_mean_smear=mT,
                               ratio_smear=mT/act, pred_mean_nosmear=mF, ratio_nosmear=mF/act))
    print('  → 로그정규 보정을 넣은 쪽이 표본평균을 맞춘다. MDP 는 기대보상을')
    print('    최대화하므로 smear=True 가 맞다 (PriceW5 docstring 참조).')

    print('\n[재활용 매각가치 — 전기간]')
    print(f"{'파라미터':<22}{'중앙편의%':>10}{'중앙절대오차%':>13}")
    lines_rc = []
    for lbl, col in (('w4 (p_rc×kWh)', 'v_w4'), ('w5 정상상태', 'v_w5'), ('w5 시점리튬', 'v_w5t')):
        b, m = bias_tab(C[col], C['pack'])
        print(f'{lbl:<22}{b:>10.1f}{m:>13.1f}')
        lines_rc.append(dict(params=lbl, bias=b, mdae=m, n=len(C)))

    # ---------------- §2 배율 ----------------
    print()
    print('=' * 78)
    print('§2 재사용/재활용 배율 — 모형 함의 vs 실측')
    print('=' * 78)
    print(f"{'기간':<12}{'실측 재사용/kWh':>16}{'실측 재활용/kWh':>16}{'실측배율':>9}")
    per = [('전기간', None), ('2023~', '2023-01-01'), ('2024~', W24)]
    rows_m = []
    for nm, lo in per:
        rr = R if lo is None else R[R.date >= lo]
        cc = C if lo is None else C[C.date >= lo]
        mu_r, mu_c = med(rr['unit']), med(cc['unit'])
        print(f'{nm:<12}{mu_r:16,.0f}{mu_c:16,.0f}{mu_r/mu_c:9.2f}')
        rows_m.append(dict(window=nm, reuse_kwh=mu_r, recyc_kwh=mu_c, mult_emp=mu_r/mu_c))
    last = R.date.max() - pd.DateOffset(months=12)
    mu_r, mu_c = med(R[R.date >= last]['unit']), med(C[C.date >= last]['unit'])
    print(f"{'최근12개월':<12}{mu_r:16,.0f}{mu_c:16,.0f}{mu_r/mu_c:9.2f}")
    rows_m.append(dict(window='최근12개월', reuse_kwh=mu_r, recyc_kwh=mu_c, mult_emp=mu_r/mu_c))

    print(f"\n{'유형':<12}{'용량':>6}{'μ':>7}{'모형배율 w5':>12}{'모형배율 w4':>12}"
          f"{'실측배율':>9}")
    for t in [k for k in T if k != '_meta' and T[k]['eligible']]:
        x = T[t]; cap, mu = x['cap'], x['mu']
        m5 = float(pw5.Erev(mu, cap)/pw5.VS(cap))
        m4 = float(pw4.Erev(mu, cap)/(pw4.p_rc*cap))
        me = x['med_re']/x['med_rc']
        print(f'{t:<12}{cap:6.1f}{mu:7.3f}{m5:12.2f}{m4:12.2f}{me:9.2f}')

    # ---------------- §3 구조 ----------------
    print()
    print('=' * 78)
    print('§3 구조 검증 — 실측 손익배수 vs 모형')
    print('=' * 78)
    elig = [k for k in T if k != '_meta' and T[k]['eligible']]
    print(f"{'유형':<12}{'q_P':>6}{'실측배수':>9}{'모형배수':>9}{'실측등급':>9}{'모형등급':>9}{'':>6}")
    rows_r = []
    for t in sorted(elig, key=lambda k: T[k]['ratio_emp']):
        x = T[t]; cap, mu, q = x['cap'], x['mu'], x['qP']
        rm = float(q*(pw5.Erev(mu, cap) - pw5.VS(cap))/CP)
        ok = reg_from_ratio(rm) == x['reg_emp']
        print(f"{t:<12}{q:6.2f}{x['ratio_emp']:9.2f}{rm:9.2f}{x['reg_emp']:>9}"
              f"{reg_from_ratio(rm):>9}{'  일치' if ok else '  불일치'}")
        rows_r.append(dict(type=t, qP=q, ratio_emp=x['ratio_emp'], ratio_model=rm,
                           reg_emp=x['reg_emp'], reg_model=reg_from_ratio(rm), agree=ok))
    tv, c, d, npair = tau([r['ratio_emp'] for r in rows_r], [r['ratio_model'] for r in rows_r])
    print(f'\n켄달 타우(실측 vs 모형 손익배수) = {tv:+.3f}   '
          f'(일치쌍 {c} / 역전쌍 {d} / 전체 {npair})')
    print(f"등급 일치 {sum(r['agree'] for r in rows_r)}/{len(rows_r)}")

    with open(out/'validation_price.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_t[0])); w.writeheader(); w.writerows(rows_t)
    with open(out/'validation_ratio.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_r[0])); w.writeheader(); w.writerows(rows_r)
    json.dump(dict(price_windows=lines1, recycle=lines_rc, multiples=rows_m,
                   mean_check=lines_mean,
                   kendall_tau=tv, tau_conc=c, tau_disc=d, tau_pairs=npair,
                   reg_agree=sum(r['agree'] for r in rows_r), reg_n=len(rows_r)),
              open(out/'validation_summary.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    pol_res = policy_section(a.policy, T, pw5, rows_r) if a.policy else None
    if pol_res is not None:
        d = json.load(open(out/'validation_summary.json', encoding='utf-8'))
        d['policy'] = pol_res
        json.dump(d, open(out/'validation_summary.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)


def tau(x, y):
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            v = (x[i]-x[j])*(y[i]-y[j])
            if v > 0: c += 1
            elif v < 0: d += 1
    return ((c-d)/(c+d) if c+d else float('nan')), c, d, n*(n-1)//2


def policy_section(path, T, pw5, rows_r):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    ty = d['opt']['types']; au = d['opt']['act_u']; asr = d['opt']['act_s']
    lam_eff = d['summary']['lam_eff']
    tot_n = sum(T[t]['n'] for t in ty)
    print()
    print('=' * 78)
    print(f"§3(b) 최적정책 검사강도 — {path} (W={d['W']}, g*={d['gstar']:,.0f}, "
          f"CAP={d['summary']['CAP']})")
    print('=' * 78)
    print('  강도 = 창고에 들어온 그 유형 1대당 비율. 분모는 sell+fast+pu+hold')
    print('  (도착 즉시 강제매각된 대수는 정책이 만진 적이 없으므로 뺀다).')
    print(f"{'유형':<12}{'실측배수':>9}{'도착/기':>8}{'취급/기':>8}{'매각':>8}"
          f"{'신속':>8}{'정밀':>8}{'신속률':>8}{'정밀률':>8}")
    tab = []
    for k, t in enumerate(ty):
        sell, fast, pu, hold = au[k]
        s_sell, s_ps, s_hold = asr[k]
        arr = lam_eff*T[t]['n']/tot_n
        handled = sell + fast + pu + hold
        f_rate = fast/handled if handled > 0 else 0.0
        p_rate = (pu + s_ps)/handled if handled > 0 else 0.0
        tab.append((t, T[t]['ratio_emp'], p_rate, f_rate, pu + s_ps))
        print(f"{t:<12}{T[t]['ratio_emp']:9.2f}{arr:8.4f}{handled:8.4f}{sell:8.4f}"
              f"{fast:8.4f}{pu+s_ps:8.4f}{100*f_rate:7.1f}%{100*p_rate:7.1f}%")
    pslot = sum(x[4] for x in tab)
    if pslot > 0:
        print(f"  정밀 슬롯 사용 {pslot:.4f}/{d['summary']['CAP']} — 유형별 점유율 "
              + ', '.join(f'{x[0]} {100*x[4]/pslot:.1f}%' for x in tab if x[4] > 1e-9))
    fslot = sum(x[3]*1 for x in tab)
    print('\n  [선별이 퍼지는가] 신속검사 비중')
    ftot = sum(au[k][1] for k in range(len(ty)))
    if ftot > 1e-12:
        for k, t in enumerate(ty):
            print(f'    {t:<12} {100*au[k][1]/ftot:5.1f}%')
        top = max(range(len(ty)), key=lambda k: au[k][1])
        print(f'    → 최대 집중 {ty[top]} {100*au[top][1]/ftot:.1f}%')
    else:
        print('    신속검사 미사용')
    tv, c, dd, npair = tau([x[1] for x in tab], [x[2] for x in tab])
    o_r = ' < '.join(x[0] for x in sorted(tab, key=lambda x: x[1]))
    o_p = ' < '.join(x[0] for x in sorted(tab, key=lambda x: x[2]))
    print(f'\n  실측 손익배수 오름차순    : {o_r}')
    print(f'  모형 정밀검사율 오름차순  : {o_p}')
    print(f'  → 켄달 타우 {tv:+.3f} (일치 {c} / 역전 {dd} / 전체 {npair}쌍)  '
          f"서열 완전일치: {'예' if o_r == o_p else '아니오'}")
    print('\n  [이분 판정] 실측 R1 vs 모형이 정밀검사를 쓰는가')
    binary = []
    for t, r_emp, p_rate, f_rate, pv in tab:
        emp = reg_from_ratio(r_emp)
        mod = 'R2' if p_rate > 1e-9 else 'R1'
        ok = (emp == 'R1') == (mod == 'R1')
        binary.append(dict(type=t, reg_emp=emp, p_rate=p_rate, reg_model=mod, agree=ok))
        print(f"    {t:<12} 실측 {emp}(배수 {r_emp:.2f})  모형 정밀검사율 "
              f"{100*p_rate:5.1f}% → {mod}   {'일치' if ok else '불일치'}")
    return dict(policy_file=str(path), W=d['W'], gstar=d['gstar'],
                CAP=d['summary']['CAP'],
                rows=[dict(type=x[0], ratio_emp=x[1], p_rate=x[2], f_rate=x[3],
                           p_slots=x[4]) for x in tab],
                fast_share={ty[k]: (au[k][1]/ftot if ftot > 1e-12 else 0.0)
                            for k in range(len(ty))},
                fast_total=ftot,
                order_emp=o_r, order_model=o_p, tau=tv, tau_conc=c, tau_disc=dd,
                order_exact=(o_r == o_p), binary=binary)


if __name__ == '__main__':
    main()
