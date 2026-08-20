#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4 [4] — 실증 검증.  모형이 예측한 것이 실거래에 나타나는가.

    python scripts/validate_w4.py [--policy results/w4/W7.json]

세 절을 낸다.
  §1 가격 예측 검증 — 재사용 낙찰가 예측 vs 실측 중앙값 (구/신 파라미터 대비)
  §2 배율 검증     — 모형 함의 재사용/재활용 배율 vs 실측
  §3 구조 검증     — 실측 손익배수 서열 vs 최적정책 검사강도 서열 (--policy 필요)

**실측값의 출처.** 원자료(입찰_재사용__0819.xlsx / 입찰_재활용__0819.xlsx)는 이 머신에
없다. 대신 TASK 가 준 차종별 **실측 손익배수**에서 실측 중앙 낙찰가를 되풀어 쓴다.

    손익배수 r = (E[재사용] - V^S) / (C_p/q_P)   ⇒   E[재사용] = r*(C_p/q_P) + p_rc*kWh

이 되풀기는 항등식이라 근사가 아니다. 다만 r 이 소수 2자리로 주어져 있어
E 에 ±0.005*(C_p/q_P) = ±4,869원의 반올림 오차가 남는다 (SM3 ±1.6%, 코나 ±0.14%).
BLOCKED_w4.md 참조 — 원자료가 붙으면 94건 전수로 다시 낸다.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import load_cached, RATIO_EMP, SOH_MED_EMP, SEL_W4, REG_EMP
from battery_diag.instance import PriceParams

CP = 500_000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='')
    ap.add_argument('--out', default=str(ROOT/'results'/'w4'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pv5 = PriceParams.from_json(ROOT/'data'/'params_v5.json')
    pw4 = PriceParams.from_json(ROOT/'data'/'params.json')
    qP = json.load(open(ROOT/'data'/'params.json'))['q_P']
    BE = CP/qP
    FLEET, _ = load_cached(ROOT/'data')

    rows = []
    for t in SEL_W4:
        cap = FLEET[t][2]; s = SOH_MED_EMP[t]; r = RATIO_EMP[t]
        VS = pw4.p_rc*cap
        E_emp = r*BE + VS
        E_v5 = float(pv5.Erev(s, cap)); E_w4 = float(pw4.Erev(s, cap))
        rows.append(dict(model=t, cap=cap, soh=s, reg=REG_EMP[t], ratio_emp=r,
                         E_emp=E_emp, E_v5=E_v5, E_w4=E_w4,
                         bias_v5=100*(E_v5/E_emp-1), bias_w4=100*(E_w4/E_emp-1),
                         kwh_emp=E_emp/cap, kwh_w4=E_w4/cap,
                         mult_emp_pooled=E_emp/VS,
                         mult_v5=E_v5/(pv5.p_rc*cap), mult_w4=E_w4/VS,
                         ratio_model_w4=(E_w4-VS)/BE, ratio_model_v5=(E_v5-pv5.p_rc*cap)/BE))

    import csv
    with open(out/'validation_price.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    med = lambda xs: sorted(xs)[len(xs)//2] if len(xs) % 2 else \
        0.5*(sorted(xs)[len(xs)//2-1]+sorted(xs)[len(xs)//2])
    print(f'손익분기 C_p/q_P = {BE:,.0f}원   p_rc(w4) = {pw4.p_rc:,.0f}  p_rc(v5) = {pv5.p_rc:,.0f}')
    print()
    print('§1 재사용 낙찰가 (원)')
    print(f"{'차종':<5}{'용량':>6}{'SOH':>7}{'실측중앙':>12}{'예측v5':>12}{'편의v5%':>9}"
          f"{'예측w4':>12}{'편의w4%':>9}")
    for x in rows:
        print(f"{x['model']:<5}{x['cap']:6.1f}{x['soh']:7.3f}{x['E_emp']:12,.0f}"
              f"{x['E_v5']:12,.0f}{x['bias_v5']:9.1f}{x['E_w4']:12,.0f}{x['bias_w4']:9.1f}")
    print(f"{'중앙':<5}{'':>6}{'':>7}{'':>12}{'':>12}"
          f"{med([x['bias_v5'] for x in rows]):9.1f}{'':>12}"
          f"{med([x['bias_w4'] for x in rows]):9.1f}")
    print(f"중앙절대오차: v5 {med([abs(x['bias_v5']) for x in rows]):.1f}%  "
          f"w4 {med([abs(x['bias_w4']) for x in rows]):.1f}%")
    print()
    print('§2 재사용/재활용 배율 (E[재사용] / (p_rc*kWh))')
    print(f"{'차종':<5}{'모형v5':>9}{'모형w4':>9}{'실측(pooled p_rc)':>20}")
    for x in rows:
        print(f"{x['model']:<5}{x['mult_v5']:9.2f}{x['mult_w4']:9.2f}{x['mult_emp_pooled']:20.2f}")
    print()
    print('§3 손익배수 — 실측 vs 모형')
    print(f"{'차종':<5}{'실측':>8}{'모형v5':>9}{'모형w4':>9}{'REG(실측)':>10}")
    for x in rows:
        print(f"{x['model']:<5}{x['ratio_emp']:8.2f}{x['ratio_model_v5']:9.2f}"
              f"{x['ratio_model_w4']:9.2f}{x['reg']:>10}")

    if a.policy:
        d = json.loads(Path(a.policy).read_text())
        ty = d['opt']['types']; au = d['opt']['act_u']; asr = d['opt']['act_s']
        tot_n = sum(FLEET[t][0] for t in ty)
        lam_eff = d['summary']['lam_eff']
        print()
        print(f"§3(b) 최적정책 검사강도 — {a.policy} (W={d['W']}, g*={d['gstar']:,.0f}, "
              f"CAP={d['summary']['CAP']})")
        print('  강도 = 창고에 들어온 그 차종 1대당 비율. 분모는 sell+fast+pu+hold '
              '(도착 즉시 강제매각된 대수는 정책이 만진 적이 없으므로 뺀다).')
        print(f"{'차종':<5}{'실측배수':>9}{'모형배수':>9}{'도착/기':>8}{'취급/기':>8}"
              f"{'매각':>8}{'신속':>8}{'정밀':>8}{'신속률':>8}{'정밀률':>8}")
        tab = []
        rmodel = {x['model']: x['ratio_model_w4'] for x in rows}
        for k, t in enumerate(ty):
            sell, fast, pu, hold = au[k]
            s_sell, s_ps, s_hold = asr[k]
            arr = lam_eff*FLEET[t][0]/tot_n
            handled = sell + fast + pu + hold
            f_rate = fast/handled if handled > 0 else 0.0
            p_rate = (pu + s_ps)/handled if handled > 0 else 0.0
            tab.append((t, RATIO_EMP.get(t, float('nan')), rmodel.get(t, float('nan')),
                        p_rate, f_rate, pu + s_ps))
            print(f"{t:<5}{RATIO_EMP.get(t,float('nan')):9.2f}{rmodel.get(t,float('nan')):9.2f}"
                  f"{arr:8.4f}{handled:8.4f}{sell:8.4f}{fast:8.4f}{pu+s_ps:8.4f}"
                  f"{100*f_rate:7.1f}%{100*p_rate:7.1f}%")
        pslot = sum(x[5] for x in tab)
        print(f"  정밀 슬롯 사용 {pslot:.4f}/{d['summary']['CAP']}  "
              f"— 차종별 점유율 " + ', '.join(f"{x[0]} {100*x[5]/pslot:.1f}%"
                                            for x in tab if x[5] > 1e-9))

        def tau(xs, ys):
            n = len(xs); c = dd = 0
            for i in range(n):
                for j in range(i+1, n):
                    s_ = (xs[i]-xs[j])*(ys[i]-ys[j])
                    if s_ > 0: c += 1
                    elif s_ < 0: dd += 1
            return (c-dd)/(c+dd) if c+dd else float('nan'), c, dd, n*(n-1)//2

        # 가장 거친, 그래서 가장 결정적인 대조: 검사하느냐 마느냐.
        print('  [이분 판정] 실측 R1/R2 vs 모형이 정밀검사를 쓰는가')
        ok = True
        for t, r_emp, r_mod, p_rate, f_rate, pv in tab:
            emp = REG_EMP.get(t, '?')
            mod = 'R2' if p_rate > 1e-9 else 'R1'
            ok &= (emp == mod)
            print(f"    {t:<5} 실측 {emp}(배수 {r_emp:.2f})  모형 정밀검사율 "
                  f"{100*p_rate:5.1f}% → {mod}   {'일치' if emp == mod else '불일치'}")
        print(f"    → 4/4 일치: {'예' if ok else '아니오'}")

        for lbl, key in (('실측 손익배수', 1), ('모형 손익배수', 2)):
            t_p, c, dd, npair = tau([x[key] for x in tab], [x[3] for x in tab])
            o_r = ' < '.join(x[0] for x in sorted(tab, key=lambda x: x[key]))
            o_p = ' < '.join(x[0] for x in sorted(tab, key=lambda x: x[3]))
            print(f"  {lbl} 오름차순: {o_r}")
            print(f"  모형 정밀검사율 오름차순: {o_p}")
            print(f"  → 켄달 타우 {t_p:+.3f}  (일치쌍 {c} / 역전쌍 {dd} / "
                  f"동점 제외 {npair-c-dd}, 전체 {npair}쌍)  "
                  f"서열 완전일치: {'예' if o_r == o_p else '아니오'}")


if __name__ == "__main__":
    main()
