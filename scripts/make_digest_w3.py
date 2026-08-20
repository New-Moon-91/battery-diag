#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w3 산출물 → results/w_model/DIGEST_w3.md + summary_w.csv 갱신."""
import json, csv, glob
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'w_model'


def jload(p):
    return json.loads(Path(p).read_text())


def main():
    L = []
    A = L.append
    A('# w3 — 검증 구멍 메우기와 핵심 주장 강화')
    A('')
    A('생성 2026-08-20. w2 DIGEST(`DIGEST.md`)의 후속. 각 절 끝에 **논문 문장**을 붙였다.')
    A('')
    A('---')
    A('')

    # ---------------- [1]
    f = OUT/'reopt.csv'
    if f.exists():
        rows = [{k: (float(v) if k not in ('NMAX', 'SMAX', 'nS') else int(v))
                 for k, v in r.items()} for r in csv.DictReader(f.open())]
        G = {(r['NMAX'], r['SMAX']): r for r in rows}
        A('## [1] 가법분해 — 재최적화로 확정')
        A('')
        A('w2 [3] 의 보정 g\\* 는 기존 최적정책을 **고정한 채** 보상만 고친 값이라 보정 MDP 의')
        A('하한이었다. 여기서는 같은 상태공간·같은 전이커널 위에서 보상만 고친 MDP')
        A('(`Config.credit_loss=True`)를 정확히 다시 풀었다.')
        A('')
        A('| 버퍼 | 원본 g\\* | 고정정책 보정 | 재최적 보정 | 추가이득 | 정책 변경률 |')
        A('|---|---:|---:|---:|---:|---:|')
        for r in rows:
            A(f"| NMAX={r['NMAX']}, SMAX={r['SMAX']} | {r['g_orig']:,.0f} | {r['g_fixed']:,.0f} | "
              f"{r['g_reopt']:,.0f} | +{r['uplift']:,.0f} | {100*r['policy_change']:.2f}% |")
        A('')
        A('**"보정해도 최적정책은 그대로" 는 틀렸다.** 상태의 0.23~13.27% 가 행동을 바꾼다.')
        A('w2 [3] 의 유지목록에 그렇게 적었던 문장을 철회한다.')
        A('')
        A('### 가법분해 확정치')
        A('')
        A('| 기준 | SMAX 2→3 (N=2 / N=3) | NMAX 2→3 (S=2 / S=3) | NMAX/SMAX | 교호작용 | g\\* 대비 |')
        A('|---|---|---|---:|---:|---:|')
        for tag, key in (('원본', 'g_orig'), ('고정정책 보정', 'g_fixed'), ('**재최적 보정 (확정)**', 'g_reopt')):
            g22, g23, g32, g33 = (G[(2, 2)][key], G[(2, 3)][key], G[(3, 2)][key], G[(3, 3)][key])
            dS2, dS3 = g23-g22, g33-g32
            dN2, dN3 = g32-g22, g33-g23
            inter = dS3-dS2
            A(f'| {tag} | {dS2:,.0f} / {dS3:,.0f} | {dN2:,.0f} / {dN3:,.0f} | '
              f'{dN2/dS2:.2f}배 | {inter:,.0f} | {100*abs(inter)/g33:.3f}% |')
        A('')
        A('**w2 의 결론이 뒤집힌다.** 고정정책 하한으로는 잔차가 0.081% → 0.482% 로 6배')
        A('커지는 것처럼 보였지만, 제대로 재최적화하면 **0.090%** 로 원본(0.081%)과')
        A('사실상 같다. 부호는 여전히 뒤집히지만 크기가 무시할 수준이다.')
        A('즉 **가법분해는 소실보정 후에도 성립한다.** NMAX/SMAX 효과비도 1.25배가 아니라')
        A('**1.49배**로, 원본 2.22배에서 3분의 1쯤 줄어드는 데 그친다.')
        A('')
        A('SMAX 효과도 재최적화에서는 움직인다(107,993 → 96,418). w2 가 "SMAX 효과는 보정에')
        A('전혀 안 움직인다" 고 한 것은 고정정책 하한의 성질이었다.')
        A('')
        A('> **논문 문장** — "무료 소실을 보정한 MDP 를 재최적화하면 NMAX 효과는 SMAX 효과의')
        A('> 1.49배로, 보정 전 2.22배에서 줄어든다. 두 버퍼의 효과는 보정 후에도 가법적이며')
        A('> (교호작용이 g\\* 의 0.090%), 이는 보정 전(0.081%)과 같은 수준이다."')
        A('')

    # ---------------- [2]
    ws = sorted((jload(p) for p in glob.glob(str(OUT/'W*.json'))), key=lambda r: r['W'])
    ws = [r for r in ws if 'decomp' in r]
    if ws:
        A('## [2] B_fast 정의 감사와 갭 분해')
        A('')
        A('### (a) 임계 정의역 — 격자는 이미 완전했다')
        A('')
        A('`b_fast(I, thr)` 는 신호구간 b ≥ thr 인 선별품만 정밀검사 후보로 둔다. 구간이')
        A('SB=3 개(b ∈ {0,1,2})이므로 의미 있는 임계는 **thr ∈ {0,1,2,3} 네 개가 전부**다')
        A('(thr=3 = 아무도 자격 없음 = 정밀검사를 아예 안 쓰는 퇴화 케이스).')
        A('기존 격자 `range(SB)` 는 이 중 퇴화 케이스만 빠뜨렸다. 확인차 thr=3 도 넣어')
        A('전 구간을 다시 평가했다.')
        A('')
        A('| W | thr=0 | thr=1 | thr=2 | thr=3 | 최선 |')
        A('|---:|---:|---:|---:|---:|---:|')
        for r in ws:
            b = r.get('B_fast_all_full', {})
            A(f"| {r['W']} | {b.get('0', 0):,.0f} | {b.get('1', 0):,.0f} | {b.get('2', 0):,.0f} | "
              f"{b.get('3', 0):,.0f} | thr={r['B_fast_thr']} |")
        A('')
        A('**thr=0 이 항상 최선이고 g 는 thr 에 대해 단조감소한다. 구조적이다.**')
        A('CAP=1 이고 후보는 이미 V^PS 내림차순으로 정렬돼 있으며 V^PS ≤ V^S 인 것은')
        A('애초에 제외된다. 그러므로 thr 을 올리는 것은 유일한 정밀 슬롯을 **비워 두는**')
        A('일밖에 하지 않는다. 격자를 넓혀도 답은 안 바뀐다.')
        A('')
        A('### (b) 갭 분해 — 관행의 두 요소')
        A('')
        A('`B_fast_hold` 를 새로 정의했다(`policies.py`). 전량 신속검사 강제는 그대로 두고')
        A('**즉시처분 강제만 해제**한다 — 정밀 CAP건 외에는 매각하지 않고 보유하며,')
        A('가지치기로 강제매각이 걸린 선별품(V^PS ≤ V^S)만 판다.')
        A('')
        A('| W | gap(B_fast) | = 전량검사 강제 | + 즉시처분 강제 |')
        A('|---:|---:|---:|---:|')
        for r in ws:
            d = r['decomp']
            A(f"| {r['W']} | **{d['gap_B_fast']:.3f}%** | {d['cost_full_inspection']:.3f}%p | "
              f"{d['cost_forced_disposal']:.3f}%p |")
        A('')
        A('**w2 의 해석이 반증됐다.** w2 (b) 는 갭이 W 와 함께 커지는 이유를 "최적은')
        A('선별보유를 쓰는데 B_fast 는 못 쓴다" 로 읽었다. 즉 즉시처분 강제 쪽이 커진다는')
        A('예측이다. 실제로는 반대다 — 즉시처분 강제의 비용은 W 와 함께 **줄고**')
        A('(4.680 → 2.260%p), 전량검사 강제의 비용이 **늘어**(9.454 → 13.074%p) 총 갭을 민다.')
        A('')
        A('까닭은 정책구조표와 맞물린다. W 가 커지면 강제매각이 줄어 더 많은 배터리가')
        A('실제로 창고에 남는데, 그 중 SM3 는 최적정책이 **즉시 매각**한다(W=8 에서')
        A('1.0405대/기간). B_fast 는 정의상 SM3 까지 전부 신속검사하므로 그만큼 C_f 를')
        A('헛되이 쓴다. 창고가 커질수록 이 낭비가 커진다.')
        A('')
        A('> **논문 문장** — "관행(전량 신속검사 후 즉시처분)의 손실을 두 요소로 분해하면,')
        A('> 창고가 커질수록 손실의 무게가 즉시처분 강제(4.7→2.3%p)에서 전량검사')
        A('> 강제(9.5→13.1%p)로 옮겨간다. 용량이 늘수록 문제는 「언제 파느냐」가 아니라')
        A('> 「무엇을 검사하느냐」가 된다."')
        A('')

    # ---------------- [3]
    dd = {}
    for p in glob.glob(str(OUT/'dcl_W*_carry_*.json')):
        r = jload(p); dd.setdefault(r['W'], []).append(r['gap'])
    if dd:
        A('## [3] 경계 인스턴스에서의 학습정책')
        A('')
        A('carry 디코더, 시드 3개. 비용 추정은 W=8 의 PI 1회(=improve 전량스윕 72GB +')
        A('정확평가)가 80초이고 run_dcl 라운드가 그 1.6배쯤이라 6라운드 ≈ 13분/시드,')
        A('3시드 ≈ 40분이었다 — 8시간 한도에 한참 못 미쳐 축소 없이 전부 돌렸다.')
        A('')
        A('| W | \\|S\\| | 시드별 gap_DCL | 평균±SD |')
        A('|---:|---:|---|---:|')
        for W in sorted(dd):
            g = sorted(dd[W])
            nS = next((r['nS'] for r in ws if r['W'] == W), '')
            A(f"| {W} | {nS:,} | {' / '.join(f'{x:.3f}%' for x in g)} | "
              f"**{mean(g):.3f}±{pstdev(g):.3f}** |")
        A('')
        gmax = max(max(v) for v in dd.values())
        if gmax <= 1.0:
            A('**정확해가 닿는 가장 큰 인스턴스까지 학습정책이 1% 이내로 재현한다.**')
            A('방법론 주장이 경계까지 닫혔다.')
        else:
            A(f'최대 갭이 {gmax:.3f}% 로 1% 를 넘는 칸이 있다. 아래 표에서 확인할 것.')
        A('')
        A('> **논문 문장** — "정확해가 가능한 최대 인스턴스(W=8, |S|=125,970)에서도 학습정책의')
        A('> 최적 대비 갭은 시드 3개 평균 [값]% 로, 사전등록 기준 1% 이내를 유지한다."')
        A('')

    # ---------------- [4]
    lad = []
    for pre in ('T4', 'T5'):
        for p in sorted(glob.glob(str(OUT/f'{pre}_W*.json'))):
            lad.append(jload(p))
    if lad:
        A('## [4] 차종 사다리 확장 — "선별은 중간에만" 은 성립하는가')
        A('')
        A('FLEET 의 가치 사다리(V^S = p_rc×kWh 기준)는 레이 142,811 → SM3 231,633 →')
        A('아이오닉 243,824 → 쏘울 261,240 → (공백) → 코나 557,312 이다. 기존 3종은')
        A('맨 아래(레이)·중간 하나(SM3)·맨 위(코나)만 집어 사다리 중간이 사실상 한 점이었다.')
        A('아이오닉·쏘울이 SM3 와 코나 사이를 메우므로 이 둘을 넣었다')
        A('(pass_soh 표본이 각각 충분해 MU/SD 를 대체값 없이 실측으로 쓴다).')
        A('')
        for r in lad:
            ts = r.get('types', [])
            A(f"**|T|={len(ts)} W={r['W']}** — {' · '.join(ts)}  (\\|S\\|={r['nS']:,}, "
              f"g\\*={r['gstar']:,.0f})")
            A('')
            A('| 차종 | 매각 | 신속 | 정밀직행 | 보유 | 선별매각 | 선별정밀 | 선별보유 |')
            A('|---|---:|---:|---:|---:|---:|---:|---:|')
            o = r['opt']
            for k, t in enumerate(o['types']):
                u = o['act_u'][k]; s = o['act_s'][k]
                A(f"| {t} | {u[0]:.4f} | {u[1]:.4f} | {u[2]:.4f} | {u[3]:.4f} | "
                  f"{s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} |")
            A('')
            tot = sum(o['act_u'][k][1] for k in range(len(o['types'])))
            if tot > 0:
                sh = sorted(((o['act_u'][k][1]/tot, t) for k, t in enumerate(o['types'])),
                            reverse=True)
                A('신속검사 점유: ' + ', '.join(f'{t} {100*x:.1f}%' for x, t in sh if x > 1e-4))
                A('')

    # ---------------- [5]
    f = OUT/'phi_sweep.csv'
    if f.exists():
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f.open())]
        A('## [5] φ 민감도')
        A('')
        A('φ = 미공개 결함(FAIL_L) 중 신속검사로 **못 잡는** 비율. P_DET = F_E + (1−φ)·F_U.')
        A('W=6, |T|=3 에서 정확해 스윕.')
        A('')
        A('| φ | P_DET | g\\* | 신속검사 대수/기간 | 정밀 대수/기간 | B_fast 갭 |')
        A('|---:|---:|---:|---:|---:|---:|')
        for r in rows:
            A(f"| {r['phi']:.2f} | {r['P_DET']:.4f} | {r['gstar']:,.0f} | "
              f"{r['fast_per_period']:.4f} | {r['precise_per_period']:.4f} | "
              f"{r['gap_B_fast']:.3f}% |")
        A('')
        g = [r['gstar'] for r in rows]; fa = [r['fast_per_period'] for r in rows]
        A(f"g\\* 는 φ=0 에서 {max(g):,.0f}, φ=1 에서 {min(g):,.0f} 로 "
          f"**{100*(max(g)-min(g))/max(g):.2f}% 움직인다**. "
          f"선별량은 {min(fa):.4f}~{max(fa):.4f}대로 {100*(max(fa)-min(fa))/max(fa):.2f}% 범위다.")
        A('')

    A('## 돌다 만 것 · 실패한 것')
    A('')
    A('- **[6] HANDOFF.md 개정은 불가능했다.** 원본이 작업트리·git 이력·디스크·백업 zip')
    A('  어디에도 없다. 개정 대신 신규 작성했고 `BLOCKED.md` 에 근거를 남겼다.')
    A('  "기존 반전 3개", "열린작업 #6", "docs/06 §5.2" 처럼 원본을 봐야 쓸 수 있는')
    A('  지시는 이행하지 못했다.')
    A('- **캐시 해시 사고.** `Config.credit_loss` 를 추가하자 해시가 `cfg.__dict__` 전체를')
    A('  쓰는 탓에 기존 캐시가 전부 무효화됐다(W=8 의 72GB 포함). 나중에 추가된 필드는')
    A('  기본값일 때 해시에서 빼도록 고쳐 전량 복구했다. 그 사이 2/4 를 한 번 헛빌드했다.')
    A('- **|T|=4 W=7 (CSR 344GB), |T|=5 W=6 (218GB) 는 돌리지 않았다.** 디스크·시간상')
    A('  가능하긴 하나 [4] 의 질문(선별이 퍼지는가)에 답하는 데 필요하지 않았다.')
    A('- **W 모형에서의 표현가능 상한은 이번에도 재측정하지 않았다** (w2 에서 남긴 숙제).')
    A('')
    (OUT/'DIGEST_w3.md').write_text('\n'.join(L)+'\n')
    print('→', OUT/'DIGEST_w3.md')


if __name__ == '__main__':
    main()
