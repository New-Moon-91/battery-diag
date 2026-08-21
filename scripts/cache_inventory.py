#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w6 [1] — 캐시 재고조사.  무엇이 어느 회차 것인지 해시를 다시 계산해 식별한다.

    python scripts/cache_inventory.py [--cache DIR]

캐시 디렉터리 이름은 md5 라 눈으로는 구분이 안 된다. 그래서 **현행 코드로 키를
다시 계산**해 w5(현행) 것을 찾아내고, 나머지는 meta.json 의 nS·생성시각으로
분류한다. 지우면 안 되는 것을 지우지 않기 위한 절차다.

분류
  KEEP-w5   현행 인스턴스(차종×용량 4유형)의 캐시 — [4] 전이 실험이 쓴다
  OLD-W     W-정식화지만 현행 키가 아닌 것 — w2~w4 (유형이 `차종` 기준)
  OLD-NS    구 정식화(NMAX/SMAX) — v5 이전
  UNKNOWN   meta.json 이 없거나 판독 불가
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

# W-정식화 |T|=4 의 상태수 — C(W+16,16)
NS_W_T4 = {4: 4845, 5: 20349, 6: 74613, 7: 245157, 8: 735471}


def dsize(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t


def current_keys(Ws=(3, 4, 5, 6, 7, 8)):
    """현행 w5 인스턴스가 쓰는 캐시 키 (stream_*, trans_*) 를 모두 계산한다."""
    from battery_diag.data import SEL_W5 as SEL, fleet_w5, load_cached
    from battery_diag.instance import Instance, PriceW5, Config
    from battery_diag import streambuild as sb, build as bd
    _, FS = load_cached(str(ROOT/'data'))
    F = fleet_w5(ROOT/'data', sel=SEL)
    tot = sum(F[t][0] for t in SEL)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in SEL}
    price = PriceW5.from_json(ROOT/'data'/'params_w5.json')
    out = {}
    for W in Ws:
        I = Instance(types, price,
                     Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                            F_E=FS['F_E'], F_U=FS['F_U']))
        out[sb._key(I, ','.join(SEL))] = f'w5 W={W} (stream)'
        out[bd._key(I, ','.join(SEL))] = f'w5 W={W} (in-RAM)'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.environ.get('BATDIAG_CACHE',
                                                      '/home/user/batdiag-cache'))
    ap.add_argument('--purge', nargs='*', default=None, metavar='CLS',
                    help='해당 분류를 삭제한다 (OLD-NS / OLD-W / UNKNOWN). '
                         'KEEP-w5 는 명시해도 거부한다 — [4] 전이 실험이 쓴다.')
    ap.add_argument('--record', default='', help='삭제 내역 JSON 경로')
    a = ap.parse_args()
    root = Path(a.cache)
    keys = current_keys()
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() and p.suffix != '.npz':
            continue
        name = p.name
        sz = p.stat().st_size if p.is_file() else dsize(p)
        meta = {}
        mj = p/'meta.json'
        if p.is_dir() and mj.exists():
            try: meta = json.loads(mj.read_text())
            except Exception: meta = {}
        nS = meta.get('nS'); built = meta.get('built')
        stem = p.stem if not p.is_dir() else name     # trans_xxx.npz → trans_xxx
        if name in keys or stem in keys:
            cls, why = 'KEEP-w5', keys.get(name) or keys[stem]
        elif not p.is_dir():
            cls, why = 'OLD-NS', 'in-RAM npz (구 빌드 산출물)'
        elif nS is None:
            cls, why = 'UNKNOWN', 'meta.json 없음/판독불가'
        elif nS in NS_W_T4.values():
            W = [k for k, v in NS_W_T4.items() if v == nS][0]
            cls, why = 'OLD-W', f'W-정식화 W={W} 이지만 현행 키 아님 (w2~w4)'
        else:
            cls, why = 'OLD-NS', f'구 정식화 추정 (nS={nS})'
        rows.append(dict(name=name, cls=cls, gb=sz/1e9, nS=nS,
                         built=(time.strftime('%m-%d %H:%M', time.localtime(built))
                                if built else ''), why=why))
    rows.sort(key=lambda r: (-r['gb']))
    tot = {}
    print(f"{'분류':<9}{'GB':>8}  {'nS':>8}  {'생성':<12} 이름 / 근거")
    for r in rows:
        tot[r['cls']] = tot.get(r['cls'], 0) + r['gb']
        print(f"{r['cls']:<9}{r['gb']:8.1f}  {str(r['nS'] or ''):>8}  {r['built']:<12} "
              f"{r['name'][:28]}  {r['why']}")
    print('\n합계 (GB)')
    for k in sorted(tot, key=lambda k: -tot[k]):
        print(f'  {k:<9} {tot[k]:8.1f}')
    print(f"  {'전체':<9} {sum(tot.values()):8.1f}")
    print('\n삭제 우선순위: OLD-NS → OLD-W → (부족하면) KEEP-w5')
    print('KEEP-w5 재생성 비용: W=4 8초 / W=5 79초 / W=6 587초 = 총 11분')

    if a.purge is None:
        return
    want = {c.upper() for c in a.purge}
    if 'KEEP-W5' in want or 'KEEP-w5' in {c for c in a.purge}:
        print('\n거부: KEEP-w5 는 이 스크립트로 지우지 않는다.')
        want.discard('KEEP-W5')
    import shutil
    avail = lambda: shutil.disk_usage(root).free/1e9
    before = avail()
    print(f'\n=== 삭제 시작 (여유 {before:,.1f}GB) ===')
    rec, freed, cnt = [], 0.0, 0
    for r in rows:
        if r['cls'] not in want:
            continue
        p = root/r['name']
        if not p.exists():
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        rec.append({k: r[k] for k in ('name', 'cls', 'gb', 'nS', 'built', 'why')})
        freed += r['gb']; cnt += 1
        if cnt % 10 == 0:
            print(f'  ... {cnt}개 / {freed:,.1f}GB  (여유 {avail():,.1f}GB)', flush=True)
    after = avail()
    print(f'=== 삭제 완료: {cnt}개 / 목록기준 {freed:,.1f}GB ===')
    print(f'    여유 {before:,.1f}GB → {after:,.1f}GB  (실확보 {after-before:,.1f}GB)')
    if a.record:
        Path(a.record).write_text(json.dumps(
            dict(deleted=rec, count=cnt, listed_gb=round(freed, 1),
                 avail_before_gb=round(before, 1), avail_after_gb=round(after, 1),
                 freed_gb=round(after-before, 1)), ensure_ascii=False, indent=1),
            encoding='utf-8')
        print(f'    내역 → {a.record}')


if __name__ == '__main__':
    main()
