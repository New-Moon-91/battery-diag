#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_rcr.py 에 --lam 플래그를 추가한다.

    python3 apply_lam_flag.py            # ~/battery-diag/run_rcr.py 를 제자리 수정
    python3 apply_lam_flag.py --check    # 적용 여부만 확인

왜 필요한가
  load = lam x NARR / CAP 이다. 정수 NARR/CAP 만으로는 부하 1.20 과 1.35 사이에
  NARR>=7 인 셀밖에 없는데, NARR=7 은 62GB 에서 OOM 으로 죽었다(2026-09-02 실측).
  lam 을 움직이면 **이미 검증된 크기의 셀**(NARR=3,4)로 그 구간을 덮을 수 있다.

기본값 0.9 에서는 jid 가 바뀌지 않으므로 기존 셀은 재실행되지 않는다.
"""
import re, sys, shutil
from pathlib import Path

p = Path.home()/'battery-diag'/'run_rcr.py'
if len(sys.argv) > 1 and sys.argv[1] != '--check':
    p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')

if '--lam' in s and 'a.lam' in s:
    print('이미 적용되어 있다. 아무것도 하지 않는다.'); sys.exit(0)
if '--check' in sys.argv:
    print('아직 적용되지 않았다.'); sys.exit(1)

orig = s

# 1) 인자 추가 — --Cp 줄 바로 뒤
m = re.search(r"^(ap\.add_argument\('--Cp'.*\n)", s, re.M)
if not m: sys.exit('실패: --Cp 인자 줄을 찾지 못했다. 민감도 패치가 먼저 적용돼야 한다.')
s = s[:m.end(1)] + (
    "ap.add_argument('--lam', type=float, default=0.9,\n"
    "                help='도착 기회당 도착 확률. load = lam x NARR / CAP')\n"
) + s[m.end(1):]

# 2) Config 에 전달
old = "cfg = Config(lam=0.9, NARR=NARR, Mcyc=CAP, W=W, prune=False, pu_all=True,"
new = "cfg = Config(lam=a.lam, NARR=NARR, Mcyc=CAP, W=W, prune=False, pu_all=True,"
if old not in s: sys.exit('실패: Config(lam=0.9, ...) 줄을 찾지 못했다.')
s = s.replace(old, new)

# 3) jid 접미사 — 기본값이면 붙이지 않는다
old2 = "        if a.Cp  != 500_000.: _tag += f'_Cp{a.Cp:g}'"
new2 = ("        if a.Cp  != 500_000.: _tag += f'_Cp{a.Cp:g}'\n"
        "        if a.lam != 0.9:      _tag += f'_lam{a.lam:g}'")
if old2 not in s: sys.exit('실패: jid 접미사 블록을 찾지 못했다.')
s = s.replace(old2, new2)

shutil.copy(p, str(p)+'.pre-lam.bak')
p.write_text(s, encoding='utf-8')
print(f'적용 완료: {p}')
print(f'백업: {p}.pre-lam.bak')
print()
print('확인:')
for ln in s.split('\n'):
    if '--lam' in ln or 'a.lam' in ln: print('   ', ln.strip())
