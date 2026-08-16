#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""실패한 잡의 체크포인트 제거 — 코드가 바뀌어 이전 체크포인트가 호환되지 않을 때 사용.
  python scripts/clean_ckpt.py /home/data/batdiag/results/sweep_selectivity        # 미완료만
  python scripts/clean_ckpt.py /home/data/batdiag/results/sweep_selectivity --all  # 전부"""
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1]); all_ = '--all' in sys.argv
n = 0
for d in sorted(root.glob('*/')):
    if not d.is_dir(): continue
    done = (d/'DONE.json').exists()
    if done and not all_: continue
    shutil.rmtree(d); n += 1; print('removed', d.name)
print(f'{n}개 제거')
