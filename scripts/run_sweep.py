#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""실험 그리드 실행. 예:
   python scripts/run_sweep.py configs/sweep_load.yaml
중단 후 같은 명령을 재실행하면 미완료 잡부터 이어서 돈다."""
import sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from battery_diag.runner import expand_grid, run_grid

cfgpath = Path(sys.argv[1]); spec = yaml.safe_load(cfgpath.read_text())
cfgs = list(expand_grid(spec.get('base', {}), spec['grid']))
run_grid(cfgs, str(Path(__file__).resolve().parent/'run_one.py'),
         spec.get('out', f'/home/data/batdiag/results/{cfgpath.stem}'),
         gpus=tuple(spec.get('gpus', [0, 1])), per_gpu=spec.get('per_gpu', 2))
