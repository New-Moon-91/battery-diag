# -*- coding: utf-8 -*-
"""체크포인트 — 장시간 학습 중단/재개. 원자적 저장(임시파일 → rename)."""
from __future__ import annotations
import json, os, shutil, tempfile
import numpy as np, torch
from pathlib import Path


class Checkpoint:
    def __init__(self, root: str | Path, job_id: str):
        self.dir = Path(root)/job_id; self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir/'state.pt'; self.meta = self.dir/'meta.json'

    def exists(self): return self.path.exists()

    def save(self, **kw):
        tmp = self.path.with_suffix('.tmp')
        torch.save(kw, tmp); os.replace(tmp, self.path)
        m = {k: v for k, v in kw.items() if isinstance(v, (int, float, str, list, dict, bool))}
        self.meta.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))

    def load(self, map_location='cpu'):
        return torch.load(self.path, map_location=map_location, weights_only=False)

    def mark_done(self, result: dict):
        (self.dir/'DONE.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))

    def is_done(self): return (self.dir/'DONE.json').exists()

    def result(self):
        return json.loads((self.dir/'DONE.json').read_text())


def rng_state():
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=np.random.get_state())


def _as_cpu_byte(t):
    """torch.load(map_location='cuda') 가 RNG 상태까지 CUDA 로 올려버린다.
    set_rng_state 는 CPU ByteTensor 만 받으므로 되돌린다."""
    return t.cpu().to(torch.uint8) if torch.is_tensor(t) else t


def set_rng_state(s):
    torch.set_rng_state(_as_cpu_byte(s['torch']))
    if s.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_cpu_byte(x) for x in s['cuda']])
        except Exception:
            pass          # GPU 개수가 바뀐 경우 등 — 재현성만 잃고 진행에는 지장 없음
    np.random.set_state(s['numpy'])
