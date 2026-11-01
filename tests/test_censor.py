# -*- coding: utf-8 -*-
"""유찰 보정 가격(PriceCensored) 회귀 — Omega 원고의 E[max(R,W)] 를 반영한 판."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import fleet_w5, SEL_W5
from battery_diag.instance import Instance, PriceW5, PriceCensored, load_price, Config

DATA = ROOT/'data'


def _types():
    F = fleet_w5(DATA); sel = list(F); tot = sum(F[t][0] for t in sel)
    return {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in sel}


def test_dispatch_and_identity_without_censor():
    p0 = load_price(DATA/'params_w5.json'); p1 = load_price(DATA/'params_w5c.json')
    assert type(p0) is PriceW5 and type(p1) is PriceCensored
    p1.censor = False
    for s, cap in ((0.88, 64.0), (0.68, 26.6), (0.95, 58.8)):
        assert abs(p1.Erev(s, cap) - p0.Erev(s, cap)) < 1e-6
        assert abs(p1.VS(cap) - p0.VS(cap)) < 1e-6


def test_ratio_bounds_and_direction():
    """보정비는 (0,1], 유찰확률은 (0,1), 큰 팩일수록 유찰이 적고 보정이 작다."""
    p = load_price(DATA/'params_w5c.json')
    for s, cap in ((0.88, 64.0), (0.68, 26.6), (0.78, 16.4), (0.85, 6.1)):
        r = p.ratio(s, cap); G = p.G(s, cap)
        assert 0 < r <= 1 and 0 < G < 1
    assert p.G(0.88, 64.0) < p.G(0.85, 28.0) < p.G(0.78, 16.4)
    assert p.ratio(0.88, 64.0) > p.ratio(0.85, 28.0)
    # 유형별 보정비는 0.70~0.85 (DIGEST 수치)
    assert 0.70 < p.ratio(0.78, 16.4) < 0.75 and 0.82 < p.ratio(0.88, 64.0) < 0.85


def test_region_shift_matches_empirical_ladder():
    """보정 후 아이오닉_28 은 R1, SM3_35.9 는 R3 — 실측 사다리(types_w5.json)와 일치한다."""
    types = _types(); cfg = Config(Mcyc=1, NARR=4, W=4)
    I0 = Instance.lite(types, load_price(DATA/'params_w5.json'), cfg)
    I1 = Instance.lite(types, load_price(DATA/'params_w5c.json'), cfg)
    assert I0.REG['아이오닉_28.0'] == 'R3' and I1.REG['아이오닉_28.0'] == 'R1'
    assert I0.REG['SM3_35.9'] == 'R2' and I1.REG['SM3_35.9'] == 'R3'
    assert I1.REG_EMP['아이오닉_28.0'] == 'R1' and I1.REG_EMP['SM3_35.9'] == 'R3'
    # 보정은 재사용 수익만 낮춘다
    for t in types:
        assert I1.VPU[t] <= I0.VPU[t] + 1e-6 and abs(I1.VS[t] - I0.VS[t]) < 1e-6


def test_cache_key_distinguishes_censored():
    p0 = load_price(DATA/'params_w5.json'); p1 = load_price(DATA/'params_w5c.json')
    assert p0.key() != p1.key() and 'censor' in p1.key()
