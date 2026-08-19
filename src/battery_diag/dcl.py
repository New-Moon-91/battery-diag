# -*- coding: utf-8 -*-
"""근사 정책반복 (Deep Controlled Learning 계열).

라운드 r:
  1) 현 정책 π_r 정확 평가 → h^{π_r}                (GPU SpMV)
  2) 1단계 전방탐색 개선 π_{r+1} = greedy(h^{π_r})   (GPU)
  3) 신경망을 π_{r+1} 에 분류학습(교사강요 CE)        (GPU, 배치)
  4) 신경망 탐욕정책을 정확 평가 → 갭 기록
퇴행 방지: 신경망이 개선라벨보다 나쁘면 다음 라운드는 개선라벨에서 출발.
최良 스냅샷 보존. 라운드마다 체크포인트 저장 → 중단 시 재개.
"""
from __future__ import annotations
import copy, time
import numpy as np, torch
from .net import PolicyNet
from .encode import state_tensors, actions_from_assign, labels_from_actions
from .ckpt import Checkpoint, rng_state, set_rng_state

# 헤드·인코더가 보는 배치 키. wcap 은 W-정식화에서만 존재한다.
BKEYS = ('U', 'mu', 'S', 'ms', 'ctx', 'allow_u', 'allow_s')


def _bk(tens):
    return BKEYS + (('wcap',) if 'wcap' in tens else ())


def fit_epochs(net, opt, tens, LU, LS, cap, w=None, epochs=40, bs=512, gen=None):
    nS = LU.shape[0]
    LUt = torch.as_tensor(LU, device=tens['U'].device)
    LSt = torch.as_tensor(LS, device=tens['U'].device)
    p = None if w is None else torch.as_tensor(0.9*w/w.sum() + 0.1/nS, device=tens['U'].device,
                                               dtype=torch.float32)
    for _ in range(epochs):
        idx = (torch.randperm(nS, device=LUt.device, generator=gen) if p is None
               else torch.multinomial(p, nS, replacement=True, generator=gen))
        for b0 in range(0, nS, bs):
            sel = idx[b0:b0+bs]
            batch = {k: tens[k][sel] for k in _bk(tens)}
            loss = net.ce_loss(batch, LUt[sel], LSt[sel], cap)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return float(loss.detach())


@torch.no_grad()
def greedy_policy(net, I, tens, bs=4096):
    nS = len(I.ST); AU, AS = [], []
    for b0 in range(0, nS, bs):
        sl = slice(b0, min(b0+bs, nS))
        batch = {k: tens[k][sl] for k in _bk(tens)}
        au, as_ = net.decode(batch, I.CAP)
        AU.append(au); AS.append(as_)
    return actions_from_assign(I, tens, np.concatenate(AU), np.concatenate(AS))


def run_dcl(I, solver, acts0, ckpt: Checkpoint, rounds=6, epochs=40, lr=5e-4,
            seed=0, device='cuda', log=print, gstar=None, carry=False):
    torch.manual_seed(seed); np.random.seed(seed)
    tens = state_tensors(I, device=device)
    net = PolicyNet(carry=carry).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    acts = np.asarray(acts0).copy(); r0 = 0
    best = (-1e18, None); hist = []
    if ckpt.exists():
        st = ckpt.load(map_location=device)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt'])
        acts = st['acts']; r0 = st['round']+1; hist = st['hist']
        best = (st['best_g'], st['best_net']); set_rng_state(st['rng'])
        log(f'  [resume] 라운드 {r0} 부터 재개 (최良 g={best[0]:,.0f})')
    gap = (lambda x: 100*(gstar-x)/gstar) if gstar else (lambda x: float('nan'))
    for r in range(r0, rounds):
        t0 = time.time()
        g_pi, h = solver.evaluate(acts)
        acts2 = solver.improve(h)
        g_imp, _ = solver.evaluate(acts2)
        w = solver.stationary(acts2)
        LU, LS = labels_from_actions(I, tens, acts2)
        for gp in opt.param_groups: gp['lr'] = lr if r < 2 else lr*0.4
        loss = fit_epochs(net, opt, tens, LU, LS, I.CAP, w=w, epochs=epochs)
        acts_net = greedy_policy(net, I, tens)
        g_net, _ = solver.evaluate(acts_net)
        if g_net > best[0]: best = (g_net, copy.deepcopy(net.state_dict()))
        hist.append(dict(round=r, g_pi=g_pi, g_improved=g_imp, g_net=g_net,
                         gap_improved=gap(g_imp), gap_net=gap(g_net), loss=loss,
                         agree=float((acts_net == acts2).mean()), sec=time.time()-t0))
        log(f'  r{r}: π {gap(g_pi):6.3f}% → 개선 {gap(g_imp):6.3f}% | 신경망 {gap(g_net):6.3f}%'
            f' (일치 {100*hist[-1]["agree"]:.1f}%, {hist[-1]["sec"]:.1f}s)')
        acts = acts_net if g_net >= g_imp - 1e-9 else acts2
        ckpt.save(net=net.state_dict(), opt=opt.state_dict(), acts=acts, round=r,
                  hist=hist, best_g=best[0], best_net=best[1], rng=rng_state())
    if best[1] is not None: net.load_state_dict(best[1])
    return net, best[0], hist
