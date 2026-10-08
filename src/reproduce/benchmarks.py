"""Matched runtime measurements with the archived optimized kernels.

Every timed implementation first passes its forward and gradient checks.
CRISP is trained in mean-field mode and sampled during inference. All models
use cross-entropy and AdamW for these controlled runtime measurements.
GPU Triton and TPU XLA implementations run only on compatible devices.
No hardware result is inferred from CPU timing.
"""
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models.crisp import Config, DendStochAudioNet, DendriticSSM as DendRef, _shift
from .models.crisp import parallel_scan as hillis_scan
from .models.lif import LIFNet, LIFLayer as LIFLayerRef, LIReadout as LIReadoutRef, SurrogateSpike
from .paths import resolve_path

XLA = CUDA = TRITON_CPU = False
device = torch.device("cpu")
DEVICE_LABEL = XLA_PREC_REQ = ""
DEVICE_KIND = "cpu"
SPSN = {"ok": False, "reason": "source checkout not loaded"}
SPSN_URL = "https://github.com/NECOTIS/Stochastic-Parallelizable-Spiking-Neuron-SPSN.git"
SPSN_COMMIT = "c46d16931b1736208ced87974587e0d4c9a76634"
SPSN_MD5 = {
    "neurons/spsn.py": "8020d8d3bbcacbda5c2bc98f49110726",
    "neurons/base.py": "2948788b8f3fe05c63b83a5a1cee8293",
    "network.py": "fccd917d89c2f39d68cab692fcaa1904",
    "neurons/lif.py": "3f46b394e45f46b8612f87bcdea85244",
    "neurons/__init__.py": "416aab08dfa846f473129e89a7625bbc",
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed_all(seed)
    if XLA:
        xm.set_rng_state(int(seed))


def chunked_scan(a, x, chunk=32):
    """Original bounded-memory Hillis scan with sequential chunk carries."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    batch, length = x.shape[:2]
    if length <= chunk:
        return hillis_scan(a, x)
    carry = x.new_zeros((batch,) + tuple(x.shape[2:]))
    output = []
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        local = hillis_scan(a, x[:, start:stop])
        powers = torch.stack([a ** (t + 1) for t in range(stop - start)], 0)
        local = local + powers.unsqueeze(0) * carry.unsqueeze(1)
        output.append(local)
        carry = local[:, -1]
    return torch.cat(output, 1)

class ConvDendriticSSM(DendRef):
    """Same parameters and same map as the reference ct_exact dendrite. The dendrite is LTI, so its K ZOH modes collapse
    into one causal filter per channel, h(tau) = sum_k mix_k Bd_k a_k^tau, with a_k = exp(lam_k dt), Bd_k = expm1(lam_k dt)/lam_k.
    conv_mode: 'fft' | 'toeplitz' | 'chunkmm' (set per instance)."""
    conv_mode = 'fft'
    chunk_L = 64

    def _decay(self):
        lam = -torch.exp(self.raw_lam)
        ld = lam * self.dt
        return (ld, torch.expm1(ld) / lam)

    def kernel(self, T):
        ld, Bd = self._decay()
        tau = torch.arange(T, device=ld.device, dtype=ld.dtype)
        return ((self.mix * Bd).unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * tau)).sum(1)

    def forward(self, x):
        if not self.ct_exact:
            raise RuntimeError('convolution mode implemented for the ct_exact dendrite only')
        B, T, C = x.shape
        if self.conv_mode == 'fft':
            h = self.kernel(T)
            n = 2 * T
            Xf = torch.fft.rfft(x.transpose(1, 2), n=n)
            y = torch.fft.irfft(Xf * torch.fft.rfft(h, n=n), n=n)[..., :T]
            return y.transpose(1, 2)
        if self.conv_mode == 'toeplitz':
            h = self.kernel(T)
            i = torch.arange(T, device=x.device)
            d = i[:, None] - i[None, :]
            Hm = torch.where(d >= 0, h[:, d.clamp(min=0)], torch.zeros((), device=x.device, dtype=x.dtype))
            return torch.bmm(Hm, x.permute(2, 1, 0)).permute(2, 1, 0)
        if self.conv_mode == 'chunkmm':
            L = self.chunk_L
            N = (T + L - 1) // L
            Tp = N * L
            K = self.K
            xp = F.pad(x, (0, 0, 0, Tp - T)) if Tp > T else x
            ld, Bd = self._decay()
            i = torch.arange(L, device=x.device, dtype=x.dtype)
            g = ((self.mix * Bd).unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * i)).sum(1)
            ii = torch.arange(L, device=x.device)
            dd = ii[:, None] - ii[None, :]
            G = torch.where(dd >= 0, g[:, dd.clamp(min=0)], torch.zeros((), device=x.device, dtype=x.dtype))
            Xc = xp.reshape(B, N, L, C).permute(3, 2, 0, 1).reshape(C, L, B * N)
            Vloc = torch.bmm(G, Xc)
            W = Bd.unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * (L - 1 - i))
            E = torch.bmm(W, Xc).reshape(C, K, B, N).permute(2, 3, 0, 1)
            H = hillis_scan(torch.exp(ld * L), E)
            Hprev = torch.cat([torch.zeros_like(H[:, :1]), H[:, :-1]], 1)
            Q = self.mix.unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * (i + 1))
            Vcar = torch.bmm(Q.transpose(1, 2), Hprev.permute(2, 3, 0, 1).reshape(C, K, B * N))
            V = (Vloc + Vcar).reshape(C, L, B, N).permute(2, 3, 1, 0).reshape(B, Tp, C)
            return V[:, :T]
        raise ValueError(self.conv_mode)

def apply_crisp_impl(net, impl):
    if impl == 'crisp_hillis':
        return net
    if impl == 'crisp_chunked':
        net.set_scan(chunked_scan)
        return net
    if impl == 'crisp_fused':
        for m in net.modules():
            if type(m) is DendRef:
                m.__class__ = FusedDendriticSSM
        return net
    mode = {'crisp_fft': 'fft', 'crisp_toeplitz': 'toeplitz'}.get(impl, 'chunkmm')
    for m in net.modules():
        if type(m) is DendRef:
            m.__class__ = ConvDendriticSSM
            m.conv_mode = mode
            m.chunk_L = cfg.chunkmm_L
    return net

def build_crisp(impl, seed=0):
    set_all_seeds(seed)
    return apply_crisp_impl(DendStochAudioNet(CCFG), impl)

KP_MODES = 16

def chunk_params(raw_lam, mix, dt, L, KP=KP_MODES):
    """(h [D,L], P^T [D,KP,L], W^T [D,L,KP], Lam [D,KP]) as differentiable functions of the reference parameters."""
    lam = -torch.exp(raw_lam)
    ld = lam * dt
    Bd = torch.expm1(ld) / lam
    D, K = raw_lam.shape
    n = torch.arange(L, device=raw_lam.device, dtype=raw_lam.dtype)
    h = ((mix * Bd).unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * n)).sum(1)
    PT = mix.unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * (n + 1))
    WT = (Bd.unsqueeze(-1) * torch.exp(ld.unsqueeze(-1) * (L - 1 - n))).transpose(1, 2)
    Lam = torch.exp(ld * L)
    pad = KP - K
    if pad > 0:
        PT = F.pad(PT, (0, 0, 0, pad))
        WT = F.pad(WT, (0, pad))
        Lam = F.pad(Lam, (0, pad))
    return (h.contiguous(), PT.contiguous(), WT.contiguous(), Lam.contiguous())

def chunk_conv_ref(xT, h, PT, WT, Lam, L):
    """Pure-PyTorch chunked convolution, any dtype / device (float64 gate and the kernel's reference). xT: [B, D, N*L]."""
    B, D, Tp = xT.shape
    N = Tp // L
    i = torch.arange(L, device=xT.device)
    dd = i[None, :] - i[:, None]
    HT = torch.where(dd >= 0, h[:, dd.clamp(min=0)], torch.zeros((), dtype=h.dtype, device=h.device))
    S = xT.new_zeros(B, D, Lam.shape[1])
    Y = []
    for c in range(N):
        U = xT[:, :, c * L:(c + 1) * L]
        Y.append(torch.einsum('bdj,dji->bdi', U, HT) + torch.einsum('bdk,dki->bdi', S, PT))
        S = S * Lam.unsqueeze(0) + torch.einsum('bdj,djk->bdk', U, WT)
    return torch.cat(Y, 2)

_DIAG_IDX = {}

def _diag_index(L, device):
    k = (L, str(device))
    if k not in _DIAG_IDX:
        i = torch.arange(L, device=device)
        n = i[:, None] - i[None, :]
        _DIAG_IDX[k] = (n.clamp(min=0).reshape(-1), (n >= 0).reshape(-1))
    return _DIAG_IDX[k]

CC = {'ok': False}

CC_L, CC_BB = (64, 32)

def _cc_block(B):
    return CC_BB if B >= CC_BB else max(16, CC['tri'].next_power_of_2(B))

def cc_forward_kernel(xT, h, PT, WT, Lam, L, prec):
    B, D, Tp = xT.shape
    N = Tp // L
    KP = Lam.shape[1]
    BB = _cc_block(B)
    nb = (B + BB - 1) // BB
    Y = torch.empty_like(xT)
    Sin = torch.empty(B, N, D, KP, device=xT.device, dtype=xT.dtype)
    CC['fwd'][D, nb](xT, h, PT, WT, Lam, Y, Sin, B, D, N, xT.stride(0), xT.stride(1), L=L, KP=KP, BB=BB, PREC=prec, num_warps=4)
    return (Y, Sin)

def cc_backward_kernel(dY, xT, h, PT, WT, Lam, Sin, L, prec):
    B, D, Tp = xT.shape
    N = Tp // L
    KP = Lam.shape[1]
    BB = _cc_block(B)
    nb = (B + BB - 1) // BB
    dY = dY.contiguous()
    dxT = torch.empty_like(xT)
    dSo = torch.empty_like(Sin)
    dH = torch.empty(nb, D, L, L, device=xT.device, dtype=xT.dtype)
    dPT = torch.empty(nb, D, KP, L, device=xT.device, dtype=xT.dtype)
    dWT = torch.empty(nb, D, L, KP, device=xT.device, dtype=xT.dtype)
    dLam = torch.empty(nb, D, KP, device=xT.device, dtype=xT.dtype)
    CC['bwd_chain'][D, nb](dY, h, PT, WT, Lam, dxT, dSo, B, D, N, xT.stride(0), xT.stride(1), L=L, KP=KP, BB=BB, PREC=prec, num_warps=4)
    CC['bwd_params'][D, nb](dY, xT, Sin, dSo, dH, dPT, dWT, dLam, B, D, N, xT.stride(0), xT.stride(1), L=L, KP=KP, BB=BB, PREC=prec, num_warps=4)
    dHs = dH.sum(0).reshape(D, L * L)
    idx, keep = _diag_index(L, xT.device)
    dh = torch.zeros(D, L, device=xT.device, dtype=xT.dtype).scatter_add_(1, idx.expand(D, -1), torch.where(keep, dHs, torch.zeros_like(dHs)))
    return (dxT, dh, dPT.sum(0), dWT.sum(0), dLam.sum(0))

class ChunkConvFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xT, h, PT, WT, Lam, L, prec):
        Y, Sin = cc_forward_kernel(xT, h, PT, WT, Lam, L, prec)
        ctx.save_for_backward(xT, h, PT, WT, Lam, Sin)
        ctx.L = L
        ctx.prec = prec
        return Y

    @staticmethod
    def backward(ctx, dY):
        xT, h, PT, WT, Lam, Sin = ctx.saved_tensors
        dxT, dh, dPT, dWT, dLam = cc_backward_kernel(dY, xT, h, PT, WT, Lam, Sin, ctx.L, ctx.prec)
        return (dxT, dh, dPT, dWT, dLam, None, None)

class FusedDendriticSSM(DendRef):
    """Same parameters and map as the reference ct_exact dendrite; forward = fused chunked convolution (Triton) on CUDA
    float32, the pure-PyTorch chunked fallback elsewhere (CPU / float64 gates)."""
    chunk_L = CC_L
    prec = 0

    def forward(self, x):
        if not self.ct_exact:
            raise RuntimeError('fused mode implemented for the ct_exact dendrite only')
        B, T, C = x.shape
        L = self.chunk_L
        N = (T + L - 1) // L
        Tp = N * L
        h, PT, WT, Lam = chunk_params(self.raw_lam, self.mix, self.dt, L)
        xT = x.transpose(1, 2)
        if Tp > T:
            xT = F.pad(xT, (0, Tp - T))
        xT = xT.contiguous()
        use_kernel = CC.get('ok') and x.dtype == torch.float32 and (x.is_cuda or TRITON_CPU)
        Y = ChunkConvFn.apply(xT, h, PT, WT, Lam, L, self.prec) if use_kernel else chunk_conv_ref(xT, h, PT, WT, Lam, L)
        return Y[:, :, :T].transpose(1, 2)

def define_crisp_kernels(triton, tl):

    @triton.jit
    def _cc_fwd(x_ptr, h_ptr, pt_ptr, wt_ptr, lam_ptr, y_ptr, s_ptr, B, D, N, sxb, sxd, L: tl.constexpr, KP: tl.constexpr, BB: tl.constexpr, PREC: tl.constexpr):
        d = tl.program_id(0)
        pb = tl.program_id(1)
        rb = pb * BB + tl.arange(0, BB)
        mb = rb < B
        i = tl.arange(0, L)
        j = tl.arange(0, L)
        k = tl.arange(0, KP)
        dd = i[None, :] - j[:, None]
        HT = tl.load(h_ptr + d * L + dd, mask=dd >= 0, other=0.0)
        PT = tl.load(pt_ptr + d * KP * L + k[:, None] * L + i[None, :])
        WT = tl.load(wt_ptr + d * L * KP + j[:, None] * KP + k[None, :])
        Lam = tl.load(lam_ptr + d * KP + k)
        xb = x_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        yb = y_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        sb = s_ptr + rb[:, None].to(tl.int64) * N * (D * KP) + d * KP + k[None, :]
        S = tl.zeros([BB, KP], dtype=tl.float32)
        for c in range(N):
            tl.store(sb + c * (D * KP), S, mask=mb[:, None])
            U = tl.load(xb + c * L + j[None, :], mask=mb[:, None], other=0.0)
            if PREC == 1:
                Y = tl.dot(U, HT, input_precision='ieee') + tl.dot(S, PT, input_precision='ieee')
                S = S * Lam[None, :] + tl.dot(U, WT, input_precision='ieee')
            else:
                Y = tl.dot(U, HT) + tl.dot(S, PT)
                S = S * Lam[None, :] + tl.dot(U, WT)
            tl.store(yb + c * L + i[None, :], Y, mask=mb[:, None])

    @triton.jit
    def _cc_bwd_chain(dy_ptr, h_ptr, pt_ptr, wt_ptr, lam_ptr, dx_ptr, ds_ptr, B, D, N, sxb, sxd, L: tl.constexpr, KP: tl.constexpr, BB: tl.constexpr, PREC: tl.constexpr):
        """reverse walk: dU_c = dY_c H + dS_c W; dS_{c-1} = dY_c P + dS_c Lam. Stores dU and dS_c (adjoint of the state
        leaving chunk c) for the parameter kernel."""
        d = tl.program_id(0)
        pb = tl.program_id(1)
        rb = pb * BB + tl.arange(0, BB)
        mb = rb < B
        i = tl.arange(0, L)
        j = tl.arange(0, L)
        k = tl.arange(0, KP)
        dd = i[:, None] - j[None, :]
        H = tl.load(h_ptr + d * L + dd, mask=dd >= 0, other=0.0)
        P = tl.load(pt_ptr + d * KP * L + k[None, :] * L + i[:, None])
        W = tl.load(wt_ptr + d * L * KP + j[None, :] * KP + k[:, None])
        Lam = tl.load(lam_ptr + d * KP + k)
        dyb = dy_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        dxb = dx_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        dsb = ds_ptr + rb[:, None].to(tl.int64) * N * (D * KP) + d * KP + k[None, :]
        dS = tl.zeros([BB, KP], dtype=tl.float32)
        for cc in range(N):
            c = N - 1 - cc
            tl.store(dsb + c * (D * KP), dS, mask=mb[:, None])
            dY = tl.load(dyb + c * L + i[None, :], mask=mb[:, None], other=0.0)
            if PREC == 1:
                dU = tl.dot(dY, H, input_precision='ieee') + tl.dot(dS, W, input_precision='ieee')
                dS = tl.dot(dY, P, input_precision='ieee') + dS * Lam[None, :]
            else:
                dU = tl.dot(dY, H) + tl.dot(dS, W)
                dS = tl.dot(dY, P) + dS * Lam[None, :]
            tl.store(dxb + c * L + j[None, :], dU, mask=mb[:, None])

    @triton.jit
    def _cc_bwd_params(dy_ptr, x_ptr, s_ptr, ds_ptr, dh_ptr, dpt_ptr, dwt_ptr, dlam_ptr, B, D, N, sxb, sxd, L: tl.constexpr, KP: tl.constexpr, BB: tl.constexpr, PREC: tl.constexpr):
        """parameter gradients, summed over the chunks of this batch block (order free): dH += dY_c^T U_c,
        dP^T += S_{c-1}^T dY_c, dW^T += U_c^T dS_c, dLam += sum_b S_{c-1} . dS_c."""
        d = tl.program_id(0)
        pb = tl.program_id(1)
        rb = pb * BB + tl.arange(0, BB)
        mb = rb < B
        i = tl.arange(0, L)
        j = tl.arange(0, L)
        k = tl.arange(0, KP)
        xb = x_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        dyb = dy_ptr + rb[:, None].to(tl.int64) * sxb + d * sxd
        sb = s_ptr + rb[:, None].to(tl.int64) * N * (D * KP) + d * KP + k[None, :]
        dsb = ds_ptr + rb[:, None].to(tl.int64) * N * (D * KP) + d * KP + k[None, :]
        adH = tl.zeros([L, L], dtype=tl.float32)
        adPT = tl.zeros([KP, L], dtype=tl.float32)
        adWT = tl.zeros([L, KP], dtype=tl.float32)
        adLam = tl.zeros([KP], dtype=tl.float32)
        for c in range(N):
            dY = tl.load(dyb + c * L + i[None, :], mask=mb[:, None], other=0.0)
            U = tl.load(xb + c * L + j[None, :], mask=mb[:, None], other=0.0)
            Sin = tl.load(sb + c * (D * KP), mask=mb[:, None], other=0.0)
            dS = tl.load(dsb + c * (D * KP), mask=mb[:, None], other=0.0)
            adLam += tl.sum(Sin * dS, 0)
            if PREC == 1:
                adH += tl.dot(tl.trans(dY), U, input_precision='ieee')
                adPT += tl.dot(tl.trans(Sin), dY, input_precision='ieee')
                adWT += tl.dot(tl.trans(U), dS, input_precision='ieee')
            else:
                adH += tl.dot(tl.trans(dY), U)
                adPT += tl.dot(tl.trans(Sin), dY)
                adWT += tl.dot(tl.trans(U), dS)
        base = pb * D + d
        tl.store(dh_ptr + base * (L * L) + i[:, None] * L + j[None, :], adH)
        tl.store(dpt_ptr + base * (KP * L) + k[:, None] * L + i[None, :], adPT)
        tl.store(dwt_ptr + base * (L * KP) + j[:, None] * KP + k[None, :], adWT)
        tl.store(dlam_ptr + base * KP + k, adLam)
    CC.update(ok=True, fwd=_cc_fwd, bwd_chain=_cc_bwd_chain, bwd_params=_cc_bwd_params, tri=triton)

def _lif_step(v, s_prev, I_t, a, theta, beta: float):
    v = a * v * (1 - s_prev) + I_t
    s = SurrogateSpike.apply(v - theta, beta)
    return (v, s)

def _li_step(v, I_t, a):
    return a * v + I_t

_COMPILED = {}

def compiled_steps():
    if not _COMPILED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _COMPILED['lif'] = torch.compile(_lif_step, dynamic=False)
        _COMPILED['li'] = torch.compile(_li_step, dynamic=False)
    return _COMPILED

class LIFLayerCompiled(LIFLayerRef):

    def forward(self, s_in):
        I = self.norm_apply(self.proj(s_in))
        B, T, d = I.shape
        a = self.alpha()
        step = compiled_steps()['lif']
        v = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        s_prev = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        spikes = []
        for t in range(T):
            v, s = step(v, s_prev, I[:, t], a, self.theta, self.beta)
            spikes.append(s)
            s_prev = s
        S = self.drop(torch.stack(spikes, 1))
        return (S, S.mean())

class LIReadoutCompiled(LIReadoutRef):

    def forward(self, s_in):
        I = self.proj(s_in)
        B, T, C = I.shape
        a = self.alpha()
        step = compiled_steps()['li']
        v = torch.zeros(B, C, device=I.device, dtype=I.dtype)
        vs = []
        for t in range(T):
            v = step(v, I[:, t], a)
            vs.append(v)
        return torch.stack(vs, 1)

def lif_manual_forward(I, a, theta, reset):
    B, T, D = I.shape
    v = I.new_zeros(B, D)
    s = I.new_zeros(B, D)
    Vs, Ss = ([], [])
    for t in range(T):
        v = a * v * (1 - s) + I[:, t] if reset else a * v + I[:, t]
        if reset:
            s = (v - theta > 0).to(I.dtype)
        Vs.append(v)
        Ss.append(s)
    return (torch.stack(Vs, 1), torch.stack(Ss, 1))

def lif_manual_backward(G, V, a, theta, beta, reset):
    B, T, D = V.shape
    dvn = V.new_zeros(B, D)
    ga = V.new_zeros(D)
    gth = V.new_zeros(D)
    GI = torch.empty_like(V)
    for t in range(T - 1, -1, -1):
        v = V[:, t]
        g = G[:, t]
        if reset:
            x = v - theta
            s = (x > 0).to(V.dtype)
            psi = beta / (2 * (1 + beta * x.abs()) ** 2)
            ds = g - a * v * dvn
            dv = ds * psi + a * (1 - s) * dvn
            ga = ga + (dvn * v * (1 - s)).sum(0)
            gth = gth - (ds * psi).sum(0)
        else:
            dv = g + a * dvn
            ga = ga + (dvn * v).sum(0)
        GI[:, t] = dv
        dvn = dv
    return (GI, ga, gth)

class ManualLIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, I, a, theta, beta, reset):
        V, S = lif_manual_forward(I, a, theta, reset)
        ctx.save_for_backward(V, a, theta)
        ctx.beta = float(beta)
        ctx.reset = reset
        return S if reset else V

    @staticmethod
    def backward(ctx, g):
        V, a, theta = ctx.saved_tensors
        GI, ga, gth = lif_manual_backward(g, V, a, theta, ctx.beta, ctx.reset)
        return (GI, ga, gth if ctx.reset else None, None, None)

TRITON = {'ok': False, 'reason': 'not attempted'}

FusedLIF = None

def load_triton():
    global FusedLIF
    if not (CUDA or TRITON_CPU):
        TRITON.update(ok=False, reason='needs CUDA')
        return
    try:
        import triton, triton.language as tl
    except Exception as e:
        TRITON.update(ok=False, reason=f'triton import failed: {repr(e)[:100]}')
        return

    @triton.jit
    def _lif_fwd_kernel(I_ptr, a_ptr, th_ptr, V_ptr, S_ptr, T, D, RESET: tl.constexpr, BLOCK: tl.constexpr):
        pb = tl.program_id(0).to(tl.int64)
        pd = tl.program_id(1)
        offs = pd * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        a = tl.load(a_ptr + offs, mask=m, other=0.0)
        th = tl.load(th_ptr + offs, mask=m, other=0.0)
        v = tl.zeros([BLOCK], dtype=tl.float32)
        s = tl.zeros([BLOCK], dtype=tl.float32)
        base = pb * T * D
        for t in range(T):
            p = base + t * D + offs
            i = tl.load(I_ptr + p, mask=m, other=0.0)
            if RESET:
                v = a * v * (1.0 - s) + i
                s = tl.where(v - th > 0.0, 1.0, 0.0)
                tl.store(S_ptr + p, s, mask=m)
            else:
                v = a * v + i
            tl.store(V_ptr + p, v, mask=m)

    @triton.jit
    def _lif_bwd_kernel(G_ptr, V_ptr, a_ptr, th_ptr, GI_ptr, GA_ptr, GTH_ptr, T, D, beta, RESET: tl.constexpr, BLOCK: tl.constexpr):
        pb = tl.program_id(0).to(tl.int64)
        pd = tl.program_id(1)
        offs = pd * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        a = tl.load(a_ptr + offs, mask=m, other=0.0)
        th = tl.load(th_ptr + offs, mask=m, other=0.0)
        dvn = tl.zeros([BLOCK], dtype=tl.float32)
        ga = tl.zeros([BLOCK], dtype=tl.float32)
        gth = tl.zeros([BLOCK], dtype=tl.float32)
        base = pb * T * D
        for r in range(T):
            t = T - 1 - r
            p = base + t * D + offs
            v = tl.load(V_ptr + p, mask=m, other=0.0)
            g = tl.load(G_ptr + p, mask=m, other=0.0)
            if RESET:
                x = v - th
                s = tl.where(x > 0.0, 1.0, 0.0)
                q = 1.0 + beta * tl.abs(x)
                psi = beta / (2.0 * q * q)
                ds = g - a * v * dvn
                dv = ds * psi + a * (1.0 - s) * dvn
                ga += dvn * v * (1.0 - s)
                gth += -ds * psi
            else:
                dv = g + a * dvn
                ga += dvn * v
            tl.store(GI_ptr + p, dv, mask=m)
            dvn = dv
        q2 = pb * D + offs
        tl.store(GA_ptr + q2, ga, mask=m)
        tl.store(GTH_ptr + q2, gth, mask=m)
    BLOCK = 32

    class _FusedLIF(torch.autograd.Function):
        """One kernel runs the whole T recurrence per (batch row, 32 channels); backward is the exact reverse recurrence
        (same formulas as lif_manual_backward). float32 only. Per-row partial sums of dL/da, dL/dtheta are reduced in
        PyTorch (deterministic, no atomics)."""

        @staticmethod
        def forward(ctx, I, a, theta, beta, reset):
            I = I.contiguous()
            B, T, D = I.shape
            V = torch.empty_like(I)
            S = torch.empty_like(I) if reset else V
            _lif_fwd_kernel[B, triton.cdiv(D, BLOCK)](I, a.contiguous(), theta.contiguous(), V, S, T, D, RESET=reset, BLOCK=BLOCK, num_warps=1)
            ctx.save_for_backward(V, a, theta)
            ctx.beta = float(beta)
            ctx.reset = reset
            return S if reset else V

        @staticmethod
        def backward(ctx, g):
            V, a, theta = ctx.saved_tensors
            B, T, D = V.shape
            GI = torch.empty_like(V)
            GA = torch.empty(B, D, device=V.device, dtype=V.dtype)
            GTH = torch.empty_like(GA)
            _lif_bwd_kernel[B, triton.cdiv(D, BLOCK)](g.contiguous(), V, a.contiguous(), theta.contiguous(), GI, GA, GTH, T, D, ctx.beta, RESET=ctx.reset, BLOCK=BLOCK, num_warps=1)
            return (GI, GA.sum(0), GTH.sum(0) if ctx.reset else None, None, None)
    FusedLIF = _FusedLIF
    define_crisp_kernels(triton, tl)
    define_lif2_kernels(triton, tl)
    TRITON.update(ok=True, reason='', version=getattr(triton, '__version__', '?'), interpreter=TRITON_CPU, kernels={'crisp_fused': dict(L=CC_L, BB=CC_BB, KP=KP_MODES, prec='ieee' if FusedDendriticSSM.prec == 1 else 'device default (TF32 on CUDA)', num_warps=4, backward='chain + params kernels'), 'lif_fused2': dict(BLOCK=LIF2_BLOCK, UN=LIF2_UN, num_warps=1)})

class LIFLayerFused(LIFLayerRef):

    def forward(self, s_in):
        I = self.norm_apply(self.proj(s_in))
        a = self.alpha()
        S = self.drop(FusedLIF.apply(I, a, self.theta, self.beta, True))
        return (S, S.mean())

class LIReadoutFused(LIReadoutRef):

    def forward(self, s_in):
        I = self.proj(s_in)
        a = self.alpha()
        return FusedLIF.apply(I, a, torch.zeros_like(a), 0.0, False)

LIF2 = {'ok': False}

LIF2_BLOCK, LIF2_UN = (32, 8)

def define_lif2_kernels(triton, tl):

    @triton.jit
    def _lif2_fwd_kernel(I_ptr, a_ptr, th_ptr, V_ptr, S_ptr, T, D, RESET: tl.constexpr, BLOCK: tl.constexpr, UN: tl.constexpr):
        pb = tl.program_id(0).to(tl.int64)
        pd = tl.program_id(1)
        offs = pd * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        a = tl.load(a_ptr + offs, mask=m, other=0.0)
        th = tl.load(th_ptr + offs, mask=m, other=0.0)
        v = tl.zeros([BLOCK], dtype=tl.float32)
        s = tl.zeros([BLOCK], dtype=tl.float32)
        base = pb * T * D
        rr = tl.arange(0, UN)
        for t0 in range(0, T, UN):
            tt = t0 + rr
            tile = tl.load(I_ptr + base + tt[:, None] * D + offs[None, :], mask=(tt < T)[:, None] & m[None, :], other=0.0)
            for r in tl.static_range(UN):
                i = tl.sum(tl.where(rr == r, 1.0, 0.0)[:, None] * tile, 0)
                p = base + (t0 + r) * D + offs
                mk = m & (t0 + r < T)
                if RESET:
                    v = a * v * (1.0 - s) + i
                    s = tl.where(v - th > 0.0, 1.0, 0.0)
                    tl.store(S_ptr + p, s, mask=mk)
                else:
                    v = a * v + i
                tl.store(V_ptr + p, v, mask=mk)

    @triton.jit
    def _lif2_bwd_kernel(G_ptr, V_ptr, a_ptr, th_ptr, GI_ptr, GA_ptr, GTH_ptr, T, D, beta, RESET: tl.constexpr, BLOCK: tl.constexpr, UN: tl.constexpr):
        pb = tl.program_id(0).to(tl.int64)
        pd = tl.program_id(1)
        offs = pd * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        a = tl.load(a_ptr + offs, mask=m, other=0.0)
        th = tl.load(th_ptr + offs, mask=m, other=0.0)
        dvn = tl.zeros([BLOCK], dtype=tl.float32)
        ga = tl.zeros([BLOCK], dtype=tl.float32)
        gth = tl.zeros([BLOCK], dtype=tl.float32)
        base = pb * T * D
        rr = tl.arange(0, UN)
        nblk = (T + UN - 1) // UN
        for kb in range(nblk):
            t0 = (nblk - 1 - kb) * UN
            tt = t0 + rr
            mk2 = (tt < T)[:, None] & m[None, :]
            vt = tl.load(V_ptr + base + tt[:, None] * D + offs[None, :], mask=mk2, other=0.0)
            gt = tl.load(G_ptr + base + tt[:, None] * D + offs[None, :], mask=mk2, other=0.0)
            for rq in tl.static_range(UN):
                r = UN - 1 - rq
                sel = tl.where(rr == r, 1.0, 0.0)[:, None]
                v = tl.sum(sel * vt, 0)
                g = tl.sum(sel * gt, 0)
                if RESET:
                    x = v - th
                    s = tl.where(x > 0.0, 1.0, 0.0)
                    q = 1.0 + beta * tl.abs(x)
                    psi = beta / (2.0 * q * q)
                    ds = g - a * v * dvn
                    dv = ds * psi + a * (1.0 - s) * dvn
                    ga += dvn * v * (1.0 - s)
                    gth += -ds * psi
                else:
                    dv = g + a * dvn
                    ga += dvn * v
                tl.store(GI_ptr + base + (t0 + r) * D + offs, dv, mask=m & (t0 + r < T))
                dvn = dv
        q2 = pb * D + offs
        tl.store(GA_ptr + q2, ga, mask=m)
        tl.store(GTH_ptr + q2, gth, mask=m)
    LIF2.update(ok=True, fwd=_lif2_fwd_kernel, bwd=_lif2_bwd_kernel, tri=triton)

class _FusedLIF2(torch.autograd.Function):
    """As _FusedLIF (same formulas, same per-row partial sums reduced in PyTorch), with the time-unrolled kernels."""

    @staticmethod
    def forward(ctx, I, a, theta, beta, reset):
        I = I.contiguous()
        B, T, D = I.shape
        nd = (D + LIF2_BLOCK - 1) // LIF2_BLOCK
        V = torch.empty_like(I)
        S = torch.empty_like(I) if reset else V
        LIF2['fwd'][B, nd](I, a.contiguous(), theta.contiguous(), V, S, T, D, RESET=reset, BLOCK=LIF2_BLOCK, UN=LIF2_UN, num_warps=1)
        ctx.save_for_backward(V, a, theta)
        ctx.beta = float(beta)
        ctx.reset = reset
        return S if reset else V

    @staticmethod
    def backward(ctx, g):
        V, a, theta = ctx.saved_tensors
        B, T, D = V.shape
        nd = (D + LIF2_BLOCK - 1) // LIF2_BLOCK
        GI = torch.empty_like(V)
        GA = torch.empty(B, D, device=V.device, dtype=V.dtype)
        GTH = torch.empty_like(GA)
        LIF2['bwd'][B, nd](g.contiguous(), V, a.contiguous(), theta.contiguous(), GI, GA, GTH, T, D, ctx.beta, RESET=ctx.reset, BLOCK=LIF2_BLOCK, UN=LIF2_UN, num_warps=1)
        return (GI, GA.sum(0), GTH.sum(0) if ctx.reset else None, None, None)

class LIFLayerFused2(LIFLayerRef):

    def forward(self, s_in):
        I = self.norm_apply(self.proj(s_in))
        a = self.alpha()
        S = self.drop(_FusedLIF2.apply(I, a, self.theta, self.beta, True))
        return (S, S.mean())

class LIReadoutFused2(LIReadoutRef):

    def forward(self, s_in):
        I = self.proj(s_in)
        a = self.alpha()
        return _FusedLIF2.apply(I, a, torch.zeros_like(a), 0.0, False)

def st_spike(x, beta):
    """Forward: (x > 0) exactly (F - F.detach() == 0). Gradient: d/dx [beta x / (2 (1 + beta|x|))] = beta / (2 (1 + beta|x|)^2),
    the reference SurrogateSpike derivative. Written without a custom autograd.Function so torch_xla's scan can trace it."""
    Fv = beta * x / (2 * (1 + beta * x.abs()))
    return (x > 0).to(x.dtype) + (Fv - Fv.detach())

class LIFLayerST(LIFLayerRef):
    """Reference loop with st_spike instead of SurrogateSpike (float64 gate of the XLA-scan step)."""

    def forward(self, s_in):
        I = self.norm_apply(self.proj(s_in))
        B, T, d = I.shape
        a = self.alpha()
        v = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        s_prev = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        spikes = []
        for t in range(T):
            v = a * v * (1 - s_prev) + I[:, t]
            s = st_spike(v - self.theta, self.beta)
            spikes.append(s)
            s_prev = s
        S = self.drop(torch.stack(spikes, 1))
        return (S, S.mean())

class LIFLayerXlaScan(LIFLayerRef):

    def forward(self, s_in):
        from torch_xla.experimental.scan import scan as xla_scan
        I = self.norm_apply(self.proj(s_in))
        B, T, d = I.shape
        a = self.alpha()
        beta = self.beta

        def step(carry, x):
            v, s, aa, tt = carry
            v = aa * v * (1 - s) + x
            s = st_spike(v - tt, beta)
            return ((v, s, aa, tt), s)
        z = I.new_zeros(B, d)
        _, S = xla_scan(step, (z, z.clone(), a.unsqueeze(0).expand(B, d).contiguous(), self.theta.unsqueeze(0).expand(B, d).contiguous()), I.transpose(0, 1).contiguous())
        S = self.drop(S.transpose(0, 1))
        return (S, S.mean())

class LIReadoutXlaScan(LIReadoutRef):

    def forward(self, s_in):
        from torch_xla.experimental.scan import scan as xla_scan
        I = self.proj(s_in)
        B, T, C = I.shape
        a = self.alpha()

        def step(carry, x):
            v, aa = carry
            v = aa * v + x
            return ((v, aa), v)
        _, Vs = xla_scan(step, (I.new_zeros(B, C), a.unsqueeze(0).expand(B, C).contiguous()), I.transpose(0, 1).contiguous())
        return Vs.transpose(0, 1)

LIF_SWAP = {'lif_loop': None, 'lif_compiled': (LIFLayerCompiled, LIReadoutCompiled), 'lif_fused': (LIFLayerFused, LIReadoutFused), 'lif_fused2': (LIFLayerFused2, LIReadoutFused2), 'lif_xla_scan': (LIFLayerXlaScan, LIReadoutXlaScan), 'lif_st': (LIFLayerST, None)}

def build_lif(impl, seed=0):
    set_all_seeds(seed)
    net = LIFNet(SimpleNamespace(**cfg.lif_recipe), cfg.n_classes)
    sw = LIF_SWAP[impl]
    if sw is not None:
        for m in net.modules():
            if type(m) is LIFLayerRef and sw[0] is not None:
                m.__class__ = sw[0]
            elif type(m) is LIReadoutRef and sw[1] is not None:
                m.__class__ = sw[1]
    return net

def build_spsn(T, seed=0):
    set_all_seeds(seed)
    return SPSN['create_network'](dict(cfg.spsn_params), device, T)

FAMILY = {'crisp_hillis': 'crisp', 'crisp_chunked': 'crisp', 'crisp_fft': 'crisp', 'crisp_toeplitz': 'crisp', 'crisp_chunkmm': 'crisp', 'crisp_fused': 'crisp', 'lif_loop': 'lif', 'lif_compiled': 'lif', 'lif_fused': 'lif', 'lif_fused2': 'lif', 'lif_xla_scan': 'lif', 'spsn_sb': 'spsn'}

def impl_name(impl):
    return f'crisp_chunkmm{cfg.chunkmm_L}' if impl == 'crisp_chunkmm' else impl

def base_impl(name):
    return 'crisp_chunkmm' if name.startswith('crisp_chunkmm') else name

def make_model(impl, T):
    fam = FAMILY[impl]
    if fam == 'crisp':
        m = build_crisp(impl)
    elif fam == 'lif':
        m = build_lif(impl)
    elif fam == 'spsn':
        m = build_spsn(T)
    else:
        raise ValueError(f'Unknown implementation: {impl}')
    return m.to(device)

def train_logits(impl, m, x):
    fam = FAMILY[impl]
    if fam == 'crisp':
        return m(x, 'graded', CCFG.beta_end)[0]
    if fam == 'lif':
        return m(x)[0]
    if fam == 'spsn':
        return m(x).mean(1)
    return m(x)

def infer_logits(impl, m, x):
    fam = FAMILY[impl]
    if fam == 'crisp':
        return m(x, 'sample', CCFG.beta_end)[0]
    return train_logits(impl, m, x)

def make_batch(B, T, seed, dense=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, T, cfg.n_in, generator=g)
    x = x if dense else (x < cfg.p_in).float()
    y = torch.randint(0, cfg.n_classes, (B,), generator=g)
    return (x.to(device), y.to(device))

def sync_wait():
    if XLA:
        try:
            torch_xla.sync(wait=True)
        except TypeError:
            xm.mark_step()
            xm.wait_device_ops()
        xm.wait_device_ops()
    elif CUDA:
        torch.cuda.synchronize()

def sync_nowait():
    if XLA:
        try:
            torch_xla.sync()
        except Exception:
            xm.mark_step()
    elif CUDA:
        torch.cuda.synchronize()

def is_oom(e):
    s = str(e).lower()
    return isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ())) or 'out of memory' in s or 'resource_exhausted' in s or ('resource exhausted' in s) or ('hbm' in s)

def free_mem():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

def device_label():
    lab = _device_label_base()
    return f'{lab} matmul-{XLA_PREC_REQ}' if XLA and XLA_PREC_REQ else lab

def _device_label_base():
    if DEVICE_LABEL.strip():
        return DEVICE_LABEL.strip()
    if CUDA:
        return torch.cuda.get_device_name(0)
    if XLA:
        acc = os.environ.get('TPU_ACCELERATOR_TYPE', '')
        ver = None
        try:
            from torch_xla._internal import tpu as _tpu
            ver = _tpu.version()
        except Exception:
            pass
        if acc:
            return f'TPU-{acc}'
        if ver:
            return f'TPU-v{ver}'
        try:
            hw = xm.xla_device_hw(device)
            if hw and hw != 'TPU':
                return f'XLA-{hw}'
        except Exception:
            pass
        return 'TPU-unknown'
    return f"CPU-{(platform.processor() or platform.machine() or 'generic')[:30]}"

def device_mem_bytes():
    if CUDA:
        return torch.cuda.get_device_properties(0).total_memory
    if XLA:
        lab = device_label().lower()
        for k, gb in (('v2', 8), ('v3', 16), ('v4', 32), ('v5p', 95), ('v5e', 16), ('v5lite', 16), ('v6', 32)):
            if k in lab:
                return gb * 2 ** 30
        return 16 * 2 ** 30
    return None

def versions():
    v = {'torch': torch.__version__, 'python': platform.python_version().rsplit('.', 1)[0], 'numpy': np.__version__}
    if CUDA:
        v['cuda'] = torch.version.cuda
        v['cudnn'] = torch.backends.cudnn.version()
    if XLA:
        v['torch_xla'] = getattr(torch_xla, '__version__', '?')
    if TRITON.get('ok'):
        v['triton'] = TRITON.get('version')
    return v

def driver_version():
    if not CUDA:
        return None
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None

def xla_counters():
    if not XLA:
        return {}
    try:
        return {n: xmet.counter_value(n) for n in xmet.counter_names() if n.startswith('aten::')}
    except Exception:
        return {}

def xla_compiles():
    if not XLA:
        return None
    try:
        d = xmet.metric_data('CompileTime')
        return int(d[0]) if d else 0
    except Exception:
        return None

def xla_precision():
    try:
        return torch_xla._XLAC._xla_get_mat_mul_precision()
    except Exception:
        return None

def _rel(a, b):
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    return float((a - b).abs().max() / (b.abs().max() + 1e-300))

def _grel(ga, gb):
    """max |ga - gb| over ALL parameters / max |gb| over ALL parameters. (Per-parameter relative error is meaningless for
    gradients that are structurally zero, e.g. a projection bias followed by BatchNorm: its 'gradient' is rounding noise.)"""
    num = max((float((ga[k].detach().double().cpu() - gb[k].detach().double().cpu()).abs().max()) for k in gb))
    den = max((float(gb[k].detach().double().cpu().abs().max()) for k in gb))
    return num / (den + 1e-300)

def _zero_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    return m

def _grads(m):
    return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in m.named_parameters()}

def _cos(ga, gb):
    a = torch.cat([ga[k].double().cpu().flatten() for k in sorted(ga)])
    b = torch.cat([gb[k].double().cpu().flatten() for k in sorted(gb)])
    return float(a @ b / (a.norm() * b.norm() + 1e-300))

def _run_fb(impl, m, x, y, dt=1.0):
    m.train()
    m.zero_grad(set_to_none=True)
    if FAMILY[impl] == 'crisp':
        m.set_dt(dt)
    lg = train_logits(impl, m, x)
    F.cross_entropy(lg, y).backward()
    return (lg.detach(), _grads(m))

def gate_cc_kernel():
    """crisp_fused only, on THIS device: the Triton kernels (forward and the five parameter/input gradients), float32,
    against the pure-PyTorch chunked fallback of the same block run in FLOAT64 on the device (a float32 fallback would
    itself run under TF32 on CUDA and cannot serve as the reference: device precision check). 'ieee' = tl.dot at full
    fp32 precision (expected ~1e-6), 'default' = the precision the timing runs use (TF32 on CUDA, expected ~1e-3). A wrong
    kernel (indexing, adjoint) is O(1), so both must stay below 1e-2. Odd B (masked rows) and several chunks are covered."""
    out = {'cases': [], 'ieee_max_rel_err': 0.0, 'default_max_rel_err': 0.0, 'reference': 'chunk_conv_ref in float64 on device'}
    names = ('Y', 'dxT', 'dh', 'dPT', 'dWT', 'dLam')
    for B, D, N in ((5, 4, 3), (70, 3, 4), (1, 2, 1)):
        g = torch.Generator().manual_seed(B * 10 + N)
        raw_lam = torch.log(torch.linspace(0.05, 2.5, CCFG.k_modes)).repeat(D, 1) + 0.01 * torch.randn(D, CCFG.k_modes, generator=g)
        mix = torch.randn(D, CCFG.k_modes, generator=g) / CCFG.k_modes ** 0.5
        xT0 = torch.randn(B, D, N * CC_L, generator=g)
        dY = torch.randn(B, D, N * CC_L, generator=g).to(device)

        def run(prec, use_kernel, dtype):
            xT = xT0.to(device=device, dtype=dtype).clone().requires_grad_(True)
            h, PT, WT, Lam = [t.detach().clone().requires_grad_(True) for t in chunk_params(raw_lam.to(device=device, dtype=dtype), mix.to(device=device, dtype=dtype), 1.0, CC_L)]
            Y = ChunkConvFn.apply(xT, h, PT, WT, Lam, CC_L, prec) if use_kernel else chunk_conv_ref(xT, h, PT, WT, Lam, CC_L)
            (Y * dY.to(dtype)).sum().backward()
            sync_wait()
            z = lambda t: torch.zeros_like(t) if t.grad is None else t.grad
            return (Y.detach(), z(xT), z(h), z(PT), z(WT), z(Lam))
        ref = run(0, False, torch.float64)
        for prec in (1, 0):
            errs = {n: _rel(a, b) for n, a, b in zip(names, run(prec, True, torch.float32), ref)}
            out['cases'].append({'B': B, 'D': D, 'N': N, 'prec': 'ieee' if prec == 1 else 'default', 'rel_err': errs})
            k = 'ieee_max_rel_err' if prec == 1 else 'default_max_rel_err'
            out[k] = max(out[k], max(errs.values()))
    out['pass'] = out['ieee_max_rel_err'] < 0.01 and out['default_max_rel_err'] < 0.01
    return out

def gate_crisp(impl):
    out = {'kind': 'crisp'}
    if impl == 'crisp_fused':
        if not CC.get('ok'):
            return {'kind': 'crisp', 'pass': False, 'error': 'fused kernels not loaded'}
        out['kernel'] = gate_cc_kernel()
    worst = 0.0
    cases = []
    for T, dt in ((77, 1.0), (200, 0.5)):
        g = torch.Generator().manual_seed(T)
        x = (torch.rand(2, T, cfg.n_in, generator=g) < cfg.p_in).double()
        y = torch.randint(0, cfg.n_classes, (2,), generator=g)
        ref = _zero_dropout(build_crisp('crisp_hillis').double())
        imp = _zero_dropout(build_crisp(impl).double())
        lr_, gr = _run_fb('crisp_hillis', ref, x, y, dt)
        li, gi = _run_fb(impl, imp, x, y, dt)
        e = max(_rel(li, lr_), _grel(gi, gr))
        worst = max(worst, e)
        cases.append({'T': T, 'dt': dt, 'max_rel_err': e})
    out['fp64_cpu'] = cases
    out['fp64_max_rel_err'] = worst
    out['fp64_pass'] = worst < 1e-09
    T = cfg.gate_T_device
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(2, T, cfg.n_in, generator=g) < cfg.p_in).float()
    y = torch.randint(0, cfg.n_classes, (2,), generator=g)
    ref = _zero_dropout(build_crisp('crisp_hillis').double())
    lr_, gr = _run_fb('crisp_hillis', ref, x.double(), y)
    imp = _zero_dropout(build_crisp(impl)).to(device)
    li, gi = _run_fb(impl, imp, x.to(device), y.to(device))
    sync_wait()
    li = li.cpu()
    gi = {k: v.cpu() for k, v in gi.items()}
    finite = bool(torch.isfinite(li).all()) and all((bool(torch.isfinite(v).all()) for v in gi.values()))
    out['device_T'] = T
    out['device_logit_rel_err'] = _rel(li, lr_)
    out['device_grad_rel_err'] = _grel(gi, gr)
    out['device_grad_cos'] = _cos(gi, gr)
    out['device_pass'] = finite and out['device_logit_rel_err'] < 0.05 and (out['device_grad_cos'] > 0.99)
    out['pass'] = out['fp64_pass'] and out['device_pass'] and out.get('kernel', {}).get('pass', True)
    del ref, imp
    free_mem()
    return out

def gate_lif_math():
    """CPU, float32 (the reference SurrogateSpike returns float32 spikes, so the LIF reference cannot run in float64):
    (a) manual-backward formulas (implemented by the Triton kernel) vs reference autograd, one layer, LIF with reset and the
    LI readout: identical forward, gradients equal up to float32 rounding; (b) straight-through spike (used by the XLA scan)
    vs reference, whole network: identical forward, gradients equal up to rounding. A wrong formula shows as O(1) error."""
    out = {'kind': 'lif_math', 'dtype': 'float32'}
    g = torch.Generator().manual_seed(3)
    B, T, D = (3, 77, 40)
    beta = cfg.lif_recipe['surrogate_beta']
    I0 = torch.randn(B, T, D, generator=g)
    a0 = torch.rand(D, generator=g) * 0.5 + 0.4
    th0 = torch.randn(D, generator=g) * 0.1
    G = torch.randn(B, T, D, generator=g)
    fwd, grad = ({}, {})
    for reset in (True, False):
        Ir, ar, tr = [t.clone().requires_grad_(True) for t in (I0, a0, th0)]
        v = torch.zeros(B, D)
        s = torch.zeros(B, D)
        outs = []
        for t in range(T):
            if reset:
                v = ar * v * (1 - s) + Ir[:, t]
                s = SurrogateSpike.apply(v - tr, beta)
                outs.append(s)
            else:
                v = ar * v + Ir[:, t]
                outs.append(v)
        Or = torch.stack(outs, 1)
        (Or * G).sum().backward()
        Im, am, tm = [t.clone().requires_grad_(True) for t in (I0, a0, th0)]
        Om = ManualLIF.apply(Im, am, tm, beta, reset)
        (Om * G).sum().backward()
        k = 'lif_reset' if reset else 'li_readout'
        fwd[k] = _rel(Om, Or)
        grad[k] = max(_rel(Im.grad, Ir.grad), _rel(am.grad, ar.grad)) if not reset else max(_rel(Im.grad, Ir.grad), _rel(am.grad, ar.grad), _rel(tm.grad, tr.grad))
    x = (torch.rand(2, 77, cfg.n_in, generator=g) < cfg.p_in).float()
    y = torch.randint(0, cfg.n_classes, (2,), generator=g)
    ref = _zero_dropout(build_lif('lif_loop'))
    st = _zero_dropout(build_lif('lif_st'))
    lr_, gr = _run_fb('lif_loop', ref, x, y)
    ls, gs = _run_fb('lif_loop', st, x, y)
    fwd['st_spike_net'] = _rel(ls, lr_)
    grad['st_spike_net'] = _grel(gs, gr)
    out['fwd_rel_err'] = fwd
    out['grad_rel_err'] = grad
    out['pass'] = max(fwd.values()) == 0.0 and max(grad.values()) < 0.0001
    return out

def gate_lif_device(impl):
    """float32 on THIS device: impl vs the reference loop on the same device, same weights, dropout off."""
    T = min(200, cfg.gate_T_device)
    g = torch.Generator().manual_seed(11)
    x = (torch.rand(4, T, cfg.n_in, generator=g) < cfg.p_in).float().to(device)
    y = torch.randint(0, cfg.n_classes, (4,), generator=g).to(device)
    ref = _zero_dropout(build_lif('lif_loop')).to(device)
    imp = _zero_dropout(build_lif(impl)).to(device)
    with torch.no_grad():
        ref.train()
        imp.train()
        S_ref = ref.layers[0](x)[0]
        S_imp = imp.layers[0](x)[0]
        sync_wait()
        agree = float((S_ref == S_imp).float().mean().cpu())
    lr_, gr = _run_fb('lif_loop', ref, x, y)
    li, gi = _run_fb(impl, imp, x, y)
    sync_wait()
    out = {'kind': 'lif_device', 'T': T, 'layer1_spike_agreement': agree, 'logit_rel_err': _rel(li, lr_), 'grad_cos': _cos(gi, gr), 'grad_max_rel_err': _grel(gi, gr)}
    out['pass'] = agree >= 0.999 and out['grad_cos'] > 0.999 and (out['logit_rel_err'] < 0.001)
    del ref, imp
    free_mem()
    return out

def gate_spsn():
    m = build_spsn(50).to(device)
    x, _ = make_batch(2, 50, 0)
    with torch.no_grad():
        o = m(x)
        sync_wait()
    ok = tuple(o.shape) == (2, 50, cfg.n_classes) and bool(torch.isfinite(o.cpu()).all())
    del m
    free_mem()
    return {'kind': 'spsn_sanity', 'out_shape': list(o.shape), 'pass': ok, 'commit': SPSN.get('head'), 'md5': SPSN.get('md5')}

def time_fn(step, n_warm):
    """first step (compile/autotune), warm-up, then an adaptive number of synced steps within the unit budget.
    The step's output is kept alive until the sync: on XLA a tensor nobody holds is never computed (a dropped inference
    output would time an empty graph)."""
    t0 = time.perf_counter()
    o = step()
    sync_wait()
    first = time.perf_counter() - t0
    del o
    last = first
    for _ in range(n_warm - 1):
        t0 = time.perf_counter()
        o = step()
        sync_wait()
        last = time.perf_counter() - t0
        del o
    n = int(max(cfg.min_timed, min(cfg.n_timed_max, cfg.unit_budget_s / max(last, 1e-06))))
    c0 = xla_compiles()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        o = step()
        sync_wait()
        ts.append(time.perf_counter() - t0)
        del o
    c1 = xla_compiles()
    r = {'first_ms': first * 1000.0, 'median_ms': float(np.median(ts)) * 1000.0, 'min_ms': float(np.min(ts)) * 1000.0, 'max_ms': float(np.max(ts)) * 1000.0, 'n_timed': n, 'n_warm': n_warm, 'times_ms': [t * 1000.0 for t in ts]}
    if XLA:
        r['recompiles_during_timing'] = c1 - c0 if c0 is not None and c1 is not None else None
        t0 = time.perf_counter()
        for _ in range(n):
            o = step()
            sync_nowait()
        xm.wait_device_ops()
        r['pipelined_ms'] = (time.perf_counter() - t0) / n * 1000.0
        del o
    return r

def mem_of(step):
    """CUDA allocations in MiB (2**20 bytes); retain legacy *_MB keys."""
    if not CUDA:
        return {}
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    o = step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del o
    return {'peak_MB': peak / 2 ** 20, 'above_base_MB': (peak - base) / 2 ** 20, 'base_MB': base / 2 ** 20}

def run_unit(u):
    impl, T, B = (u['impl'], u['T'], u['B'])
    m = make_model(impl, T)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    x, y = make_batch(B, T, seed=1000 + T + B, dense=False)
    sync_wait()
    n_warm = cfg.n_warm_xla if XLA else cfg.n_warm_gpu
    fb0 = xla_counters()

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(train_logits(impl, m, x), y)
        loss.backward()
        opt.step()
        return loss.detach()
    m.train()
    tr = time_fn(train_step, n_warm)
    tr['mem'] = mem_of(train_step)

    def infer_step():
        with torch.no_grad():
            return infer_logits(impl, m, x)
    m.eval()
    inf = time_fn(infer_step, max(1, n_warm - 1))
    inf['mem'] = mem_of(infer_step)
    fb1 = xla_counters()
    rec = {'status': 'ok', 'train': tr, 'infer': inf, 'train_samples_per_s': B / (tr['median_ms'] / 1000.0), 'infer_samples_per_s': B / (inf['median_ms'] / 1000.0), 'n_params': sum((p.numel() for p in m.parameters()))}
    if XLA:
        rec['xla_fallback_ops'] = {k: fb1[k] - fb0.get(k, 0) for k in fb1 if fb1[k] - fb0.get(k, 0) > 0}
    del m, opt, x, y
    free_mem()
    return rec

def unit_id(impl, T, B):
    return f'{impl}__T{T}__B{B}'

def est_act_bytes(impl, T, B):
    """Rough training-activation estimate, used only on XLA (an XLA OOM kills the runtime) and for the Toeplitz guard."""
    H, K, nl, C = (CCFG.n_hidden, CCFG.k_modes, CCFG.n_layers, cfg.n_classes)
    if impl == 'crisp_hillis':
        return 4 * B * T * H * K * (2 * math.ceil(math.log2(max(T, 2))) + 4) * nl
    if impl == 'crisp_chunked':
        return 4 * B * T * H * K * (2 * 5 + 6) * nl
    if impl == 'crisp_toeplitz':
        return 4 * (nl * H + C) * T * T * 3 + 4 * B * T * H * 12 * nl
    if impl in ('crisp_fft', 'crisp_chunkmm', 'spsn_sb'):
        return 4 * B * T * H * 16 * nl
    if impl == 'crisp_fused':
        return 4 * B * T * H * 8 * nl
    return 4 * B * T * H * 12 * nl

def project(u, done):
    """Projected single-step and first-step time from the nearest finished unit of the same implementation."""
    best = None
    for r in done.values():
        if r.get('impl') != u['name'] or r.get('status') != 'ok':
            continue
        if r['T'] > u['T'] or r['B'] > u['B']:
            continue
        sc = u['T'] / r['T'] * (u['B'] / r['B'] if r['B'] > 1 else 1.0) ** 0.5
        d = u['T'] / r['T'] * (u['B'] / r['B'])
        fsc = sc * (u['T'] / r['T']) if XLA else sc
        if best is None or d < best[0]:
            best = (d, sc * r['train']['median_ms'] / 1000.0, fsc * r['train']['first_ms'] / 1000.0)
    return (None, None) if best is None else (best[1], best[2])

def dominated(u, failed):
    for f in failed:
        if f['impl'] == u['name'] and u['T'] >= f['T'] and (u['B'] >= f['B']):
            return f
    return None


def configure(config=None):
    """Select a device and the original matched model/timing settings."""
    global cfg, CCFG, device, CUDA, XLA, DEVICE_KIND, DEVICE_LABEL, XLA_PREC_REQ
    global torch_xla, xm, xmet
    config = config or {}
    options = config.get('benchmark', {})
    model = config.get('model', {})
    CCFG = Config(**{key: value for key, value in model.items() if key in Config.__dataclass_fields__})
    if not CCFG.ct_exact or not CCFG.zoh_b or CCFG.norm_type != 'batchnorm':
        raise ValueError('The matched benchmark requires exact continuous-time dendrites and batch normalization')
    lif_recipe = dict(n_in=CCFG.n_in, n_hidden=CCFG.n_hidden, n_layers=CCFG.n_layers,
                      hidden_dropout=CCFG.hidden_dropout, norm_type='batchnorm', surrogate_beta=5.0,
                      learnable_tau=True, tau_init_alpha=0.70, theta_init=0.0, readout_type='li',
                      readout_pool='mean', readout_learnable_tau=True, reg_scope='hidden')
    cfg = SimpleNamespace(
        n_in=CCFG.n_in, n_classes=CCFG.n_classes, p_in=float(options.get('input_density', 0.05)),
        lr=1e-3, weight_decay=1e-4, tf32=bool(options.get('tf32', True)),
        chunkmm_L=int(options.get('chunk_length', 64)), gate_T_device=int(options.get('gate_timesteps', 1000)),
        n_warm_gpu=int(options.get('warmup_gpu', options.get('warmup', 3))),
        n_warm_xla=int(options.get('warmup_xla', options.get('warmup', 5))),
        n_timed_max=int(options.get('repeats', 15)), min_timed=int(options.get('min_repeats', 3)),
        unit_budget_s=float(options.get('unit_budget_seconds', 30)),
        max_step_s=float(options.get('max_step_seconds', 60)),
        max_first_s=float(options.get('max_first_seconds', 900)),
        toeplitz_mem_frac=float(options.get('toeplitz_memory_fraction', 0.6)),
        xla_mem_frac=float(options.get('xla_memory_fraction', 1.0)),
        lif_recipe=lif_recipe,
        spsn_params=dict(neuron='SPSN-SB', nb_layers=CCFG.n_layers, input_size=CCFG.n_in,
                         hidden_size=CCFG.n_hidden, nb_class=CCFG.n_classes, tau_mem=2e-2, tau_syn=2e-2),
    )
    cfg.min_timed = min(cfg.min_timed, cfg.n_timed_max)
    if min(cfg.n_warm_gpu, cfg.n_warm_xla, cfg.n_timed_max, cfg.min_timed, cfg.chunkmm_L, cfg.gate_T_device) < 1:
        raise ValueError('Warmup, repeats, chunk length and gate timesteps must be positive')
    if not 0 < cfg.p_in <= 1:
        raise ValueError('input_density must lie in (0, 1]')
    requested = str(config.get('device', 'cpu'))
    XLA = requested in {'xla', 'tpu'} or requested.startswith('xla:')
    CUDA = requested.startswith('cuda')
    if CUDA and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if XLA:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as xmet
        device = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
        DEVICE_KIND = 'tpu'
    else:
        device = torch.device(requested)
        DEVICE_KIND = 'gpu' if CUDA else 'cpu'
    DEVICE_LABEL = str(options.get('device_label', ''))
    XLA_PREC_REQ = str(options.get('xla_matmul_precision', ''))
    if XLA and XLA_PREC_REQ:
        try:
            torch_xla.backends.set_mat_mul_precision(XLA_PREC_REQ)
        except AttributeError:
            torch_xla._XLAC._xla_set_mat_mul_precision(XLA_PREC_REQ)
        if str(xla_precision()).lower() != XLA_PREC_REQ.lower():
            raise RuntimeError('XLA matmul precision differs from the requested precision')
    if CUDA:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high' if cfg.tf32 else 'highest')
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    FusedDendriticSSM.prec = 0 if cfg.tf32 else 1
    return cfg


def load_spsn(path='external_code/spsn', fetch=False):
    """Load the official SPSN source only after revision and file checks."""
    directory = resolve_path(path)
    SPSN.clear()
    SPSN.update(ok=False, reason='source checkout unavailable')
    if not directory.exists() and not fetch:
        SPSN['reason'] = 'Pinned external_code/spsn checkout missing; set fetch_dependencies=true to retrieve it'
        return
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', SPSN_URL, str(directory)], check=True, timeout=300)
        subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', SPSN_COMMIT], check=True, timeout=120)
    head = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'], check=True,
                          text=True, stdout=subprocess.PIPE, timeout=30).stdout.strip()
    if head != SPSN_COMMIT:
        raise RuntimeError('SPSN checkout differs from the pinned revision')
    hashes = {name: hashlib.md5((directory / name).read_bytes()).hexdigest() for name in SPSN_MD5}
    if hashes != SPSN_MD5:
        raise RuntimeError('SPSN source does not match the pinned source checksums')
    previous = {key: sys.modules.pop(key) for key in list(sys.modules)
                if key == 'network' or key == 'neurons' or key.startswith('neurons.')}
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module('network')
    finally:
        sys.path.remove(str(directory))
        for key in list(sys.modules):
            if key == 'network' or key == 'neurons' or key.startswith('neurons.'):
                del sys.modules[key]
        sys.modules.update(previous)
    SPSN.update(ok=True, reason='', head=head, md5=hashes, create_network=module.create_network)


def available_impls(requested, compile_cpu=False):
    available, unavailable = [], {}
    for name in requested:
        if name not in FAMILY:
            raise ValueError(f'Unknown benchmark implementation: {name}')
        if name == 'lif_compiled' and not (CUDA or compile_cpu):
            unavailable[name] = 'Compiled-step comparison requires CUDA; compile_cpu permits an explicit CPU check'
        elif name in {'crisp_fused', 'lif_fused', 'lif_fused2'} and not TRITON.get('ok'):
            unavailable[name] = TRITON.get('reason', 'Triton unavailable')
        elif name == 'lif_xla_scan' and not XLA:
            unavailable[name] = 'Requires torch_xla and an XLA device'
        elif name == 'spsn_sb' and not SPSN.get('ok'):
            unavailable[name] = SPSN.get('reason', 'Pinned SPSN source unavailable')
        else:
            available.append(name)
    return available, unavailable


def plan_units(implementations, options):
    lengths = options.get('timesteps', [100, 250, 500, 1000, 2000, 5000, 10000])
    batches = options.get('batch_sizes', [64])
    points = {(int(length), int(batch)) for length in lengths for batch in batches}
    points |= {(int(length), int(batch)) for length in options.get('batch_sweep_timesteps', [])
               for batch in options.get('batch_sweep_sizes', [])}
    points |= {(int(length), 1) for length in options.get('latency_timesteps', [])}
    if not points or any(length < 1 or batch < 1 for length, batch in points):
        raise ValueError('Benchmark timesteps and batch sizes must be positive')
    units = [{'impl': impl, 'name': impl_name(impl), 'T': length, 'B': batch,
              'uid': unit_id(impl_name(impl), length, batch)}
             for impl in implementations for length, batch in points]
    order = {name: index for index, name in enumerate(implementations)}
    return sorted(units, key=lambda unit: (unit['T'] * max(unit['B'], 16), unit['T'], unit['B'], order[unit['impl']]))


def _write_result(result, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(config):
    """Run all requested implementations after their equivalence checks.

    Configuration: device, output, optional model dimensions, and benchmark
    {timesteps, batch_sizes, implementations, warmup, repeats}. Additional
    timing budgets and batch/latency sweeps preserve the complete protocol.
    Only completed measurements enter throughput ratios. Projection-based
    skips are explicitly recorded and are never reported as measurements.
    """
    configure(config)
    options = config.get('benchmark', {})
    requested = list(options.get('implementations', FAMILY))
    if not requested:
        raise ValueError('At least one benchmark implementation is required')
    output = resolve_path(config.get('output', 'outputs/runtime_benchmark'))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'benchmark.json'
    load_triton()
    if 'spsn_sb' in requested:
        try:
            load_spsn(options.get('spsn_path', 'external_code/spsn'), bool(options.get('fetch_dependencies', False)))
        except (OSError, subprocess.SubprocessError, ImportError) as error:
            SPSN.update(ok=False, reason=f'{type(error).__name__}: {error}')
    implementations, skipped = available_impls(requested, bool(options.get('compile_cpu', False)))
    for name, reason in skipped.items():
        print(f'[unavailable] {name}: {reason}', flush=True)
    identity = {'configuration': config, 'versions': versions(), 'device': device_label(),
                'driver': driver_version(), 'xla_precision': xla_precision() if XLA else None,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = {'fingerprint': fingerprint, 'metadata': identity, 'gates': {}, 'units': {},
              'unavailable': skipped, 'projected_skips': [],
              'protocol': {'input': 'Seeded Bernoulli spikes',
                           'training': 'Cross-entropy on mean-pooled logits; AdamW; no rate penalty',
                           'crisp_training': 'mean-field', 'crisp_inference': 'sampled',
                           'timing': 'Synchronized per step; first step and warmup reported separately',
                           'memory': 'CUDA allocated peak and increment above baseline in MiB (2**20 bytes); legacy *_MB keys; unavailable elsewhere',
                           'memory_unit': 'MiB',
                           'precision': 'float32; shared TF32 setting on CUDA'}}
    if target.exists():
        stored = json.loads(target.read_text(encoding='utf-8'))
        if stored.get('fingerprint') != fingerprint:
            raise RuntimeError('Output contains a different benchmark configuration or runtime; choose another relative output path')
        result = stored
        result['unavailable'] = skipped
        result['projected_skips'] = []
    _write_result(result, target)
    if any(FAMILY[name] == 'lif' for name in implementations):
        print('[gate] LIF manual adjoint and straight-through recurrence', flush=True)
        result['gates']['lif_math'] = gate_lif_math()
        _write_result(result, target)
    passed = []
    for name in implementations:
        print(f'[gate] {name}', flush=True)
        try:
            gate = (gate_crisp(name) if FAMILY[name] == 'crisp'
                    else gate_spsn() if FAMILY[name] == 'spsn'
                    else gate_lif_device(name))
            if FAMILY[name] == 'lif' and not result['gates']['lif_math']['pass']:
                gate['pass'] = False
                gate['error'] = 'The LIF mathematical equivalence check failed'
        except Exception as error:
            gate = {'pass': False, 'error': f'{type(error).__name__}: {error}'}
        result['gates'][name] = gate
        _write_result(result, target)
        if gate['pass']:
            passed.append(name)
        else:
            print(f'[gate failed] {name}; no timing measurements will be made', flush=True)
            if XLA:
                raise RuntimeError('An XLA equivalence check failed; results saved before timing')
    units = plan_units(passed, options)
    failed = [{'impl': record['impl'], 'T': record['T'], 'B': record['B']}
              for record in result['units'].values() if record['status'] in {'oom', 'error', 'crashed'}]
    in_flight = result.pop('in_flight', None)
    if in_flight and in_flight not in result['units']:
        crashes = result.setdefault('crash_counts', {})
        crashes[in_flight] = crashes.get(in_flight, 0) + 1
        if crashes[in_flight] >= 2:
            unit = next((unit for unit in units if unit['uid'] == in_flight), None)
            if unit is not None:
                record = {'status': 'crashed', 'impl': unit['name'], 'T': unit['T'], 'B': unit['B'],
                          'error': 'The process stopped twice during this workload'}
                result['units'][in_flight] = record
                failed.append(record)
    for index, unit in enumerate(units, 1):
        uid = unit['uid']
        if uid in result['units']:
            print(f'[{index}/{len(units)}] {uid}: already recorded', flush=True)
            continue
        reason = None
        prior_failure = dominated(unit, failed)
        step, first = project(unit, result['units'])
        capacity = options.get('memory_bytes') or device_mem_bytes()
        if prior_failure:
            reason = f"Previous failure at T={prior_failure['T']}, B={prior_failure['B']}"
        elif step is not None and (step > cfg.max_step_s or first > cfg.max_first_s):
            reason = f'Projected {step:.3g} seconds per step and {first:.3g} seconds for the first step'
        elif unit['impl'] == 'crisp_toeplitz' and capacity and 4 * (CCFG.n_layers * CCFG.n_hidden + cfg.n_classes) * unit['T'] ** 2 * 3 > cfg.toeplitz_mem_frac * capacity:
            reason = 'Toeplitz allocation exceeds the configured memory guard'
        elif XLA and capacity and est_act_bytes(unit['impl'], unit['T'], unit['B']) > cfg.xla_mem_frac * capacity:
            reason = 'Estimated activation storage exceeds the configured XLA memory guard'
        if reason:
            result['projected_skips'].append({'workload': uid, 'reason': reason})
            print(f'[{index}/{len(units)}] {uid}: skipped ({reason})', flush=True)
            _write_result(result, target)
            continue
        print(f'[{index}/{len(units)}] {uid}: measuring', flush=True)
        result['in_flight'] = uid
        _write_result(result, target)
        try:
            record = run_unit(unit)
        except Exception as error:
            record = {'status': 'oom' if is_oom(error) else 'error',
                      'error': f'{type(error).__name__}: {error}'}
            failed.append({'impl': unit['name'], 'T': unit['T'], 'B': unit['B']})
            free_mem()
        record.update(impl=unit['name'], base_implementation=unit['impl'], T=unit['T'], B=unit['B'])
        result['units'][uid] = record
        result.pop('in_flight', None)
        _write_result(result, target)
        if XLA and record['status'] != 'ok':
            raise RuntimeError('XLA runtime failed; recorded results are preserved')
    result['status'] = 'complete' if all(record['status'] == 'ok' for record in result['units'].values()) else 'partial'
    if (not passed or result['unavailable'] or result['projected_skips']
            or any(not gate.get('pass', False) for gate in result['gates'].values())):
        result['status'] = 'partial'
    _write_result(result, target)
    return result


configure()
