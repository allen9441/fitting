#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saxs_autofit.py -- Automated batch SAXS form-factor fitting pipeline.

Python port + automation of platonicFFexplorer.m (and the run_*.m launchers)
originally written for MATLAB.  Everything that used to require an
interactive GUI session per file is done here in one non-interactive batch
run:

    1. Scan a folder of .dat files. Each file is ONE SAXS curve (q, I[, sigma])
       captured at one instant.  Filenames encode time as
           [Title]_[batch, 01,02,...]_[second, 0..149].dat
       so absolute time (s) = (batch-1)*BATCH_LEN + second.
    2. Group files by Title (one experiment/sample per group).
    3. For each group, auto-detect which crystal habit (sphere / the five
       Platonic solids / rhombic dodecahedron == ZIF-8 {110} habit) best
       explains the data, using a reference frame (default: the last, i.e.
       most-crystallized, frame).
    4. Fit every frame in the group with that shape.  Population 1 (a single
       Schulz-polydisperse population) is always fit.  Population 2 -- a
       second, independent Schulz population sharing the same shape -- is
       automatically tried as well and is KEPT only when it explains the
       data meaningfully better, which is exactly the case described for
       the crystallization transition region (data with two slopes in the
       mid-q range = a coexisting not-yet-crystallized + already-crystallized
       particle population).
    5. Write one CSV of fitted parameters vs. time per group, and PNG plots
       of every parameter vs. time.

Physics/algorithm notes (all ported from platonicFFexplorer.m):
  - Model: I(q) = sum_k A_k * S(q; R0_k, PD_k) + B
  - S(q;R0,PD) = Schulz-size-distribution average of (R/R0)^6 * P0(q R),
    P0 = orientation-averaged, normalized polyhedron form factor.
  - P0 is precomputed once per shape on a fixed x=qR grid ("master curve"),
    using adaptive-direction-count Monte-Carlo orientation averaging below
    x=xcut and the *exact* Porod law c/x^4 (c = 2*pi*S/V^2, S,V the exact
    polyhedron surface/volume) above it, blended smoothly across the seam.
    This master curve is cached (in memory and on disk) and reused for every
    frame and every population, exactly like the MATLAB code precomputes it
    once per GUI session.
  - R0_k, PD_k are found by a Nelder-Mead multistart (three starting
    multipliers 0.3/1/3 per free R0, as in the original); A_k, B are solved
    at every trial point by non-negative least squares (variable projection),
    also as in the original.
  - Parameter uncertainties come from a numeric Jacobian and
    chi2_red * pinv(J'WJ), matching computeErrors() in the .m file.

Requires: numpy, scipy, pandas, matplotlib.

Usage:
    python saxs_autofit.py /path/to/dat_folder [options]

Run with -h for all options.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import itertools
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# dataviz palette (see codes/saxs_autofit.py docstring / dataviz skill) --
# used only for the static PNG plots produced at the end of the pipeline.
# ============================================================================
COL_POP1 = "#2a78d6"     # categorical slot 1 (blue)
COL_POP2 = "#eb6834"     # categorical slot 2 (orange)
COL_TOTAL = "#0b0b0b"    # primary ink
COL_GRID = "#e1e0d9"
COL_MUTED = "#898781"
COL_SECONDARY = "#52514e"

PHI = (1 + np.sqrt(5)) / 2
SHAPES = ["sphere", "tetra", "cube", "octa", "dodeca", "icosa", "rhombic"]
SHAPE_ALIASES = {
    "sph": "sphere", "ball": "sphere",
    "tetrahedron": "tetra",
    "hexahedron": "cube",
    "octahedron": "octa",
    "dodecahedron": "dodeca",
    "icosahedron": "icosa",
    "rd": "rhombic", "rhombicdodecahedron": "rhombic",
    "zif8": "rhombic", "zif-8": "rhombic",
}

EPS = np.finfo(float).eps


def canon_shape(name: str) -> str:
    name = name.strip().lower()
    return SHAPE_ALIASES.get(name, name)


# ============================================================================
# 1. Polyhedron geometry (vertices -> convex hull -> triangulated faces)
# ============================================================================

def _vertices(shape: str) -> np.ndarray:
    if shape == "tetra":
        V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    elif shape == "cube":
        V = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    elif shape == "octa":
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    elif shape == "dodeca":
        V = list(itertools.product([-1.0, 1.0], repeat=3))
        for s1, s2 in itertools.product([-1.0, 1.0], [-1.0, 1.0]):
            V.append((0.0, s1 / PHI, s2 * PHI))
            V.append((s1 / PHI, s2 * PHI, 0.0))
            V.append((s1 * PHI, 0.0, s2 / PHI))
        V = np.array(V, dtype=float)
    elif shape == "icosa":
        V = []
        for s1, s2 in itertools.product([-1.0, 1.0], [-1.0, 1.0]):
            V.append((0.0, s1, s2 * PHI))
            V.append((s1, s2 * PHI, 0.0))
            V.append((s2 * PHI, 0.0, s1))
        V = np.array(V, dtype=float)
    elif shape == "rhombic":
        V = list(itertools.product([-1.0, 1.0], repeat=3))
        V += [(2.0, 0, 0), (-2.0, 0, 0), (0, 2.0, 0), (0, -2.0, 0), (0, 0, 2.0), (0, 0, -2.0)]
        V = np.array(V, dtype=float)
    else:
        raise ValueError(f'Unknown shape "{shape}". Valid: sphere, ' + ", ".join(s for s in SHAPES if s != "sphere"))
    return V


def build_geom(shape: str):
    """Return (faces, Vol) with faces = list of (p[3x3], n[3]) triangles,
    normalized so the equivalent-sphere volume is 1 (R_eq = 1)."""
    V = _vertices(shape)
    vol0 = ConvexHull(V).volume
    V = V * ((4 * np.pi / 3) / vol0) ** (1.0 / 3.0)
    hull = ConvexHull(V)
    Vol = hull.volume
    c = V.mean(axis=0)
    faces = []
    for simplex in hull.simplices:
        p = V[simplex]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if np.dot(n, p.mean(axis=0) - c) < 0:
            p = p[[0, 2, 1]]
            n = -n
        faces.append((p, n))
    return faces, Vol


def _mysinc(x: np.ndarray) -> np.ndarray:
    s = np.ones_like(x)
    nz = x != 0
    s[nz] = np.sin(x[nz]) / x[nz]
    return s


def polyhedron_amplitude(Q: np.ndarray, faces) -> np.ndarray:
    """Exact single-orientation scattering amplitude F(Q) for a convex
    polyhedron built from triangular faces (analytic polygon FT)."""
    N = Q.shape[0]
    q2 = np.sum(Q ** 2, axis=1)
    F = np.zeros(N, dtype=complex)
    for p, n in faces:
        K = p.shape[0]
        qn = Q @ n
        qt2 = q2 - qn ** 2
        If = np.zeros(N, dtype=complex)
        for e in range(K):
            a = p[e]
            b = p[(e + 1) % K]
            t = b - a
            L = np.linalg.norm(t)
            m = np.cross(t / L, n)
            qm = Q @ m
            psia = Q @ a
            psib = Q @ b
            If = If + qm * L * np.exp(1j * (psia + psib) / 2) * _mysinc((psib - psia) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            Iface = -1j * If / qt2
        small = qt2 < (1e-8 * np.maximum(q2, 1e-30))
        if np.any(small):
            dpl = p[0] @ n
            pr = np.roll(p, -1, axis=0)
            va = 0.5 * np.sum(np.cross(p, pr), axis=0)
            area = abs(va @ n)
            Iface = Iface.copy()
            Iface[small] = np.exp(1j * qn[small] * dpl) * area
        F = F + (-1j) * qn * Iface / q2
    return F


def fib_sphere(N: int) -> np.ndarray:
    i = np.arange(N) + 0.5
    ph = np.arccos(1 - 2 * i / N)
    th = np.pi * (1 + np.sqrt(5)) * i
    return np.column_stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])


# ============================================================================
# 2. Master (orientation-averaged, normalized) form factor curve, per shape
# ============================================================================

@dataclass
class Master:
    shape: str
    xgrid: np.ndarray
    P0: np.ndarray
    cPorod: float
    RgOverR: float
    xgmax: float = field(init=False)

    def __post_init__(self):
        self.xgmax = float(self.xgrid.max())


def _porod_const_sphere() -> float:
    return 4.5  # 2*pi*S/V^2 for R_eq = 1


def compute_master(shape: str, Norient0=2500, NorMax=40000, xfine=60.0, xcut=100.0,
                    xmax=600.0, Nx1=800, Nx2=80, Nx3=200, verbose=True) -> Master:
    shape = canon_shape(shape)
    xgrid = np.unique(np.concatenate([
        np.linspace(0.0, xfine, Nx1),
        np.linspace(xfine, xcut, Nx2),
        np.logspace(np.log10(xcut), np.log10(xmax), Nx3),
    ]))
    Nx = xgrid.size

    if shape == "sphere":
        P0 = np.ones(Nx)
        x = xgrid[1:]
        P0[1:] = (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
        cPorod = _porod_const_sphere()
        m = Master(shape, xgrid, P0, cPorod, _rg_over_r(xgrid, P0))
        return m

    if verbose:
        print(f'Computing master form factor for "{shape}" (once) ...')
    faces, Vol = build_geom(shape)
    Surf = sum(0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) for p, _ in faces)
    cPorod = 2 * np.pi * Surf / Vol ** 2

    lv = Norient0 * 4.0 ** np.arange(0, 7)
    lv = lv[lv <= NorMax]
    if lv.size == 0:
        lv = np.array([Norient0])
    D = [fib_sphere(int(n)) for n in lv]

    P0 = np.ones(Nx)
    nb = 0
    if verbose:
        sys.stdout.write("   [")
        sys.stdout.flush()
    for k in range(1, Nx):
        xk = xgrid[k]
        if xk > xcut:
            P0[k] = cPorod / xk ** 4
        else:
            need = Norient0 * (xk / 30.0) ** 2
            j = int(np.searchsorted(lv, need))
            if j >= len(lv):
                j = len(lv) - 1
            F = polyhedron_amplitude(xk * D[j], faces)
            P0[k] = np.mean(np.abs(F) ** 2) / Vol ** 2
        if verbose and (k / Nx * 40) > nb:
            sys.stdout.write(".")
            sys.stdout.flush()
            nb += 1
    if verbose:
        print("]")

    m = (xgrid > 0.7 * xcut) & (xgrid <= xcut)
    tt = (xgrid[m] - 0.7 * xcut) / (0.3 * xcut)
    sb = tt ** 2 * (3 - 2 * tt)
    P0[m] = (1 - sb) * P0[m] + sb * cPorod / xgrid[m] ** 4

    RgOverR = _rg_over_r(xgrid, P0)
    if verbose:
        print(f"   done.  Rg/R = {RgOverR:.4f}  (monodisperse)   Porod const 2*pi*S/V^2 = {cPorod:.4f}")
    return Master(shape, xgrid, P0, cPorod, RgOverR)


def _rg_over_r(xgrid: np.ndarray, P0: np.ndarray) -> float:
    kk = np.arange(1, 9)  # xgrid[1:9], matches MATLAB kk=(2:9) 1-indexed
    yy = 3 * (1 - P0[kk]) / xgrid[kk] ** 2
    pp = np.polyfit(xgrid[kk] ** 2, yy, 1)
    return float(np.sqrt(max(pp[1], 0.0)))


def master_lookup(X: np.ndarray, master: Master) -> np.ndarray:
    shp = X.shape
    Xf = X.ravel()
    Pv = np.interp(Xf, master.xgrid, master.P0, left=1.0, right=np.nan)
    hi = Xf > master.xgmax
    if np.any(hi):
        Pv[hi] = master.cPorod / Xf[hi] ** 4
    Pv[Xf <= master.xgrid[0]] = 1.0
    Pv = np.nan_to_num(Pv, nan=0.0)
    return Pv.reshape(shp)


# ============================================================================
# 3. Schulz polydispersity quadrature + population signal S(q; R0, PD)
# ============================================================================

def schulz_nodes(R0: float, PD: float, Nr: int = 201):
    if PD < 1e-4:
        return np.array([R0]), np.array([1.0])
    z = 1.0 / PD ** 2 - 1.0
    mu = (z + 7) * R0 / (z + 1)
    sd = np.sqrt(z + 7) * R0 / (z + 1)
    Rhi = max(mu + 10 * sd, R0 * (1 + 8 * PD))
    Rlo = max(R0 * 1e-4, R0 * (1 - 8 * PD))
    Nr = max(Nr, 201)
    Rn = np.linspace(Rlo, Rhi, Nr)
    lf = z * np.log(Rn) - (z + 1) * Rn / R0
    fpdf = np.exp(lf - lf.max())
    dR = np.diff(Rn)
    tw = np.zeros(Nr)
    tw[0] = dR[0] / 2
    tw[-1] = dR[-1] / 2
    tw[1:-1] = (dR[:-1] + dR[1:]) / 2
    W = tw * fpdf
    W = W / W.sum()
    return Rn, W


def shape_sig(qv: np.ndarray, R0: float, PD: float, master: Master, Nr: int = 201) -> np.ndarray:
    Rn, W = schulz_nodes(R0, PD, Nr)
    X = np.outer(qv, Rn)
    Pv = master_lookup(X, master)
    return Pv @ (W * (Rn / R0) ** 6)


def rg_apparent(R0: float, PD: float, master: Master, Nr: int = 201):
    Rn, W = schulz_nodes(R0, PD, Nr)
    wt = W * (Rn / R0) ** 6
    S0 = wt.sum()
    Rgapp = master.RgOverR * np.sqrt((wt * Rn ** 2).sum() / S0)
    return Rgapp, S0


# ============================================================================
# 4. Data loading (.dat reader + Guinier seed), mirroring readDataFile/onLoad
# ============================================================================

def read_data_file(path: str) -> np.ndarray:
    rows = []
    ncols = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip().replace("\r", "")
            if not s or s[0] in "#%":
                continue
            toks = [t for t in re.split(r"[\s,;]+", s) if t]
            if len(toks) < 2:
                continue
            try:
                v = [float(t) for t in toks]
            except ValueError:
                continue
            rows.append(v)
            ncols.append(len(v))
    if not rows:
        raise ValueError(f"No numeric data rows found in: {path}")
    ncols = np.array(ncols)
    m = np.bincount(ncols).argmax()
    sel = [r for r, n in zip(rows, ncols) if n == m]
    return np.array(sel, dtype=float)


def load_frame(path: str):
    d = read_data_file(path)
    d = d[np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])]
    d = d[(d[:, 0] > 0) & (d[:, 1] > 0)]
    d = d[np.argsort(d[:, 0])]
    if d.shape[0] < 5:
        raise ValueError(f"Too few usable data points in: {path}")
    if d.shape[1] >= 3:
        s3 = d[:, 2]
        good = np.isfinite(s3) & (s3 > 0)
        if good.mean() >= 0.5:
            sig = s3.copy()
            sig[~good] = 0.03 * d[~good, 1]
            err_mode = "file"
        else:
            sig = 0.03 * d[:, 1]
            err_mode = "3pct"
    else:
        sig = 0.03 * d[:, 1]
        err_mode = "3pct"
    sig = np.maximum(sig, EPS)
    return d[:, 0], d[:, 1], sig, err_mode


def guinier_R0(q: np.ndarray, I: np.ndarray, sig: np.ndarray, RgOverR: float) -> float:
    ok = I > 3 * sig
    if ok.sum() < 8:
        ok = np.ones_like(q, dtype=bool)
    idx = np.flatnonzero(ok)
    i0 = idx[0]
    n = q.size
    span = max(round(0.15 * n), 6)
    m = np.arange(i0, min(i0 + span, n))
    if m.size < 4:
        m = np.arange(0, min(10, n))
    p = np.polyfit(q[m] ** 2, np.log(I[m]), 1)
    if p[0] < 0:
        Rg = np.sqrt(-3 * p[0])
    else:
        Rg = 1.0 / np.median(q)
    R0g = Rg / RgOverR
    if not np.isfinite(R0g) or R0g <= 0:
        R0g = 1.0 / np.median(q) / RgOverR
    return float(R0g)


# ============================================================================
# 5. Variable-projection fit: NNLS for {A_k, B}, Nelder-Mead multistart for
#    {R0_k, PD_k}, exactly as fitObjGen / solveLin / onFit in the .m file.
# ============================================================================

PDMAX = 1.0


def _invlogit(p, pmax=PDMAX):
    p = min(max(p, 1e-4), pmax - 1e-4)
    return -np.log(pmax / p - 1)


def _pack_u(R0s, PDs):
    u = []
    for R0, PD in zip(R0s, PDs):
        u.append(np.log10(R0))
        u.append(_invlogit(PD))
    return np.array(u)


def _unpack_u(u, npop):
    R0s = np.array([10 ** u[2 * k] for k in range(npop)])
    PDs = np.array([PDMAX / (1 + np.exp(-u[2 * k + 1])) for k in range(npop)])
    return R0s, PDs


def solve_lin(S: np.ndarray, y: np.ndarray, w: np.ndarray):
    sw = np.sqrt(w)
    M = np.column_stack([S, np.ones_like(y)])
    Mw = M * sw[:, None]
    yw = y * sw
    try:
        coef, _ = nnls(Mw, yw)
    except Exception:
        coef = np.linalg.lstsq(Mw, yw, rcond=None)[0]
        coef = np.maximum(coef, 0.0)
    return coef[:-1], coef[-1]


def _all_sig(q, R0s, PDs, master, Nr):
    return np.column_stack([shape_sig(q, R0, PD, master, Nr) for R0, PD in zip(R0s, PDs)])


def _fit_objective(u, q, y, w, npop, master, Nr):
    R0s, PDs = _unpack_u(u, npop)
    S = _all_sig(q, R0s, np.maximum(PDs, 0), master, Nr)
    A, B = solve_lin(S, y, w)
    resid = y - (S @ A + B)
    chi2 = float(np.sum(w * resid ** 2))
    return chi2 if np.isfinite(chi2) else 1e300


def model_of(theta, q, npop, master, Nr):
    R0s = theta[0:2 * npop:2]
    PDs = theta[1:2 * npop:2]
    As = theta[2 * npop:3 * npop]
    B = theta[3 * npop]
    y = np.full_like(q, B, dtype=float)
    for k in range(npop):
        y = y + As[k] * shape_sig(q, R0s[k], max(PDs[k], 0.0), master, Nr)
    return y


@dataclass
class FitResult:
    npop: int
    R0: np.ndarray
    PD: np.ndarray
    A: np.ndarray
    B: float
    seR0: np.ndarray
    sePD: np.ndarray
    seA: np.ndarray
    seB: float
    Rg: np.ndarray
    seRg: np.ndarray
    RgMono: np.ndarray
    frac: np.ndarray
    I0: float
    RgTot: float
    chi2red: float
    dof: int
    npts: int
    pd_at_ceiling: bool


def fit_shape(q, y, w, npop: int, master: Master, R0seed, PDseed, Nr=201) -> FitResult:
    mlt = [0.3, 1.0, 3.0]
    ncomb = 3 ** npop
    best_chi = np.inf
    best_u = None
    opt = dict(method="Nelder-Mead",
               options=dict(xatol=1e-5, fatol=1e-5, maxiter=4000, maxfev=4000))
    for ii in range(ncomb):
        R0try = list(R0seed)
        t = ii
        for j in range(npop):
            R0try[j] = R0seed[j] * mlt[t % 3]
            t //= 3
        u0 = _pack_u(R0try, PDseed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_fit_objective, u0, args=(q, y, w, npop, master, Nr), **opt)
        if np.isfinite(res.fun) and res.fun < best_chi:
            best_chi = res.fun
            best_u = res.x
    if best_u is None:
        raise RuntimeError("Fit did not converge to any finite solution.")

    R0s, PDs = _unpack_u(best_u, npop)
    S = _all_sig(q, R0s, PDs, master, Nr)
    A, B = solve_lin(S, y, w)

    theta = np.concatenate([np.ravel(np.column_stack([R0s, PDs])), A, [B]])
    np_ = theta.size
    f0 = model_of(theta, q, npop, master, Nr)
    dof = max(q.size - np_, 1)
    chi2 = float(np.sum(w * (y - f0) ** 2))
    chi2red = chi2 / dof

    J = np.zeros((q.size, np_))
    for j in range(np_):
        h = max(1e-4 * abs(theta[j]), 1e-7)
        tk = theta.copy()
        tk[j] += h
        J[:, j] = (model_of(tk, q, npop, master, Nr) - f0) / h
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = chi2red * np.linalg.pinv(J.T @ (w[:, None] * J))
    se = np.sqrt(np.maximum(np.diag(C), 0.0))
    seR0 = se[0:2 * npop:2]
    sePD = se[1:2 * npop:2]
    seA = se[2 * npop:3 * npop]
    seB = se[3 * npop]

    Rg = np.zeros(npop)
    S0v = np.zeros(npop)
    seRg = np.zeros(npop)
    RgMono = master.RgOverR * R0s
    for k in range(npop):
        Rg[k], S0v[k] = rg_apparent(R0s[k], PDs[k], master, Nr)
        seRg[k] = Rg[k] / R0s[k] * seR0[k]

    wI = A * S0v
    I0 = float(wI.sum())
    if I0 > 0:
        RgTot = float(np.sqrt(np.sum(wI * Rg ** 2) / I0))
        frac = 100 * wI / I0
    else:
        RgTot = np.nan
        frac = np.full(npop, np.nan)

    pd_ceiling = bool(np.any(PDs > 0.97 * PDMAX))

    # Stable population labeling: sort by R0 ascending so "population 1" is
    # always the smaller (not-yet-crystallized) population and "population 2"
    # the larger (already-crystallized) one, consistently across frames --
    # the nonlinear solver has no notion of population identity on its own.
    if npop > 1:
        order = np.argsort(R0s)
        R0s, PDs, A = R0s[order], PDs[order], A[order]
        seR0, sePD, seA = seR0[order], sePD[order], seA[order]
        Rg, seRg, RgMono, frac = Rg[order], seRg[order], RgMono[order], frac[order]

    return FitResult(npop, R0s, PDs, A, B, seR0, sePD, seA, seB, Rg, seRg, RgMono,
                      frac, I0, RgTot, chi2red, dof, q.size, pd_ceiling)


# ============================================================================
# 6. Filename parsing: [Title]_[batch]_[second].dat -> absolute time (s)
# ============================================================================

FNAME_RE = re.compile(r"^(?P<title>.+)_(?P<batch>\d+)_(?P<sec>\d+)\.dat$", re.IGNORECASE)


def parse_filename(path: str, batch_len: int):
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    title = m.group("title")
    batch = int(m.group("batch"))
    sec = int(m.group("sec"))
    t = (batch - 1) * batch_len + sec
    return title, batch, sec, t


def group_files(folder: str, batch_len: int):
    # Recursive: real datasets are often laid out one subfolder per batch
    # (e.g. data/W100_1/W100_1_0.dat ... data/W100_7/W100_7_149.dat), and the
    # batch/second are parsed from the filename itself regardless of which
    # subfolder it sits in, so scanning recursively lets a whole multi-batch
    # experiment be pointed at with a single input_dir.
    files = sorted(glob.glob(os.path.join(folder, "**", "*.dat"), recursive=True))
    groups: dict[str, list] = {}
    skipped = []
    for f in files:
        parsed = parse_filename(f, batch_len)
        if parsed is None:
            skipped.append(f)
            continue
        title, batch, sec, t = parsed
        groups.setdefault(title, []).append((t, batch, sec, f))
    for title in groups:
        groups[title].sort(key=lambda r: r[0])
    return groups, skipped


# ============================================================================
# 7. Automatic crystal-system (shape) selection
# ============================================================================

def auto_select_shape(q, I, sig, candidate_shapes, master_cache, cache_dir=None, Nr=201, verbose=True):
    scores = {}
    for shape in candidate_shapes:
        master = get_master(shape, master_cache, cache_dir, verbose=verbose)
        R0g = guinier_R0(q, I, sig, master.RgOverR)
        w = 1.0 / sig ** 2
        try:
            fr = fit_shape(q, I, w, 1, master, [R0g], [0.05], Nr)
            scores[shape] = fr.chi2red
        except Exception as exc:
            if verbose:
                print(f"   shape {shape}: fit failed ({exc})")
            scores[shape] = np.inf
    best = min(scores, key=scores.get)
    if verbose:
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        print("   shape candidates (reduced chi^2, lower is better):")
        for s, c in ranked:
            print(f"     {s:8s}  {c:.4g}" + ("   <-- selected" if s == best else ""))
    return best, scores


# ============================================================================
# 8. Master-curve caching (disk .npz + in-memory) so it is computed once per
#    shape no matter how many groups/frames use it.
# ============================================================================

def get_master(shape: str, cache: dict, cache_dir: str | None = None, verbose=True) -> Master:
    shape = canon_shape(shape)
    if shape in cache:
        return cache[shape]
    npz_path = os.path.join(cache_dir, f"master_{shape}.npz") if cache_dir else None
    if npz_path and os.path.exists(npz_path):
        d = np.load(npz_path)
        m = Master(shape, d["xgrid"], d["P0"], float(d["cPorod"]), float(d["RgOverR"]))
        cache[shape] = m
        return m
    m = compute_master(shape, verbose=verbose)
    cache[shape] = m
    if npz_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(npz_path, xgrid=m.xgrid, P0=m.P0, cPorod=m.cPorod, RgOverR=m.RgOverR)
    return m


def _save_master_npz(m: Master, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, f"master_{m.shape}.npz"),
              xgrid=m.xgrid, P0=m.P0, cPorod=m.cPorod, RgOverR=m.RgOverR)


def warm_masters(shapes, master_cache: dict, cache_dir: str | None, jobs: int, verbose=True):
    """Precompute (and cache, in memory + on disk) the master curve for every
    shape in `shapes` that isn't already available, in parallel across
    `jobs` worker processes when there is more than one to do. This is the
    single biggest win from multi-core: computing the 7 candidate shapes'
    master curves (the expensive orientation-averaging step) is completely
    independent per shape."""
    todo = []
    for s in shapes:
        s = canon_shape(s)
        if s in master_cache:
            continue
        npz_path = os.path.join(cache_dir, f"master_{s}.npz") if cache_dir else None
        if npz_path and os.path.exists(npz_path):
            get_master(s, master_cache, cache_dir, verbose=False)
            continue
        todo.append(s)
    if not todo:
        return
    if jobs <= 1 or len(todo) <= 1:
        for s in todo:
            get_master(s, master_cache, cache_dir, verbose=verbose)
        return

    nworkers = min(jobs, len(todo))
    if verbose:
        print(f"Computing master form factors for {len(todo)} shape(s) using "
              f"{nworkers} worker process(es): {', '.join(todo)}")
    with cf.ProcessPoolExecutor(max_workers=nworkers) as ex:
        futs = {ex.submit(compute_master, s, verbose=False): s for s in todo}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            m = fut.result()
            master_cache[s] = m
            if cache_dir:
                _save_master_npz(m, cache_dir)
            if verbose:
                print(f"   done: {s:8s}  Rg/R = {m.RgOverR:.4f}   Porod const = {m.cPorod:.4f}")


# ============================================================================
# 9. Per-frame fitting with automatic Population-2 (transition region) detection
# ============================================================================

def fit_frame(q, I, sig, master: Master, max_npop: int, pop2_rel_improve: float, Nr=201):
    w = 1.0 / sig ** 2
    R0g = guinier_R0(q, I, sig, master.RgOverR)

    fr1 = fit_shape(q, I, w, 1, master, [R0g], [0.05], Nr)
    best = fr1

    if max_npop >= 2:
        try:
            fr2 = fit_shape(q, I, w, 2, master, [R0g, R0g / 4.0], [0.05, 0.20], Nr)
            improves = (fr2.chi2red < pop2_rel_improve * fr1.chi2red)
            # A population's PD railed at the ceiling only invalidates the fit
            # (per the .m file's PD-ceiling warning) if that population carries
            # real weight. A negligible (<5% of I(0)) population pinned at the
            # ceiling is just noise soaked up by an extra degree of freedom --
            # it must not veto an otherwise excellent, well-determined fit
            # (e.g. a genuine transition-region frame where the dominant
            # population is fine but a tiny secondary one is unconstrained).
            dominant_ceiling = any(
                fr2.PD[k] > 0.97 * PDMAX and np.isfinite(fr2.frac[k]) and fr2.frac[k] > 5.0
                for k in range(fr2.npop)
            )
            if improves and not dominant_ceiling:
                best = fr2
        except Exception:
            pass

    return best


# ============================================================================
# 10. Pipeline driver
# ============================================================================

def _pad(vals, npop_used, npop_slot, fill=np.nan):
    """Return vals[npop_slot] if that population was fit, else fill."""
    if npop_slot < npop_used:
        return vals[npop_slot]
    return fill


def _process_frame(t, batch, sec, path, master: Master, max_npop, pop2_rel_improve, Nr):
    """Fit one .dat file. Module-level and picklable (only plain data / the
    Master dataclass go in and out) so it can run in a worker process."""
    fname = os.path.basename(path)
    try:
        q, I, sig, err_mode = load_frame(path)
        fr = fit_frame(q, I, sig, master, max_npop, pop2_rel_improve, Nr)
        row = dict(
            file=fname, time_s=t, batch=batch, second=sec, shape=master.shape,
            npop_used=fr.npop, err_mode=err_mode,
            chi2red=fr.chi2red, npts=fr.npts, dof=fr.dof,
            R0_1=fr.R0[0], seR0_1=fr.seR0[0], PD_1=fr.PD[0], sePD_1=fr.sePD[0],
            A_1=fr.A[0], seA_1=fr.seA[0], Rg_1=fr.Rg[0], seRg_1=fr.seRg[0],
            RgMono_1=fr.RgMono[0], frac_1=fr.frac[0],
            R0_2=_pad(fr.R0, fr.npop, 1), seR0_2=_pad(fr.seR0, fr.npop, 1),
            PD_2=_pad(fr.PD, fr.npop, 1), sePD_2=_pad(fr.sePD, fr.npop, 1),
            A_2=_pad(fr.A, fr.npop, 1), seA_2=_pad(fr.seA, fr.npop, 1),
            Rg_2=_pad(fr.Rg, fr.npop, 1), seRg_2=_pad(fr.seRg, fr.npop, 1),
            RgMono_2=_pad(fr.RgMono, fr.npop, 1), frac_2=_pad(fr.frac, fr.npop, 1),
            B=fr.B, seB=fr.seB, I0=fr.I0, RgTot=fr.RgTot,
            pd_at_ceiling=fr.pd_at_ceiling, error="",
        )
    except Exception as exc:
        row = dict(file=fname, time_s=t, batch=batch, second=sec, shape=master.shape,
                   error=str(exc))
    return row


def _process_frame_task(task):
    return _process_frame(*task)


def run_group(title: str, frames, shape: str, master_cache, out_dir: str,
              length_unit: str, max_npop: int, pop2_rel_improve: float,
              cache_dir: str | None, Nr=201, verbose=True, jobs: int = 1):
    master = get_master(shape, master_cache, cache_dir, verbose=verbose)
    n = len(frames)
    tasks = [(t, batch, sec, path, master, max_npop, pop2_rel_improve, Nr)
             for (t, batch, sec, path) in frames]

    rows_by_index = {}
    if jobs <= 1 or n <= 1:
        for i, task in enumerate(tasks):
            row = _process_frame_task(task)
            rows_by_index[i] = row
            if verbose:
                status = "ok" if not row.get("error") else f"FAILED: {row['error']}"
                print(f"   [{i + 1}/{n}] t={task[0]:>6} s  {row['file']}  -> {status}")
    else:
        nworkers = min(jobs, n)
        if verbose:
            print(f"   fitting {n} frame(s) using {nworkers} worker process(es) ...")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=nworkers) as ex:
            futs = {ex.submit(_process_frame_task, task): i for i, task in enumerate(tasks)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                row = fut.result()
                rows_by_index[i] = row
                done += 1
                if verbose:
                    status = "ok" if not row.get("error") else f"FAILED: {row['error']}"
                    print(f"   [{done}/{n}] t={tasks[i][0]:>6} s  {row['file']}  -> {status}")

    rows = [rows_by_index[i] for i in range(n)]
    df = pd.DataFrame(rows)
    grp_dir = os.path.join(out_dir, _safe(title))
    os.makedirs(grp_dir, exist_ok=True)
    csv_path = os.path.join(grp_dir, f"{_safe(title)}_fit_results.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"   wrote {csv_path}")

    plot_group(df, title, grp_dir, length_unit, verbose=verbose)
    return df


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "group"


def plot_group(df: pd.DataFrame, title: str, out_dir: str, length_unit: str, verbose=True):
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    ok = df[df["error"].fillna("") == ""].copy() if "error" in df.columns else df.copy()
    if ok.empty:
        if verbose:
            print("   no successful fits to plot.")
        return
    t = ok["time_s"].to_numpy()
    has2 = ok["npop_used"] >= 2

    def _style_ax(ax, ylabel):
        ax.set_xlabel("time (s)", color=COL_SECONDARY)
        ax.set_ylabel(ylabel, color=COL_SECONDARY)
        ax.grid(True, color=COL_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COL_MUTED)
        ax.tick_params(colors=COL_SECONDARY)

    panels = [
        ("R0", f"R0 ({length_unit})"),
        ("PD", "PD (sigma / R0)"),
        ("Rg", f"Rg ({length_unit})"),
        ("A", "A (scale)"),
        ("frac", "fraction of I(0) (%)"),
        ("chi2red", "reduced chi^2"),
    ]
    for key, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        if key == "chi2red":
            ax.plot(t, ok["chi2red"], "-o", color=COL_TOTAL, markersize=4, linewidth=1.5,
                    label="reduced chi^2")
        else:
            ax.plot(t, ok[f"{key}_1"], "-o", color=COL_POP1, markersize=4, linewidth=1.5,
                    label="population 1")
            if has2.any():
                y2 = ok[f"{key}_2"].to_numpy()
                ax.plot(t[has2], y2[has2], "o", color=COL_POP2, markersize=5,
                        label="population 2 (transition region)")
            if key == "Rg":
                # I(0)-weighted average across populations, i.e. RgTot from
                # the .m file's computeErrors(): Rg_tot = sqrt(sum(A_k*S0_k*Rg_k^2)
                # / sum(A_k*S0_k)). Equals Rg_1 wherever only population 1 is
                # active, and blends smoothly into the weighted value across
                # the transition region where population 2 turns on.
                ax.plot(t, ok["RgTot"], "--", color=COL_TOTAL, linewidth=1.6,
                        label="weighted average (total)")
        ax.set_title(f"{title}: {ylabel} vs time", color="#0b0b0b")
        _style_ax(ax, ylabel)
        if key != "chi2red":
            ax.legend(frameon=False, fontsize=8, labelcolor=COL_SECONDARY)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{_safe(title)}_{key}_vs_time.png"), dpi=150)
        plt.close(fig)

    if verbose:
        print(f"   wrote plots to {plots_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch-fit SAXS .dat files (Guinier/small-angle region) to "
                    "polyhedral (Platonic solid / rhombic-dodecahedron / sphere) "
                    "form factors, auto-detecting the crystal habit and the "
                    "single- vs. two-population (crystallization transition) model, "
                    "and plot the resulting parameters vs. time.")
    ap.add_argument("input_dir", help="folder containing the .dat files")
    ap.add_argument("--out", default=None,
                     help="output folder (default: <input_dir>/saxs_fit_output)")
    ap.add_argument("--shapes", default=",".join(SHAPES),
                     help=f"comma-separated candidate shapes to test (default: all = {','.join(SHAPES)})")
    ap.add_argument("--force-shape", default=None,
                     help="skip auto-detection and use this shape for every group")
    ap.add_argument("--shape-frame", default="last", choices=["first", "last"],
                     help="which frame in each group to use for shape auto-detection (default: last)")
    ap.add_argument("--batch-len", type=int, default=150,
                     help="seconds per batch, i.e. per-file second index range (default: 150, matching 0..149)")
    ap.add_argument("--max-npop", type=int, default=2, choices=[1, 2],
                     help="1 = single population only; 2 = also try population 2 for the "
                          "crystallization transition region (default: 2)")
    ap.add_argument("--pop2-threshold", type=float, default=0.9,
                     help="keep the 2-population fit only if its reduced chi^2 is below "
                          "this fraction of the 1-population chi^2 (default: 0.9, i.e. "
                          "require >=10%% improvement)")
    ap.add_argument("--length-unit", default="A", help="label for R0/Rg axis units (default: A)")
    ap.add_argument("--nr", type=int, default=201, help="Schulz quadrature nodes (default: 201)")
    ap.add_argument("--cache-dir", default=None,
                     help="folder to cache computed master curves (default: <out>/.master_cache)")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                     help="number of worker processes for multi-core speedup (default: 1 = "
                          "sequential). Used both to compute several candidate shapes' master "
                          "curves in parallel during auto-detection, and to fit multiple frames "
                          "at once. Pass -1 (or 0) to use all available CPU cores.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    out_dir = args.out or os.path.join(args.input_dir, "saxs_fit_output")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(out_dir, ".master_cache")

    candidate_shapes = [canon_shape(s) for s in args.shapes.split(",") if s.strip()]
    for s in candidate_shapes:
        if s not in SHAPES:
            ap.error(f"Unknown shape '{s}'. Valid: {', '.join(SHAPES)}")

    groups, skipped = group_files(args.input_dir, args.batch_len)
    if not groups:
        print("No .dat files matched the expected [Title]_[batch]_[second].dat naming pattern.")
        return 1
    if skipped and verbose:
        print(f"Skipped {len(skipped)} file(s) that did not match the naming pattern:")
        for f in skipped[:10]:
            print(f"   {os.path.basename(f)}")
        if len(skipped) > 10:
            print(f"   ... and {len(skipped) - 10} more")

    master_cache: dict[str, Master] = {}

    # Precompute master curves up front, in parallel across `jobs` processes
    # when possible -- computing N candidate shapes is embarrassingly
    # parallel, and it is the most expensive one-time cost in the pipeline.
    shapes_needed = [canon_shape(args.force_shape)] if args.force_shape else candidate_shapes
    warm_masters(shapes_needed, master_cache, cache_dir, jobs, verbose=verbose)

    for title, frames in groups.items():
        print(f"\n=== group '{title}': {len(frames)} frame(s), "
              f"t = {frames[0][0]}..{frames[-1][0]} s ===")

        if args.force_shape:
            shape = canon_shape(args.force_shape)
            print(f"   using forced shape: {shape}")
        else:
            ref = frames[-1] if args.shape_frame == "last" else frames[0]
            print(f"   auto-detecting crystal system from frame: {os.path.basename(ref[3])}")
            q, I, sig, _ = load_frame(ref[3])
            shape, scores = auto_select_shape(q, I, sig, candidate_shapes, master_cache,
                                               cache_dir=cache_dir, Nr=args.nr, verbose=verbose)
            print(f"   -> detected crystal system: {shape}")

        run_group(title, frames, shape, master_cache, out_dir, args.length_unit,
                  args.max_npop, args.pop2_threshold, cache_dir, Nr=args.nr, verbose=verbose,
                  jobs=jobs)

    print(f"\nAll results written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
