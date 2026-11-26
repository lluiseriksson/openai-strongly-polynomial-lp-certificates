"""Generate all 16 canonical SPLP-SMALE9-CERT-1.0 certificates and 4 experiment JSONs.

Uses exact integer/Fraction arithmetic for all certificate generation and sweeps:
  - Exp 1: 16-certificate atlas & warm-up timing (4 per mode)
  - Exp 2: Sec. 1.2 circuit imbalance det([[1, u], [u, 1]]) = 1 - u^2 & Newton identity k! det(M) scaling
  - Exp 3: Sec. 9.3 Fig. 5 clipped update error halving trajectory across 6 halving steps (exact Fraction)
  - Exp 4: Sec. 7.2 monotone LCP exact gap decomposition components across 4 canonical LCP/QP-KKT instances
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splp_certifier import PROTOCOL_ID, verify_certificate_dict
from src.splp_exact import (
    dot_frac,
    eval_clipped_operator,
    mat_rank_exact,
    mat_vec_frac,
    newton_det_scaled_int,
)

CERT_DIR = ROOT / "certificates"
RES_DIR = ROOT / "results"


def _fp(f: Fraction) -> List[int]:
    return [f.numerator, f.denominator]


def _fvec(v: List[Fraction]) -> List[List[int]]:
    return [_fp(x) for x in v]


def _fmat(M: List[List[Fraction]]) -> List[List[List[int]]]:
    return [_fvec(row) for row in M]


# ---------------------------------------------------------------------------
# Certificate Builders (4 per Mode = 16 Total)
# ---------------------------------------------------------------------------

def build_mode1_opt(
    cert_id: str,
    A: List[List[Fraction]],
    b: List[Fraction],
    c: List[Fraction],
    x: List[Fraction],
    y: List[Fraction],
) -> Dict[str, Any]:
    opt = dot_frac(c, x)
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "lp_trichotomy_certificate_cert",
        "description": f"Thm. 1 optimal branch ({len(A)}x{len(A[0])}) with exact strict complementarity b-Ax+y > 0 (opt={opt})",
        "payload": {
            "branch": "optimal",
            "m_rows": len(A),
            "n_cols": len(A[0]),
            "A": _fmat(A),
            "b": _fvec(b),
            "c": _fvec(c),
            "witness": {"x": _fvec(x), "y": _fvec(y), "claimed_opt_value": _fp(opt)},
        },
    }


def build_mode1_unb(
    cert_id: str,
    A: List[List[Fraction]],
    b: List[Fraction],
    c: List[Fraction],
    x: List[Fraction],
    r: List[Fraction],
) -> Dict[str, Any]:
    gain = dot_frac(c, r)
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "lp_trichotomy_certificate_cert",
        "description": f"Thm. 1 unbounded branch ({len(A)}x{len(A[0])}) with feasible x & improving recession ray (c^T r={gain})",
        "payload": {
            "branch": "unbounded",
            "m_rows": len(A),
            "n_cols": len(A[0]),
            "A": _fmat(A),
            "b": _fvec(b),
            "c": _fvec(c),
            "witness": {"x": _fvec(x), "r": _fvec(r), "claimed_ray_gain": _fp(gain)},
        },
    }


def build_mode1_inf(
    cert_id: str,
    A: List[List[Fraction]],
    b: List[Fraction],
    c: List[Fraction],
    y: List[Fraction],
) -> Dict[str, Any]:
    rk = mat_rank_exact(A)
    rhs = dot_frac(b, y)
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "lp_trichotomy_certificate_cert",
        "description": f"Thm. 1 infeasible branch ({len(A)}x{len(A[0])}) minimal-support Farkas cert (supp <= 1+rank={1+rk})",
        "payload": {
            "branch": "infeasible",
            "m_rows": len(A),
            "n_cols": len(A[0]),
            "A": _fmat(A),
            "b": _fvec(b),
            "c": _fvec(c),
            "witness": {"y": _fvec(y), "claimed_rank_A": rk, "claimed_farkas_rhs": _fp(rhs)},
        },
    }


def build_mode2_cert(
    cert_id: str,
    N: int,
    C_basis: List[List[Fraction]],
    Cp_basis: List[List[Fraction]],
    prices: List[int],
    M_int: List[List[int]],
) -> Dict[str, Any]:
    k_det = newton_det_scaled_int(M_int)
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "reciprocal_scaling_and_newton_det_cert",
        "description": f"Secs. 1.2/3.1/3.4 N={N} reciprocal scaling Q(yD)=0 & {len(M_int)}x{len(M_int)} division-free Newton k!det={k_det}",
        "payload": {
            "N_dim": N,
            "C_basis": _fmat(C_basis),
            "C_perp_basis": _fmat(Cp_basis),
            "prices_p": prices,
            "newton_int_matrix": M_int,
            "claimed_k_fact_det": k_det,
        },
    }


def build_mode3_cert(
    cert_id: str,
    M: List[List[Fraction]],
    q: List[Fraction],
    z_star: List[Fraction],
    z_probe: List[Fraction],
) -> Dict[str, Any]:
    d = len(M)
    w_star = [v + q[i] for i, v in enumerate(mat_vec_frac(M, z_star))]
    w_probe = [v + q[i] for i, v in enumerate(mat_vec_frac(M, z_probe))]
    dz = [z_probe[i] - z_star[i] for i in range(d)]
    quad = dot_frac(dz, mat_vec_frac(M, dz))
    gap = dot_frac(z_probe, w_probe)
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "monotone_lcp_and_qp_kkt_cert",
        "description": f"Thm. 5/Cor. 6 {d}D monotone LCP & Sec. 7.2 gap decomposition (gap={gap}, quad={quad})",
        "payload": {
            "d_dim": d,
            "M": _fmat(M),
            "q": _fvec(q),
            "z_star": _fvec(z_star),
            "w_star": _fvec(w_star),
            "z_probe": _fvec(z_probe),
            "claimed_probe_gap": _fp(gap),
            "claimed_quad_term": _fp(quad),
        },
    }


def build_mode4_cert(
    cert_id: str,
    owners: List[str],
    actions: List[List[Tuple[Fraction, List[Fraction], List[Fraction]]]],
    gamma: Fraction,
    v: List[Fraction],
    x_query: List[Fraction],
    y_center: List[Fraction],
    H: Fraction,
    u_clipped: List[Fraction],
) -> Dict[str, Any]:
    n = len(owners)
    new_err = max(abs(v[i] - (y_center[i] + u_clipped[i])) for i in range(n))
    return {
        "protocol": PROTOCOL_ID,
        "certificate_id": cert_id,
        "mode": "max_norm_contraction_and_game_cert",
        "description": f"Thms. 8/11 {n}-state game (gamma={gamma}), Bellman fixed point & Eq. (9.3)-(9.5) clipped update",
        "payload": {
            "n_states": n,
            "owners": owners,
            "actions": [
                [{"r": _fp(r), "P": _fvec(P), "B": _fvec(B)} for (r, P, B) in s_acts]
                for s_acts in actions
            ],
            "gamma_contraction": _fp(gamma),
            "v_fixed_point": _fvec(v),
            "x_query": _fvec(x_query),
            "y_center": _fvec(y_center),
            "H_radius": _fp(H),
            "u_clipped": _fvec(u_clipped),
            "claimed_new_err": _fp(new_err),
        },
    }


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Mode 1: Exact Theorem 1 LP Trichotomy (4 certs: 2 optimal, 1 unbounded, 1 infeasible) ---
    # C01: 2D box+cut optimal LP: max 3x1 + 2x2 s.t. x1+x2<=4, 2x1+x2<=6, -x1<=0, -x2<=0
    # Optimum x=(2,2), value 10. Tight rows 0,1 with dual y=(1,1,0,0): A^T y = (1+2, 1+1) = (3,2)=c!
    # Slacks b-Ax = (0,0,2,2), b-Ax+y = (1,1,2,2) > 0 strictly complementary!
    A1 = [
        [Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(1)],
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(-1)],
    ]
    b1 = [Fraction(4), Fraction(6), Fraction(0), Fraction(0)]
    c1 = [Fraction(3), Fraction(2)]
    x1 = [Fraction(2), Fraction(2)]
    y1 = [Fraction(1), Fraction(1), Fraction(0), Fraction(0)]

    # C02: 3D optimal LP: max 5x1 + 4x2 + 3x3 s.t. x1+x2+x3<=5, 2x1+x2<=6, x2+2x3<=5, -x<=0
    # Choose tight rows 0,1,2 -> solve x1+x2+x3=5, 2x1+x2=6, x2+2x3=5:
    # From row 1: x2 = 6 - 2x1. Row 2: 2x3 = 5 - (6 - 2x1) = 2x1 - 1 => x3 = x1 - 1/2.
    # Row 0: x1 + (6 - 2x1) + (x1 - 1/2) = 11/2 != 5 -> linearly dependent! Good check!
    # Instead: rows x1+x2+x3<=6, 2x1+x2<=8, x1+2x3<=7 -> x=(3,2,1)? Check: 3+2+1=6, 6+2=8, 3+2=5<=7.
    # Tight rows 0,1,2 with x=(3,2,2): b=(7,8,7)! Let's set c = A^T y with y=(1,2,1,0,0,0):
    # c = (1+4+1, 1+2+0, 1+0+2) = (6, 3, 3)! Opt = 6*3 + 3*2 + 3*2 = 30, b^T y = 7 + 16 + 7 = 30!
    A2 = [
        [Fraction(1), Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(1), Fraction(0)],
        [Fraction(1), Fraction(0), Fraction(2)],
        [Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(-1)],
    ]
    b2 = [Fraction(7), Fraction(8), Fraction(7), Fraction(0), Fraction(0), Fraction(0)]
    c2 = [Fraction(6), Fraction(3), Fraction(3)]
    x2 = [Fraction(3), Fraction(2), Fraction(2)]
    y2 = [Fraction(1), Fraction(2), Fraction(1), Fraction(0), Fraction(0), Fraction(0)]

    # C03: Unbounded LP: max 2x1 + 3x2 s.t. x1 - x2 <= 2, -x1 <= 0, -x2 <= 0
    # Feasible x=(1, 1), improving ray r=(1, 2): Ar = (-1, -1, -2) <= 0, c^T r = 8 > 0!
    A3 = [
        [Fraction(1), Fraction(-1)],
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(-1)],
    ]
    b3 = [Fraction(2), Fraction(0), Fraction(0)]
    c3 = [Fraction(2), Fraction(3)]
    x3 = [Fraction(1), Fraction(1)]
    r3 = [Fraction(1), Fraction(2)]

    # C04: Infeasible LP with minimal-support Farkas cert (rank=2, supp=3 <= 1+rank=3):
    # x1 + x2 <= 1, -x1 <= -2, -x2 <= -2, x1 - x2 <= 5 (redundant 4th row, y4=0!)
    # y = (1, 1, 1, 0): A^T y = (0, 0), b^T y = 1 - 2 - 2 = -3 < 0, supp(y) = 3 <= 1 + 2!
    A4 = [
        [Fraction(1), Fraction(1)],
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(-1)],
        [Fraction(1), Fraction(-1)],
    ]
    b4 = [Fraction(1), Fraction(-2), Fraction(-2), Fraction(5)]
    c4 = [Fraction(1), Fraction(1)]
    y4 = [Fraction(1), Fraction(1), Fraction(1), Fraction(0)]

    # --- Mode 2: Reciprocal Scaling & Sec. 3.4 Division-Free Newton Det (4 certs) ---
    # C05: Sec. 1.2 circuit imbalance subspace C = rowspace([[1,0,1,u],[0,1,u,1]]) with u=10!
    # C_perp = rowspace([[-1,-10,1,0],[-10,-1,0,1]])! Check dot products:
    # (1,0,1,10).(-1,-10,1,0) = 0; (1,0,1,10).(-10,-1,0,1) = 0;
    # (0,1,10,1).(-1,-10,1,0) = 0; (0,1,10,1).(-10,-1,0,1) = 0!
    C5 = [[Fraction(1), Fraction(0), Fraction(1), Fraction(10)], [Fraction(0), Fraction(1), Fraction(10), Fraction(1)]]
    Cp5 = [[Fraction(-1), Fraction(-10), Fraction(1), Fraction(0)], [Fraction(-10), Fraction(-1), Fraction(0), Fraction(1)]]
    M5 = [[1, 10], [10, 1]]  # det = -99, 2! det = -198

    # C06: Sec. 1.2 high-imbalance u=1000 with steep prices p=[-12, 8, -5, 10] and 3x3 Newton matrix
    C6 = [[Fraction(1), Fraction(0), Fraction(1), Fraction(1000)], [Fraction(0), Fraction(1), Fraction(1000), Fraction(1)]]
    Cp6 = [[Fraction(-1), Fraction(-1000), Fraction(1), Fraction(0)], [Fraction(-1000), Fraction(-1), Fraction(0), Fraction(1)]]
    M6 = [[2, -1, 3], [0, 4, -2], [5, 1, -1]]

    # C07: N=5 orthogonal subspaces (dim 2 + dim 3 = 5) & 4x4 Newton matrix
    C7 = [
        [Fraction(1), Fraction(2), Fraction(-1), Fraction(0), Fraction(1)],
        [Fraction(0), Fraction(1), Fraction(1), Fraction(-2), Fraction(1)],
    ]
    # Solve C7 * v = 0:
    # v1 = [1, 0, 1, 1/2, 0] -> [2, 0, 2, 1, 0]: (2-2=0, 2-2=0)
    # v2 = [-3, 1, 0, 1, 1]: (-3+2+1=0, 1-2+1=0)
    # v3 = [3, -1, 1, 0, 0]: (3-2-1=0, -1+1=0)
    Cp7 = [
        [Fraction(2), Fraction(0), Fraction(2), Fraction(1), Fraction(0)],
        [Fraction(-3), Fraction(1), Fraction(0), Fraction(1), Fraction(1)],
        [Fraction(3), Fraction(-1), Fraction(1), Fraction(0), Fraction(0)],
    ]
    M7 = [[3, 1, 0, -2], [1, 4, -1, 1], [0, 2, 5, -3], [-1, 0, 2, 2]]

    # C08: N=6 orthogonal subspaces (dim 3 + dim 3 = 6) & 5x5 Newton matrix
    C8 = [
        [Fraction(1), Fraction(0), Fraction(0), Fraction(2), Fraction(-1), Fraction(3)],
        [Fraction(0), Fraction(1), Fraction(0), Fraction(-1), Fraction(4), Fraction(1)],
        [Fraction(0), Fraction(0), Fraction(1), Fraction(5), Fraction(2), Fraction(-2)],
    ]
    Cp8 = [
        [Fraction(-2), Fraction(1), Fraction(-5), Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(1), Fraction(-4), Fraction(-2), Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(-3), Fraction(-1), Fraction(2), Fraction(0), Fraction(0), Fraction(1)],
    ]
    M8 = [
        [2, 1, 0, -1, 3],
        [0, 3, -2, 1, 0],
        [1, 0, 4, -2, 1],
        [-2, 1, 1, 3, -1],
        [1, -1, 2, 0, 2],
    ]

    # --- Mode 3: Monotone LCP & Cor. 6 Convex QP KKT (4 certs) ---
    # C09: 2D symmetric PD LCP: M=[[2,1],[1,2]], q=[-3,-2] -> z*=(4/3, 1/3), w*=(0,0)
    # Probe z'=(2, 1) -> w'=(2, 2) >= 0, gap = 6
    LCP9_M = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    LCP9_q = [Fraction(-3), Fraction(-2)]
    LCP9_zs = [Fraction(4, 3), Fraction(1, 3)]
    LCP9_zp = [Fraction(2), Fraction(1)]

    # C10: 3D asymmetric monotone LCP (skew-symmetric + PD): M=[[2,1,-1],[-1,3,2],[1,-2,2]]
    # M + M^T = diag(4, 6, 4) PD! Choose z*=(1, 0, 2), w*=(0, 3, 0):
    # q = w* - M z* = (0 - (2-2), 3 - (-1+4), 0 - (1+4)) = (0, 0, -5)!
    # Probe z'=(2, 1, 2): M z' + q = (4+1-2+0, -2+3+4+0, 2-2+4-5) -> last is -1, adjust z'=(2, 1, 3):
    # M z' + q = (4+1-3, -2+3+6, 2-2+6-5) = (2, 7, 1) >= 0!
    LCP10_M = [
        [Fraction(2), Fraction(1), Fraction(-1)],
        [Fraction(-1), Fraction(3), Fraction(2)],
        [Fraction(1), Fraction(-2), Fraction(2)],
    ]
    LCP10_q = [Fraction(0), Fraction(0), Fraction(-5)]
    LCP10_zs = [Fraction(1), Fraction(0), Fraction(2)]
    LCP10_zp = [Fraction(2), Fraction(1), Fraction(3)]

    # C11: Cor. 6 Convex QP KKT block (d=4: 2 primal u>=0 + 2 dual multipliers mu>=0)
    # G = [[2, 0], [0, 2]], A = [[1, 1], [1, 0]], M = [[G, A^T], [-A, 0]] (PSD monotone!)
    # Choose z*=(1, 1, 1, 0), w*=(0, 0, 0, 1):
    # M z* = (2+1, 2+1, -2, -1) = (3, 3, -2, -1) => q = (-3, -3, 2, 2)!
    # Check: b = (2, 2), c = (-3, -3) -> min x1^2+x2^2 - 3x1 - 3x2 s.t. x1+x2<=2, x1<=2, x>=0!
    # Probe z'=(1/2, 1/2, 2, 1): M z' + q = (1+2+1-3, 1+2-3, -1+2, -1/2+2) = (1, 0, 1, 3/2) >= 0!
    LCP11_M = [
        [Fraction(2), Fraction(0), Fraction(1), Fraction(1)],
        [Fraction(0), Fraction(2), Fraction(1), Fraction(0)],
        [Fraction(-1), Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(-1), Fraction(0), Fraction(0), Fraction(0)],
    ]
    LCP11_q = [Fraction(-3), Fraction(-3), Fraction(2), Fraction(2)]
    LCP11_zs = [Fraction(1), Fraction(1), Fraction(1), Fraction(0)]
    LCP11_zp = [Fraction(1, 2), Fraction(1, 2), Fraction(2), Fraction(1)]

    # C12: 5D degenerate PSD monotone LCP (rank-deficient symmetric part + skew KKT block)
    # G = [[1, 1, 0], [1, 1, 0], [0, 0, 2]] (PSD rank 2!), A = [[1, 0, 1], [0, 1, 1]]
    LCP12_M = [
        [Fraction(1), Fraction(1), Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(1)],
        [Fraction(0), Fraction(0), Fraction(2), Fraction(1), Fraction(1)],
        [Fraction(-1), Fraction(0), Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(-1), Fraction(-1), Fraction(0), Fraction(0)],
    ]
    # z* = (1, 0, 1, 1, 0), w* = (0, 0, 0, 0, 1) -> M z* = (2, 1, 3, -2, -1) -> q = (-2, -1, -3, 2, 2)
    # Wait: check w* = M z* + q = (0, 0, 0, 0, 1) >= 0 and z* . w* = 0!
    # Probe z' = (1/2, 1/2, 1/2, 2, 1): M z' + q = (1+2-2, 1+1-1, 1+2+1-3, -1+2, -1+2) = (1, 1, 1, 1, 1) >= 0!
    LCP12_q = [Fraction(-2), Fraction(-1), Fraction(-3), Fraction(2), Fraction(2)]
    LCP12_zs = [Fraction(1), Fraction(0), Fraction(1), Fraction(1), Fraction(0)]
    LCP12_zp = [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(2), Fraction(1)]

    # --- Mode 4: Max-Norm Contraction, Bellman Fixed Point & Clipped Update (4 certs) ---
    # C13: Exact Figure 5 two-state game from Section 9.3! Low center y=(0, 1/4), H=1 -> u=(1, 1/2), z=(0,-1/4)!
    fig5_owners = ["max", "min"]
    fig5_actions = [
        [
            (Fraction(3, 4), [Fraction(1), Fraction(0)], [Fraction(1, 2), Fraction(0)]),
            (Fraction(0), [Fraction(0), Fraction(1)], [Fraction(0), Fraction(1, 2)]),
        ],
        [
            (Fraction(1), [Fraction(0), Fraction(1)], [Fraction(0), Fraction(1, 2)]),
            (Fraction(-1, 4), [Fraction(1), Fraction(0)], [Fraction(1, 2), Fraction(0)]),
        ],
    ]
    fig5_v = [Fraction(3, 2), Fraction(1, 2)]

    # C14: Same Fig. 5 game at neutral center y=(0, 2), H=1 (left panel of Fig. 5!)
    # Check clipped fixed point at y=(0, 2), H=1:
    # State 0 (max): stay -> 3/4 + 0 - 0 + u0 = 3/4 + u0; move -> 0 + 1 - 0 + u1 = 1 + u1 -> u0 = 1!
    # State 1 (min): stay -> 1 + 1 - 2 + u1 = u1; move -> -1/4 + 0 - 2 + u0 = -9/4 + u0 = -5/4 -> clipped to -1!
    # Check u=(1, -1): State 0 max(7/4, 0) -> clip 1; State 1 min(-1, -5/4) -> clip -1! Exact!
    # New err: |3/2 - 1| = 1/2, |1/2 - 1| = 1/2 <= H=1!

    # C15: 3-state stochastic discounted turn-based game (gamma = 1/4, beta = 3/4)
    # Owners: ["max", "min", "max"]
    # State 0 (max): A0: r=1/2, P=[1/2,1/2,0], B=[3/8,3/8,0]; A1: r=1/4, P=[0,0,1], B=[0,0,3/4]
    # State 1 (min): A0: r=0, P=[1/2,0,1/2], B=[3/8,0,3/8]; A1: r=1, P=[0,1,0], B=[0,3/4,0]
    # State 2 (max): A0: r=1/4, P=[0,1/2,1/2], B=[0,3/8,3/8]; A1: r=1/2, P=[1,0,0], B=[3/4,0,0]
    # Let's pick an exact simple fixed point v = [2, 1, 2] and tune r cleanly!
    # With v = [2, 1, 2], beta = 1/2 (gamma = 1/2):
    # State 0 (max): A0 (active): B=[1/4, 1/4, 0] -> Bv = 3/4 -> r = 5/4 gives 2; A1: B=[0, 0, 1/2] -> Bv = 1 -> r = 1/2 gives 3/2 <= 2
    # State 1 (min): A0 (active): B=[1/4, 0, 1/4] -> Bv = 1 -> r = 0 gives 1; A1: B=[0, 1/2, 0] -> Bv = 1/2 -> r = 1 gives 3/2 >= 1
    # State 2 (max): A0 (active): B=[0, 1/4, 1/4] -> Bv = 3/4 -> r = 5/4 gives 2; A1: B=[1/2, 0, 0] -> Bv = 1 -> r = 3/4 gives 7/4 <= 2
    g15_owners = ["max", "min", "max"]
    g15_actions = [
        [
            (Fraction(5, 4), [Fraction(1, 2), Fraction(1, 2), Fraction(0)], [Fraction(1, 4), Fraction(1, 4), Fraction(0)]),
            (Fraction(1, 2), [Fraction(0), Fraction(0), Fraction(1)], [Fraction(0), Fraction(0), Fraction(1, 2)]),
        ],
        [
            (Fraction(0), [Fraction(1, 2), Fraction(0), Fraction(1, 2)], [Fraction(1, 4), Fraction(0), Fraction(1, 4)]),
            (Fraction(1), [Fraction(0), Fraction(1), Fraction(0)], [Fraction(0), Fraction(1, 2), Fraction(0)]),
        ],
        [
            (Fraction(5, 4), [Fraction(0), Fraction(1, 2), Fraction(1, 2)], [Fraction(0), Fraction(1, 4), Fraction(1, 4)]),
            (Fraction(3, 4), [Fraction(1), Fraction(0), Fraction(0)], [Fraction(1, 2), Fraction(0), Fraction(0)]),
        ],
    ]
    g15_v = [Fraction(2), Fraction(1), Fraction(2)]
    # Low center y = [1, 1/2, 1], H = 1 -> ||v-y||_inf = 1 <= 2H!
    # Solve exact clipped fixed point u by simple iteration (exact Fraction)
    g15_y = [Fraction(1), Fraction(1, 2), Fraction(1)]
    g15_H = Fraction(1)
    u15 = [Fraction(0), Fraction(0), Fraction(0)]
    for _ in range(40):
        nxt = eval_clipped_operator(g15_owners, g15_actions, g15_y, u15, g15_H)
        if nxt == u15:
            break
        u15 = nxt
    assert eval_clipped_operator(g15_owners, g15_actions, g15_y, u15, g15_H) == u15

    # C16: 4-state discounted turn-based game with small contraction gamma = 1/4 (beta = 3/4)
    # Fixed point v = [4, 2, 3, 1]
    # State 0 (max): A0 active P=[1,0,0,0] B=[3/4,0,0,0] -> Bv=3 -> r=1 (4); A1 P=[0,1,0,0] B=[0,3/4,0,0] -> Bv=3/2 -> r=2 (7/2)
    # State 1 (min): A0 active P=[0,0,0,1] B=[0,0,0,3/4] -> Bv=3/4 -> r=5/4 (2); A1 P=[0,1,0,0] B=[0,3/4,0,0] -> Bv=3/2 -> r=1 (5/2)
    # State 2 (max): A0 active P=[0,0,1,0] B=[0,0,3/4,0] -> Bv=9/4 -> r=3/4 (3); A1 P=[0,1,0,0] B=[0,3/4,0,0] -> Bv=3/2 -> r=1 (5/2)
    # State 3 (min): A0 active P=[0,0,0,1] B=[0,0,0,3/4] -> Bv=3/4 -> r=1/4 (1); A1 P=[1,0,0,0] B=[3/4,0,0,0] -> Bv=3 -> r=-1 (2)
    g16_owners = ["max", "min", "max", "min"]
    g16_actions = [
        [
            (Fraction(1), [Fraction(1), Fraction(0), Fraction(0), Fraction(0)], [Fraction(3, 4), Fraction(0), Fraction(0), Fraction(0)]),
            (Fraction(2), [Fraction(0), Fraction(1), Fraction(0), Fraction(0)], [Fraction(0), Fraction(3, 4), Fraction(0), Fraction(0)]),
        ],
        [
            (Fraction(5, 4), [Fraction(0), Fraction(0), Fraction(0), Fraction(1)], [Fraction(0), Fraction(0), Fraction(0), Fraction(3, 4)]),
            (Fraction(1), [Fraction(0), Fraction(1), Fraction(0), Fraction(0)], [Fraction(0), Fraction(3, 4), Fraction(0), Fraction(0)]),
        ],
        [
            (Fraction(3, 4), [Fraction(0), Fraction(0), Fraction(1), Fraction(0)], [Fraction(0), Fraction(0), Fraction(3, 4), Fraction(0)]),
            (Fraction(1), [Fraction(0), Fraction(1), Fraction(0), Fraction(0)], [Fraction(0), Fraction(3, 4), Fraction(0), Fraction(0)]),
        ],
        [
            (Fraction(1, 4), [Fraction(0), Fraction(0), Fraction(0), Fraction(1)], [Fraction(0), Fraction(0), Fraction(0), Fraction(3, 4)]),
            (Fraction(-1), [Fraction(1), Fraction(0), Fraction(0), Fraction(0)], [Fraction(3, 4), Fraction(0), Fraction(0), Fraction(0)]),
        ],
    ]
    g16_v = [Fraction(4), Fraction(2), Fraction(3), Fraction(1)]
    g16_y = [Fraction(3), Fraction(1), Fraction(2), Fraction(0)]
    g16_H = Fraction(1)
    u16 = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    for _ in range(40):
        nxt = eval_clipped_operator(g16_owners, g16_actions, g16_y, u16, g16_H)
        if nxt == u16:
            break
        u16 = nxt
    assert eval_clipped_operator(g16_owners, g16_actions, g16_y, u16, g16_H) == u16

    certs: List[Dict[str, Any]] = [
        # Mode 1 (4 certs)
        build_mode1_opt("m1_lp_optimal_2d_strict_comp", A1, b1, c1, x1, y1),
        build_mode1_opt("m1_lp_optimal_3d_strict_comp", A2, b2, c2, x2, y2),
        build_mode1_unb("m1_lp_unbounded_recession_ray", A3, b3, c3, x3, r3),
        build_mode1_inf("m1_lp_infeasible_farkas_minsupp", A4, b4, c4, y4),
        # Mode 2 (4 certs)
        build_mode2_cert("m2_recip_circuit_u10_newton2", 4, C5, Cp5, [3, -2, 5, -4], M5),
        build_mode2_cert("m2_recip_circuit_u1000_newton3", 4, C6, Cp6, [-12, 8, -5, 10], M6),
        build_mode2_cert("m2_recip_n5_ortho_newton4", 5, C7, Cp7, [4, -6, 2, -3, 7], M7),
        build_mode2_cert("m2_recip_n6_ortho_newton5", 6, C8, Cp8, [-9, 5, -1, 8, -4, 6], M8),
        # Mode 3 (4 certs)
        build_mode3_cert("m3_lcp_2d_sym_pd", LCP9_M, LCP9_q, LCP9_zs, LCP9_zp),
        build_mode3_cert("m3_lcp_3d_asym_monotone", LCP10_M, LCP10_q, LCP10_zs, LCP10_zp),
        build_mode3_cert("m3_qp_kkt_4d_skew_block", LCP11_M, LCP11_q, LCP11_zs, LCP11_zp),
        build_mode3_cert("m3_qp_kkt_5d_psd_degenerate", LCP12_M, LCP12_q, LCP12_zs, LCP12_zp),
        # Mode 4 (4 certs)
        build_mode4_cert("m4_game_fig5_low_center", fig5_owners, fig5_actions, Fraction(1, 2), fig5_v, [Fraction(0), Fraction(0)], [Fraction(0), Fraction(1, 4)], Fraction(1), [Fraction(1), Fraction(1, 2)]),
        build_mode4_cert("m4_game_fig5_neutral_center", fig5_owners, fig5_actions, Fraction(1, 2), fig5_v, [Fraction(3), Fraction(-1)], [Fraction(0), Fraction(2)], Fraction(1), [Fraction(1), Fraction(-1)]),
        build_mode4_cert("m4_game_3state_stochastic", g15_owners, g15_actions, Fraction(1, 2), g15_v, [Fraction(0), Fraction(3), Fraction(0)], g15_y, g15_H, u15),
        build_mode4_cert("m4_game_4state_gamma_quarter", g16_owners, g16_actions, Fraction(1, 4), g16_v, [Fraction(1), Fraction(4), Fraction(0), Fraction(3)], g16_y, g16_H, u16),
    ]

    atlas_rows: List[Dict[str, Any]] = []
    for cert in certs:
        cid = cert["certificate_id"]
        path = CERT_DIR / f"{cid}.cert.json"
        path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
        summary = verify_certificate_dict(cert)
        times_ms: List[float] = []
        for _ in range(7):
            t0 = time.perf_counter()
            verify_certificate_dict(cert)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(times_ms) / len(times_ms)
        atlas_rows.append(
            {
                "certificate_id": cid,
                "mode": cert["mode"],
                "mean_ms": round(mean_ms, 3),
                "summary": summary,
            }
        )
        print(f"[OK] {cid:38s} | {mean_ms:7.3f} ms | {summary}")

    (RES_DIR / "exp1_certificate_atlas.json").write_text(
        json.dumps({"protocol": PROTOCOL_ID, "count": len(atlas_rows), "certificates": atlas_rows}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Exp 2: Sec. 1.2 circuit imbalance det([[1, u], [u, 1]]) = 1 - u^2 across u in 2^1..2^12
    exp2_rows = []
    for exp_k in range(1, 13):
        u_val = 1 << exp_k
        nd = newton_det_scaled_int([[1, u_val], [u_val, 1]])
        det_abs = abs(nd) // 2
        exp2_rows.append(
            {
                "u": u_val,
                "log2_u": exp_k,
                "det_abs": det_abs,
                "newton_2fact_det": nd,
            }
        )
    (RES_DIR / "exp2_circuit_imbalance.json").write_text(
        json.dumps({"rows": exp2_rows}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Exp 3: Sec. 9.3 Fig. 5 exact clipped update halving sequence starting from y=(0, 2), H0=1 -> H_t = 2^{-t}
    exp3_rows = []
    cur_y = [Fraction(0), Fraction(2)]
    cur_H = Fraction(1)
    for step in range(7):
        err_before = max(abs(fig5_v[i] - cur_y[i]) for i in range(2))
        exp3_rows.append(
            {
                "step": step,
                "H_str": f"{cur_H.numerator}/{cur_H.denominator}",
                "H_float": float(cur_H),
                "err_str": f"{err_before.numerator}/{err_before.denominator}",
                "err_float": float(err_before),
            }
        )
        u_cur = [Fraction(0), Fraction(0)]
        for _ in range(30):
            nxt = eval_clipped_operator(fig5_owners, fig5_actions, cur_y, u_cur, cur_H)
            if nxt == u_cur:
                break
            u_cur = nxt
        cur_y = [cur_y[0] + u_cur[0], cur_y[1] + u_cur[1]]
        cur_H = cur_H * Fraction(1, 2)
    (RES_DIR / "exp3_clipped_halving.json").write_text(
        json.dumps({"rows": exp3_rows}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Exp 4: Sec. 7.2 Monotone LCP gap split across the 4 Mode-3 canonical certificates
    exp4_rows = []
    for row in atlas_rows[8:12]:
        cid = row["certificate_id"]
        c = json.loads((CERT_DIR / f"{cid}.cert.json").read_text(encoding="utf-8"))["payload"]
        d = c["d_dim"]
        M = [[Fraction(p[0], p[1]) for p in r] for r in c["M"]]
        q = [Fraction(p[0], p[1]) for p in c["q"]]
        zs = [Fraction(p[0], p[1]) for p in c["z_star"]]
        ws = [Fraction(p[0], p[1]) for p in c["w_star"]]
        zp = [Fraction(p[0], p[1]) for p in c["z_probe"]]
        wp = [v + q[i] for i, v in enumerate(mat_vec_frac(M, zp))]
        dz = [zp[i] - zs[i] for i in range(d)]
        cross_term = dot_frac(zp, ws) + dot_frac(zs, wp)
        quad_term = dot_frac(dz, mat_vec_frac(M, dz))
        total_gap = dot_frac(zp, wp)
        assert cross_term + quad_term == total_gap
        exp4_rows.append(
            {
                "cert_id": cid,
                "d": d,
                "cross_float": float(cross_term),
                "quad_float": float(quad_term),
                "total_gap_str": f"{total_gap.numerator}/{total_gap.denominator}",
                "quad_str": f"{quad_term.numerator}/{quad_term.denominator}",
            }
        )
    (RES_DIR / "exp4_lcp_gap_split.json").write_text(
        json.dumps({"rows": exp4_rows}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("All 16 certificates and 4 experiment JSON files generated successfully.")


if __name__ == "__main__":
    main()
