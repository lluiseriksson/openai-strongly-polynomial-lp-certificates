"""Zero-float 4-mode certificate auditor for SPLP-SMALE9-CERT-1.0.

Strictly parses JSON without floats (rejecting IEEE-754 decimals, NaN/Infinity,
duplicate keys via object_pairs_hook, and unreduced fractions) and verifies four
canonical modes aligned with OpenAI's November 24, 2026 preprint "Strongly
polynomial linear programming":
  1. lp_trichotomy_certificate_cert         (Secs. 1, 3.3, Thm. 1 optimal/unbounded/infeasible)
  2. reciprocal_scaling_and_newton_det_cert (Secs. 1.2, 3.1, 3.2, 3.4, Thm. 3)
  3. monotone_lcp_and_qp_kkt_cert           (Secs. 7.2, 7.3, Thm. 5, Cor. 6)
  4. max_norm_contraction_and_game_cert     (Secs. 8, 9.1, 9.3, Thms. 8, 11)

NO floating-point operations, true division '/', or math imports appear anywhere in this module.
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from src.splp_exact import (
    CertificateError,
    dot_frac,
    eval_bellman_operator,
    eval_clipped_operator,
    factorial_exact,
    gcd_int,
    is_symmetric_psd_exact,
    mat_det_exact,
    mat_rank_exact,
    mat_transpose_vec_frac,
    mat_vec_frac,
    newton_det_scaled_int,
    pow2_frac,
)

PROTOCOL_ID = "SPLP-SMALE9-CERT-1.0"
MAX_JSON_BYTES = 1_000_000
MAX_RATIONAL_ABS = 10**12
VALID_MODES = frozenset(
    {
        "lp_trichotomy_certificate_cert",
        "reciprocal_scaling_and_newton_det_cert",
        "monotone_lcp_and_qp_kkt_cert",
        "max_norm_contraction_and_game_cert",
    }
)


def _reject_float(val: str) -> Any:
    raise CertificateError(f"Floating-point literal rejected by zero-float policy: {val!r}")


def _reject_constant(val: str) -> Any:
    raise CertificateError(f"Non-standard JSON constant rejected: {val!r}")


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise CertificateError(f"Duplicate JSON key rejected: {k!r}")
        out[k] = v
    return out


def load_strict_json_str(raw: str) -> Dict[str, Any]:
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise CertificateError(f"JSON payload exceeds MAX_JSON_BYTES={MAX_JSON_BYTES}")
    try:
        data = json.loads(
            raw,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except CertificateError:
        raise
    except Exception as exc:
        raise CertificateError(f"Malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CertificateError("Top-level certificate payload must be a JSON object")
    return data


def _req_int(obj: Mapping[str, Any], key: str) -> int:
    if key not in obj:
        raise CertificateError(f"Missing required integer field {key!r}")
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise CertificateError(f"Field {key!r} must be an exact integer")
    return v


def _req_str(obj: Mapping[str, Any], key: str) -> str:
    if key not in obj:
        raise CertificateError(f"Missing required string field {key!r}")
    v = obj[key]
    if not isinstance(v, str) or not v.strip():
        raise CertificateError(f"Field {key!r} must be a non-empty string")
    return v


def _parse_frac_pair(pair: Any, label: str) -> Fraction:
    if not isinstance(pair, list) or len(pair) != 2:
        raise CertificateError(f"{label} must be [num, den]")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in pair) or pair[1] <= 0:
        raise CertificateError(f"{label} must have integer numerator and positive integer denominator")
    if abs(pair[0]) > MAX_RATIONAL_ABS or pair[1] > MAX_RATIONAL_ABS:
        raise CertificateError(f"{label} rational entries exceed magnitude cap {MAX_RATIONAL_ABS}")
    if gcd_int(pair[0], pair[1]) != 1:
        raise CertificateError(f"{label} rational {pair} must be reduced in lowest terms (gcd == 1)")
    return Fraction(pair[0], pair[1])


def _parse_frac_vec(raw: Any, n: int, label: str) -> List[Fraction]:
    if not isinstance(raw, list) or len(raw) != n:
        raise CertificateError(f"{label} must be a length-{n} rational vector")
    return [_parse_frac_pair(item, f"{label}[{i}]") for i, item in enumerate(raw)]


def _parse_frac_mat(raw: Any, m: int, n: int, label: str) -> List[List[Fraction]]:
    if not isinstance(raw, list) or len(raw) != m:
        raise CertificateError(f"{label} must be a {m}x{n} rational matrix")
    return [_parse_frac_vec(row, n, f"{label}[{i}]") for i, row in enumerate(raw)]


def _check_keys(obj: Mapping[str, Any], expected: set[str], label: str) -> None:
    keys = set(obj.keys())
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise CertificateError(f"Missing keys in {label}: {sorted(missing)}")
    if extra:
        raise CertificateError(f"Unexpected keys in {label}: {sorted(extra)}")


# ---------------------------------------------------------------------------
# Mode 1: Exact Theorem 1 LP Trichotomy (Optimal / Unbounded / Infeasible)
# ---------------------------------------------------------------------------

def _verify_mode1(payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {"branch", "m_rows", "n_cols", "A", "b", "c", "witness"}
    _check_keys(payload, expected, "Mode 1 payload")

    branch = _req_str(payload, "branch")
    if branch not in {"optimal", "unbounded", "infeasible"}:
        raise CertificateError(f"Unknown Theorem 1 branch {branch!r}")
    m = _req_int(payload, "m_rows")
    n = _req_int(payload, "n_cols")
    if not (2 <= m <= 12 and 2 <= n <= 8):
        raise CertificateError("LP dimensions must satisfy 2 <= m <= 12 and 2 <= n <= 8")

    A = _parse_frac_mat(payload.get("A"), m, n, "A")
    b = _parse_frac_vec(payload.get("b"), m, "b")
    c = _parse_frac_vec(payload.get("c"), n, "c")
    wit = payload.get("witness")
    if not isinstance(wit, Mapping):
        raise CertificateError("witness must be a JSON object")

    if branch == "optimal":
        _check_keys(wit, {"x", "y", "claimed_opt_value"}, "optimal witness")
        x = _parse_frac_vec(wit.get("x"), n, "witness.x")
        y = _parse_frac_vec(wit.get("y"), m, "witness.y")
        opt_val = _parse_frac_pair(wit.get("claimed_opt_value"), "witness.claimed_opt_value")

        Ax = mat_vec_frac(A, x)
        slacks = [b[i] - Ax[i] for i in range(m)]
        if any(s < 0 for s in slacks):
            raise CertificateError("Primal infeasibility: A*x <= b violated")
        if any(yi < 0 for yi in y):
            raise CertificateError("Dual infeasibility: y >= 0 violated")
        ATy = mat_transpose_vec_frac(A, y)
        if ATy != c:
            raise CertificateError("Dual stationarity A^T y == c violated")
        cx = dot_frac(c, x)
        by = dot_frac(b, y)
        if cx != by or cx != opt_val:
            raise CertificateError(f"Strong duality gap nonzero or opt mismatch (c^Tx={cx}, b^Ty={by}, claimed={opt_val})")
        # Strict complementarity (Theorem 1 / Goldman-Tucker): b - Ax + y > 0 componentwise!
        for i in range(m):
            if slacks[i] * y[i] != 0:
                raise CertificateError(f"Complementary slackness violated at row {i}")
            if slacks[i] + y[i] <= 0:
                raise CertificateError(f"Strict complementarity (b - Ax + y > 0) violated at row {i}")
        return {
            "mode": "lp_trichotomy_certificate_cert",
            "branch": "optimal",
            "dims": f"{m}x{n}",
            "opt_value": f"{opt_val.numerator}/{opt_val.denominator}",
            "strict_comp_rows": m,
        }

    if branch == "unbounded":
        _check_keys(wit, {"x", "r", "claimed_ray_gain"}, "unbounded witness")
        x = _parse_frac_vec(wit.get("x"), n, "witness.x")
        r = _parse_frac_vec(wit.get("r"), n, "witness.r")
        gain = _parse_frac_pair(wit.get("claimed_ray_gain"), "witness.claimed_ray_gain")

        Ax = mat_vec_frac(A, x)
        if any(Ax[i] > b[i] for i in range(m)):
            raise CertificateError("Base point x is infeasible for A*x <= b")
        Ar = mat_vec_frac(A, r)
        if any(v > 0 for v in Ar):
            raise CertificateError("Ray r violates recession condition A*r <= 0")
        cr = dot_frac(c, r)
        if cr <= 0 or cr != gain:
            raise CertificateError(f"Improving ray gain c^T r={cr} must be positive and equal claimed {gain}")
        return {
            "mode": "lp_trichotomy_certificate_cert",
            "branch": "unbounded",
            "dims": f"{m}x{n}",
            "ray_gain": f"{cr.numerator}/{cr.denominator}",
        }

    # branch == "infeasible"
    _check_keys(wit, {"y", "claimed_rank_A", "claimed_farkas_rhs"}, "infeasible witness")
    y = _parse_frac_vec(wit.get("y"), m, "witness.y")
    claimed_rank = _req_int(wit, "claimed_rank_A")
    claimed_rhs = _parse_frac_pair(wit.get("claimed_farkas_rhs"), "witness.claimed_farkas_rhs")
    if any(yi < 0 for yi in y):
        raise CertificateError("Farkas witness violates y >= 0")
    ATy = mat_transpose_vec_frac(A, y)
    if any(v != 0 for v in ATy):
        raise CertificateError("Farkas witness violates A^T y == 0")
    by = dot_frac(b, y)
    if by >= 0 or by != claimed_rhs:
        raise CertificateError(f"Farkas witness b^T y={by} must be strictly negative and match claimed {claimed_rhs}")
    rank_A = mat_rank_exact(A)
    if claimed_rank != rank_A:
        raise CertificateError(f"claimed_rank_A={claimed_rank} != exact rank {rank_A}")
    supp_y = sum(1 for yi in y if yi > 0)
    if supp_y == 0 or supp_y > 1 + rank_A:
        raise CertificateError(f"Farkas witness support {supp_y} violates Theorem 1 bound <= 1 + rank(A) = {1 + rank_A}")
    return {
        "mode": "lp_trichotomy_certificate_cert",
        "branch": "infeasible",
        "dims": f"{m}x{n}",
        "rank_A": rank_A,
        "farkas_support": supp_y,
    }


# ---------------------------------------------------------------------------
# Mode 2: Reciprocal Scaling Orthogonality & Sec. 3.4 Division-Free Newton Det
# ---------------------------------------------------------------------------

def _verify_mode2(payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {
        "N_dim",
        "C_basis",
        "C_perp_basis",
        "prices_p",
        "newton_int_matrix",
        "claimed_k_fact_det",
    }
    _check_keys(payload, expected, "Mode 2 payload")

    N = _req_int(payload, "N_dim")
    if not (2 <= N <= 8):
        raise CertificateError("N_dim must lie in [2, 8]")

    raw_C = payload.get("C_basis")
    raw_Cp = payload.get("C_perp_basis")
    if not isinstance(raw_C, list) or not isinstance(raw_Cp, list) or not raw_C or not raw_Cp:
        raise CertificateError("C_basis and C_perp_basis must be non-empty lists")
    if len(raw_C) + len(raw_Cp) != N:
        raise CertificateError(f"dim(C) + dim(C_perp) = {len(raw_C)+len(raw_Cp)} != N = {N}")
    C = _parse_frac_mat(raw_C, len(raw_C), N, "C_basis")
    Cp = _parse_frac_mat(raw_Cp, len(raw_Cp), N, "C_perp_basis")
    if mat_rank_exact(C) != len(C) or mat_rank_exact(Cp) != len(Cp):
        raise CertificateError("C_basis and C_perp_basis must each have linearly independent rows")

    # Verify orthogonality C perp C_perp
    for i, u in enumerate(C):
        for j, v in enumerate(Cp):
            if dot_frac(u, v) != 0:
                raise CertificateError(f"C_basis[{i}] and C_perp_basis[{j}] are not orthogonal")

    # Verify reciprocal scaling D_{i+} = 2^{-p_i}, D_{i-} = 2^{p_i} preserves Q(y) = sum y_{i+} y_{i-} = 0
    raw_p = payload.get("prices_p")
    if not isinstance(raw_p, list) or len(raw_p) != N:
        raise CertificateError(f"prices_p must be a length-{N} integer list")
    prices: List[int] = []
    for idx, pv in enumerate(raw_p):
        if isinstance(pv, bool) or not isinstance(pv, int) or not (-16 <= pv <= 16):
            raise CertificateError(f"prices_p[{idx}] must be an integer in [-16, 16]")
        prices.append(pv)
    if all(pv == 0 for pv in prices):
        raise CertificateError("prices_p must be nonzero for non-vacuous reciprocal scaling")

    D_pos = [pow2_frac(-pv) for pv in prices]
    D_neg = [pow2_frac(pv) for pv in prices]
    for i in range(N):
        if D_pos[i] * D_neg[i] != Fraction(1):
            raise CertificateError("Reciprocal scaling invariant D_{i+} * D_{i-} == 1 failed")

    # Audit Q(y D) = 0 across all basis pairs (u, v) in C x C_perp
    for u in C:
        u_scaled = [u[i] * D_pos[i] for i in range(N)]
        for v in Cp:
            v_scaled = [v[i] * D_neg[i] for i in range(N)]
            if dot_frac(u_scaled, v_scaled) != 0:
                raise CertificateError("Reciprocal scaling failed to preserve paired quadratic form Q(yD) = 0")

    # Audit Section 3.4 division-free Newton's identities determinant k! * det(M)
    raw_M = payload.get("newton_int_matrix")
    if not isinstance(raw_M, list) or not (2 <= len(raw_M) <= 5):
        raise CertificateError("newton_int_matrix must be k x k with 2 <= k <= 5")
    k = len(raw_M)
    M_int: List[List[int]] = []
    for i, row in enumerate(raw_M):
        if not isinstance(row, list) or len(row) != k:
            raise CertificateError(f"newton_int_matrix[{i}] must have length {k}")
        r_vals: List[int] = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, int) or abs(x) > 10**6:
                raise CertificateError(f"newton_int_matrix[{i}] entries must be integers with |x| <= 10^6")
            r_vals.append(x)
        M_int.append(r_vals)

    newton_val = newton_det_scaled_int(M_int)
    gauss_det = mat_det_exact([[Fraction(x) for x in row] for row in M_int])
    expected_scaled = factorial_exact(k) * gauss_det.numerator
    if gauss_det.denominator != 1 or newton_val != expected_scaled:
        raise CertificateError("Newton identity k!*det(M) disagreed with exact Gaussian determinant")
    claimed_det = _req_int(payload, "claimed_k_fact_det")
    if claimed_det != newton_val:
        raise CertificateError(f"claimed_k_fact_det={claimed_det} != exact {newton_val}")

    return {
        "mode": "reciprocal_scaling_and_newton_det_cert",
        "N_dim": N,
        "dim_C": len(C),
        "newton_k": k,
        "k_fact_det": newton_val,
    }


# ---------------------------------------------------------------------------
# Mode 3: Monotone LCP & Convex QP KKT Gap Decomposition (Secs. 7.2, 7.3)
# ---------------------------------------------------------------------------

def _verify_mode3(payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {
        "d_dim",
        "M",
        "q",
        "z_star",
        "w_star",
        "z_probe",
        "claimed_probe_gap",
        "claimed_quad_term",
    }
    _check_keys(payload, expected, "Mode 3 payload")

    d = _req_int(payload, "d_dim")
    if not (2 <= d <= 8):
        raise CertificateError("LCP dimension d_dim must lie in [2, 8]")

    M = _parse_frac_mat(payload.get("M"), d, d, "M")
    q = _parse_frac_vec(payload.get("q"), d, "q")
    z_star = _parse_frac_vec(payload.get("z_star"), d, "z_star")
    w_star = _parse_frac_vec(payload.get("w_star"), d, "w_star")
    z_probe = _parse_frac_vec(payload.get("z_probe"), d, "z_probe")

    # 1. Verify monotonicity (M + M^T PSD) and non-vacuous nonzero M
    if all(v == 0 for row in M for v in row):
        raise CertificateError("Matrix M cannot be identically zero")
    S = [[M[i][j] + M[j][i] for j in range(d)] for i in range(d)]
    if not is_symmetric_psd_exact(S):
        raise CertificateError("Matrix M is NOT monotone (M + M^T is not positive semidefinite)")

    # 2. Verify (z_star, w_star) is an exact LCP solution (z >= 0, w >= 0, w = Mz + q, z^T w = 0)
    if any(v < 0 for v in z_star) or any(v < 0 for v in w_star):
        raise CertificateError("LCP solution (z_star, w_star) must be nonnegative")
    Mz_q = [v + q[i] for i, v in enumerate(mat_vec_frac(M, z_star))]
    if Mz_q != w_star:
        raise CertificateError("LCP equation w_star == M * z_star + q violated")
    for i in range(d):
        if z_star[i] * w_star[i] != 0:
            raise CertificateError(f"Complementarity z_star[{i}] * w_star[{i}] != 0")

    # 3. Verify Section 7.2 exact gap decomposition on feasible non-identical probe z_probe
    if z_probe == z_star:
        raise CertificateError("z_probe must be distinct from z_star for non-vacuous gap audit")
    if any(v < 0 for v in z_probe):
        raise CertificateError("z_probe must be nonnegative")
    w_probe = [v + q[i] for i, v in enumerate(mat_vec_frac(M, z_probe))]
    if any(v < 0 for v in w_probe):
        raise CertificateError("w_probe = M * z_probe + q must be nonnegative")

    dz = [z_probe[i] - z_star[i] for i in range(d)]
    Mdz = mat_vec_frac(M, dz)
    quad_term = dot_frac(dz, Mdz)
    if quad_term < 0:
        raise CertificateError("Quadratic term (z' - z*)^T M (z' - z*) < 0 violates monotonicity")
    gap_direct = dot_frac(z_probe, w_probe)
    gap_split = dot_frac(z_probe, w_star) + dot_frac(z_star, w_probe) + quad_term
    if gap_direct != gap_split:
        raise CertificateError("Section 7.2 LCP gap decomposition identity failed")

    if _parse_frac_pair(payload.get("claimed_probe_gap"), "claimed_probe_gap") != gap_direct:
        raise CertificateError(f"claimed_probe_gap disagrees with exact {gap_direct}")
    if _parse_frac_pair(payload.get("claimed_quad_term"), "claimed_quad_term") != quad_term:
        raise CertificateError(f"claimed_quad_term disagrees with exact {quad_term}")

    return {
        "mode": "monotone_lcp_and_qp_kkt_cert",
        "d_dim": d,
        "probe_gap": f"{gap_direct.numerator}/{gap_direct.denominator}",
        "quad_term": f"{quad_term.numerator}/{quad_term.denominator}",
    }


# ---------------------------------------------------------------------------
# Mode 4: Max-Norm Contraction, Bellman Fixed Point & Clipped Update (Secs. 8, 9)
# ---------------------------------------------------------------------------

def _verify_mode4(payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {
        "n_states",
        "owners",
        "actions",
        "gamma_contraction",
        "v_fixed_point",
        "x_query",
        "y_center",
        "H_radius",
        "u_clipped",
        "claimed_new_err",
    }
    _check_keys(payload, expected, "Mode 4 payload")

    n = _req_int(payload, "n_states")
    if not (2 <= n <= 6):
        raise CertificateError("n_states must lie in [2, 6]")

    raw_owners = payload.get("owners")
    if not isinstance(raw_owners, list) or len(raw_owners) != n:
        raise CertificateError(f"owners must be a length-{n} list of 'max'/'min'")
    owners: List[str] = []
    for i, ow in enumerate(raw_owners):
        if ow not in {"max", "min"}:
            raise CertificateError(f"owners[{i}] must be 'max' or 'min'")
        owners.append(ow)

    gamma = _parse_frac_pair(payload.get("gamma_contraction"), "gamma_contraction")
    if not (Fraction(0) < gamma <= Fraction(1)):
        raise CertificateError("gamma_contraction must lie in (0, 1]")
    beta_max = Fraction(1) - gamma

    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != n:
        raise CertificateError(f"actions must contain {n} state action lists")

    parsed_actions: List[List[Tuple[Fraction, List[Fraction], List[Fraction]]]] = []
    for i, state_acts in enumerate(raw_actions):
        if not isinstance(state_acts, list) or not (1 <= len(state_acts) <= 6):
            raise CertificateError(f"actions[{i}] must contain 1..6 actions")
        s_list: List[Tuple[Fraction, List[Fraction], List[Fraction]]] = []
        for j, act in enumerate(state_acts):
            if not isinstance(act, Mapping):
                raise CertificateError(f"actions[{i}][{j}] must be a JSON object")
            _check_keys(act, {"r", "P", "B"}, f"actions[{i}][{j}]")
            r_a = _parse_frac_pair(act["r"], f"actions[{i}][{j}].r")
            P_a = _parse_frac_vec(act["P"], n, f"actions[{i}][{j}].P")
            B_a = _parse_frac_vec(act["B"], n, f"actions[{i}][{j}].B")
            if any(p < 0 for p in P_a) or sum(P_a) != Fraction(1):
                raise CertificateError(f"actions[{i}][{j}].P must be a valid probability distribution summing to 1")
            if any(b < 0 or b > p for b, p in zip(B_a, P_a)):
                raise CertificateError(f"actions[{i}][{j}].B must satisfy 0 <= B_a <= P_a coordinatewise")
            if sum(B_a) > beta_max:
                raise CertificateError(f"actions[{i}][{j}].B row sum exceeds 1 - gamma = {beta_max}")
            s_list.append((r_a, P_a, B_a))
        parsed_actions.append(s_list)

    v = _parse_frac_vec(payload.get("v_fixed_point"), n, "v_fixed_point")
    x = _parse_frac_vec(payload.get("x_query"), n, "x_query")
    y = _parse_frac_vec(payload.get("y_center"), n, "y_center")
    H = _parse_frac_pair(payload.get("H_radius"), "H_radius")
    u = _parse_frac_vec(payload.get("u_clipped"), n, "u_clipped")
    claimed_new_err = _parse_frac_pair(payload.get("claimed_new_err"), "claimed_new_err")

    if H <= 0:
        raise CertificateError("H_radius must be positive")

    # 1. Exact Bellman fixed point T(v) == v
    Tv = eval_bellman_operator(owners, parsed_actions, v)
    if Tv != v:
        raise CertificateError(f"v_fixed_point failed exact Bellman equation T(v) == v (got {Tv})")

    # 2. Exact Theorem 8 max-norm contraction & extremal coordinate movement sign on x_query != v
    if x == v:
        raise CertificateError("x_query must be distinct from v_fixed_point")
    Tx = eval_bellman_operator(owners, parsed_actions, x)
    err_x = max(abs(x[i] - v[i]) for i in range(n))
    err_Tx = max(abs(Tx[i] - v[i]) for i in range(n))
    if err_Tx > beta_max * err_x:
        raise CertificateError("Max-norm contraction inequality ||T(x) - v||_inf <= (1 - gamma)||x - v||_inf failed")
    for i in range(n):
        if abs(x[i] - v[i]) == err_x:
            # Movement T_i(x) - x_i must strictly share sign with v_i - x_i (Theorem 8 extremal label!)
            if (Tx[i] - x[i]) * (v[i] - x[i]) <= 0:
                raise CertificateError(f"Extremal coordinate {i} failed strict movement toward fixed point v")

    # 3. Exact Section 9.3 Eq. (9.3)-(9.5) clipped update audit at non-vacuous y_center != v
    if y == v:
        raise CertificateError("y_center must be distinct from v_fixed_point for non-vacuous clipped update")
    err_y = max(abs(v[i] - y[i]) for i in range(n))
    if err_y > Fraction(2) * H:
        raise CertificateError(f"y_center violates ||v - y||_inf <= 2H (got {err_y} > {2*H})")
    if any(abs(ui) > H for ui in u):
        raise CertificateError("u_clipped exceeds clipping box [-H, H]")
    u_eval = eval_clipped_operator(owners, parsed_actions, y, u, H)
    if u_eval != u:
        raise CertificateError(f"u_clipped failed exact Eq. (9.3) clipped fixed-point equation (got {u_eval})")

    new_err = max(abs(v[i] - (y[i] + u[i])) for i in range(n))
    if new_err > H:
        raise CertificateError(f"Eq. (9.4) error halving ||v - (y+u)||_inf <= H failed (got {new_err})")
    if new_err != claimed_new_err:
        raise CertificateError(f"claimed_new_err disagrees with exact {new_err}")

    # If y < v componentwise, certify Eq. (9.5) lower policy guarantee z = y + u - H*1 <= T(z) and z <= v!
    low_guarantee_verified = False
    if all(y[i] < v[i] for i in range(n)):
        z = [y[i] + u[i] - H for i in range(n)]
        Tz = eval_bellman_operator(owners, parsed_actions, z)
        if any(z[i] > Tz[i] for i in range(n)) or any(z[i] > v[i] for i in range(n)):
            raise CertificateError("Eq. (9.5) low-center policy guarantee z <= T(z) and z <= v failed")
        low_guarantee_verified = True

    return {
        "mode": "max_norm_contraction_and_game_cert",
        "n_states": n,
        "err_y": f"{err_y.numerator}/{err_y.denominator}",
        "new_err": f"{new_err.numerator}/{new_err.denominator}",
        "low_guarantee": low_guarantee_verified,
    }


# ---------------------------------------------------------------------------
# Public Entry Points
# ---------------------------------------------------------------------------

def verify_certificate_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise CertificateError("Certificate must be a mapping")
    allowed_top = {"protocol", "certificate_id", "mode", "payload", "description", "note"}
    extra_top = set(data.keys()) - allowed_top
    if extra_top:
        raise CertificateError(f"Unexpected top-level keys: {sorted(extra_top)}")
    for opt_k in ("description", "note"):
        if opt_k in data and not isinstance(data[opt_k], str):
            raise CertificateError(f"Optional field {opt_k!r} must be a string")

    proto = _req_str(data, "protocol")
    if proto != PROTOCOL_ID:
        raise CertificateError(f"Unsupported protocol {proto!r}; expected {PROTOCOL_ID!r}")
    cert_id = _req_str(data, "certificate_id")
    mode = _req_str(data, "mode")
    if mode not in VALID_MODES:
        raise CertificateError(f"Unknown certificate mode {mode!r}")

    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        raise CertificateError("Field 'payload' must be a JSON object")

    try:
        if mode == "lp_trichotomy_certificate_cert":
            summary = _verify_mode1(payload)
        elif mode == "reciprocal_scaling_and_newton_det_cert":
            summary = _verify_mode2(payload)
        elif mode == "monotone_lcp_and_qp_kkt_cert":
            summary = _verify_mode3(payload)
        else:
            summary = _verify_mode4(payload)
    except CertificateError:
        raise
    except ValueError as exc:
        raise CertificateError(f"Exact verification value error: {exc}") from exc

    summary["certificate_id"] = cert_id
    summary["protocol"] = PROTOCOL_ID
    return summary


def verify_certificate_str(raw: str) -> Dict[str, Any]:
    return verify_certificate_dict(load_strict_json_str(raw))


def verify_certificate_file(path: Path | str) -> Dict[str, Any]:
    return verify_certificate_str(Path(path).read_text(encoding="utf-8"))
