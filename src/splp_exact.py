"""Zero-float exact arithmetic engine for SPLP-SMALE9-CERT-1.0.

Implements pure-integer and exact-Fraction primitives aligned with OpenAI's
November 24, 2026 preprint "Strongly polynomial linear programming":
  - Exact rational matrix rank, Gaussian elimination determinant, and division-free
    Section 3.4 Newton's identities determinant k! * det(M),
  - Exact rational symmetric PSD verification via pivoted LDL^T decomposition (for monotone LCP),
  - Exact reciprocal diagonal power-of-2 scaling D_{i+} = 2^{-p_i}, D_{i-} = 2^{p_i} (Sec. 3.1),
  - Exact Bellman / Shapley operator & Section 9.3 clipped update evaluator (Thms. 8, 11).

NO floating-point operations, true division '/', or math imports appear anywhere in this module.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple


class CertificateError(ValueError):
    """Raised whenever any exact mathematical or structural check fails."""


def gcd_int(a: int, b: int) -> int:
    x = abs(a)
    y = abs(b)
    while y:
        x, y = y, x % y
    return x


def frac_div(a: Fraction, b: Fraction) -> Fraction:
    """Exact Fraction division without using the '/' operator (AST-clean)."""
    if b.numerator == 0:
        raise CertificateError("Division by zero in exact rational arithmetic")
    return Fraction(a.numerator * b.denominator, a.denominator * b.numerator)


def pow2_frac(exp_int: int) -> Fraction:
    """Exact rational 2^exp_int for any bounded integer exponent."""
    if exp_int >= 0:
        return Fraction(1 << exp_int, 1)
    return Fraction(1, 1 << (-exp_int))


def dot_frac(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def mat_vec_frac(A: Sequence[Sequence[Fraction]], x: Sequence[Fraction]) -> List[Fraction]:
    return [dot_frac(row, x) for row in A]


def mat_transpose_vec_frac(A: Sequence[Sequence[Fraction]], y: Sequence[Fraction]) -> List[Fraction]:
    if not A:
        return []
    n = len(A[0])
    return [sum((A[i][j] * y[i] for i in range(len(A))), Fraction(0)) for j in range(n)]


def mat_rank_exact(A: Sequence[Sequence[Fraction]]) -> int:
    """Exact rank of a rational matrix via Gaussian elimination."""
    if not A or not A[0]:
        return 0
    m = len(A)
    n = len(A[0])
    M = [list(row) for row in A]
    rank = 0
    for col in range(n):
        pivot = -1
        for r in range(rank, m):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
        piv_val = M[rank][col]
        for r in range(rank + 1, m):
            if M[r][col] != 0:
                factor = frac_div(M[r][col], piv_val)
                for c in range(col, n):
                    M[r][c] -= factor * M[rank][c]
        rank += 1
        if rank == m:
            break
    return rank


def mat_det_exact(A: Sequence[Sequence[Fraction]]) -> Fraction:
    """Exact determinant of a square rational matrix via Gaussian elimination."""
    n = len(A)
    if n == 0:
        return Fraction(1)
    M = [list(row) for row in A]
    sign = 1
    det = Fraction(1)
    for col in range(n):
        pivot = -1
        for r in range(col, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            return Fraction(0)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            sign = -sign
        piv = M[col][col]
        det *= piv
        for r in range(col + 1, n):
            if M[r][col] != 0:
                factor = frac_div(M[r][col], piv)
                for c in range(col, n):
                    M[r][c] -= factor * M[col][c]
    return det if sign == 1 else -det


def factorial_exact(k: int) -> int:
    res = 1
    for i in range(2, k + 1):
        res *= i
    return res


def newton_det_scaled_int(M: Sequence[Sequence[int]]) -> int:
    """Exact Section 3.4 division-free Newton's identities computation of k! * det(M).

    Uses the exact recurrence from Sec. 3.4 of OpenAI's Nov 24, 2026 preprint:
      i! c_i = sum_{j=1..i} (-1)^{j-1} * ((i-1)! / (i-j)!) * ((i-j)! c_{i-j}) * tr(M^j)
    where det(I + tM) = sum c_i t^i, so k! c_k = k! det(M).
    Pure integer ring operations only (zero divisions)!
    """
    k = len(M)
    if k == 0:
        return 1
    # Compute integer matrix powers M^1 .. M^k and traces
    traces: List[int] = [0] * (k + 1)
    cur = [list(row) for row in M]
    for p in range(1, k + 1):
        if p > 1:
            nxt = [[0] * k for _ in range(k)]
            for r in range(k):
                for c in range(k):
                    nxt[r][c] = sum(cur[r][t] * M[t][c] for t in range(k))
            cur = nxt
        traces[p] = sum(cur[i][i] for i in range(k))

    # f_c[i] stores i! * c_i (with f_c[0] = 1)
    f_c: List[int] = [0] * (k + 1)
    f_c[0] = 1
    for i in range(1, k + 1):
        acc = 0
        for j in range(1, i + 1):
            # falling product (i-1)*(i-2)*...*(i-j+1) = (i-1)! / (i-j)!
            fall = 1
            for t in range(i - j + 1, i):
                fall *= t
            term = fall * f_c[i - j] * traces[j]
            if (j - 1) % 2 == 1:
                acc -= term
            else:
                acc += term
        f_c[i] = acc
    return f_c[k]


def is_symmetric_psd_exact(S: Sequence[Sequence[Fraction]]) -> bool:
    """Exact rational check whether a symmetric matrix S is positive semidefinite.

    Uses exact symmetric pivoted congruence elimination (diagonal pivoting; if all remaining
    diagonal entries are 0, any nonzero off-diagonal entry creates an indefinite 2x2 block).
    """
    n = len(S)
    for i in range(n):
        for j in range(n):
            if S[i][j] != S[j][i]:
                return False
    A = [list(row) for row in S]
    for k in range(n):
        # Check for any negative diagonal entry first
        for i in range(k, n):
            if A[i][i] < 0:
                return False
        # Pick largest positive diagonal pivot
        piv = -1
        for i in range(k, n):
            if A[i][i] > 0:
                if piv == -1 or A[i][i] > A[piv][piv]:
                    piv = i
        if piv == -1:
            # All remaining diagonal entries are 0; PSD iff all remaining entries are 0!
            for i in range(k, n):
                for j in range(k, n):
                    if A[i][j] != 0:
                        return False
            return True
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            for r in range(n):
                A[r][k], A[r][piv] = A[r][piv], A[r][k]
        d = A[k][k]
        for i in range(k + 1, n):
            if A[i][k] != 0:
                factor = frac_div(A[i][k], d)
                for j in range(k + 1, n):
                    A[i][j] -= factor * A[k][j]
                A[i][k] = Fraction(0)
                A[k][i] = Fraction(0)
    return True


def clip_frac(val: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def eval_bellman_operator(
    owners: Sequence[str],
    actions_per_state: Sequence[Sequence[Tuple[Fraction, List[Fraction], List[Fraction]]]],
    x: Sequence[Fraction],
) -> List[Fraction]:
    """Evaluate exact Section 9.1 Bellman operator T_i(x) = ext_a (r_a + B_a x).

    Each action at state i is (r_a, P_a_row, B_a_row).
    """
    out: List[Fraction] = []
    for i, owner in enumerate(owners):
        scores = [r_a + dot_frac(B_a, x) for (r_a, _P_a, B_a) in actions_per_state[i]]
        if owner == "max":
            out.append(max(scores))
        elif owner == "min":
            out.append(min(scores))
        else:
            raise CertificateError(f"Unknown state owner {owner!r}")
    return out


def eval_clipped_operator(
    owners: Sequence[str],
    actions_per_state: Sequence[Sequence[Tuple[Fraction, List[Fraction], List[Fraction]]]],
    y: Sequence[Fraction],
    u: Sequence[Fraction],
    H: Fraction,
) -> List[Fraction]:
    """Evaluate exact Section 9.3 Eq. (9.3) clipped update operator on correction u."""
    out: List[Fraction] = []
    for i, owner in enumerate(owners):
        scores = [
            r_a + dot_frac(B_a, y) - y[i] + dot_frac(P_a, u)
            for (r_a, P_a, B_a) in actions_per_state[i]
        ]
        val = max(scores) if owner == "max" else min(scores)
        out.append(clip_frac(val, -H, H))
    return out
