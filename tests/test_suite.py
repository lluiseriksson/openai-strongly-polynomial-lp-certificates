"""Comprehensive 80-test suite for SPLP-SMALE9-CERT-1.0.

Includes:
  - 1 aggregate canonical 16-certificate replay,
  - 4 raw-JSON zero-float / NaN / duplicate-key / oversized-payload parser tests,
  - 2 exact arithmetic soundness regressions (Sec. 3.4 Newton identities & PSD LDL^T check),
  - 73 parametrized live adversarial forgery tests (12 schema/parser + 15 Mode 1 + 16 Mode 2 +
    15 Mode 3 + 15 Mode 4), each asserting an exact regex match on the intended CertificateError branch.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path
import pytest

from src.splp_certifier import (
    PROTOCOL_ID,
    verify_certificate_dict,
    verify_certificate_file,
    verify_certificate_str,
)
from src.splp_exact import (
    CertificateError,
    is_symmetric_psd_exact,
    mat_det_exact,
    newton_det_scaled_int,
)

ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "certificates"


def _load(name: str) -> dict:
    return json.loads((CERT_DIR / f"{name}.cert.json").read_text(encoding="utf-8"))


def test_all_16_canonical_certificates_verify() -> None:
    files = sorted(CERT_DIR.glob("*.cert.json"))
    assert len(files) == 16
    for f in files:
        res = verify_certificate_file(f)
        assert res["protocol"] == PROTOCOL_ID


def test_raw_json_rejects_float_literal() -> None:
    raw = '{"protocol": "SPLP-SMALE9-CERT-1.0", "certificate_id": "x", "mode": "lp_trichotomy_certificate_cert", "payload": {"m_rows": 2.5}}'
    with pytest.raises(CertificateError, match="Floating-point literal rejected"):
        verify_certificate_str(raw)


def test_raw_json_rejects_nan_constant() -> None:
    raw = '{"protocol": "SPLP-SMALE9-CERT-1.0", "certificate_id": "x", "mode": "lp_trichotomy_certificate_cert", "payload": {"m_rows": NaN}}'
    with pytest.raises(CertificateError, match="Non-standard JSON constant rejected"):
        verify_certificate_str(raw)


def test_raw_json_rejects_duplicate_keys() -> None:
    raw = '{"protocol": "SPLP-SMALE9-CERT-1.0", "protocol": "SPLP-SMALE9-CERT-1.0"}'
    with pytest.raises(CertificateError, match="Duplicate JSON key rejected"):
        verify_certificate_str(raw)


def test_raw_json_rejects_oversized_payload() -> None:
    raw = '{"protocol": "SPLP-SMALE9-CERT-1.0", "note": "' + ("a" * 1_000_050) + '"}'
    with pytest.raises(CertificateError, match="exceeds MAX_JSON_BYTES"):
        verify_certificate_str(raw)


def test_regression_newton_det_matches_gaussian() -> None:
    M = [[3, 1, 0, -2], [1, 4, -1, 1], [0, 2, 5, -3], [-1, 0, 2, 2]]
    nd = newton_det_scaled_int(M)
    gd = mat_det_exact([[Fraction(x) for x in r] for r in M])
    assert gd.denominator == 1 and nd == 24 * gd.numerator


def test_regression_psd_exact_indefinite_rejected() -> None:
    assert is_symmetric_psd_exact([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]])
    assert not is_symmetric_psd_exact([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]])
    assert not is_symmetric_psd_exact([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])


def _build_73_forgeries() -> list[tuple[str, dict, str]]:
    m1_opt = _load("m1_lp_optimal_2d_strict_comp")
    m1_unb = _load("m1_lp_unbounded_recession_ray")
    m1_inf = _load("m1_lp_infeasible_farkas_minsupp")
    m2 = _load("m2_recip_circuit_u10_newton2")
    m3 = _load("m3_lcp_3d_asym_monotone")
    m4 = _load("m4_game_fig5_low_center")

    cases: list[tuple[str, dict, str]] = []

    def add(name: str, base: dict, mutator, regex: str) -> None:
        c = copy.deepcopy(base)
        mutator(c)
        cases.append((name, c, regex))

    # --- Schema / Top-Level (12) ---
    add("S01_bad_protocol", m1_opt, lambda c: c.__setitem__("protocol", "FAKE-1.0"), "Unsupported protocol")
    add("S02_whitespace_cert_id", m1_opt, lambda c: c.__setitem__("certificate_id", "  "), "non-empty string")
    add("S03_unknown_mode", m1_opt, lambda c: c.__setitem__("mode", "bogus_mode"), "Unknown certificate mode")
    add("S04_extra_top_key", m1_opt, lambda c: c.__setitem__("rogue", 1), "Unexpected top-level keys")
    add("S05_bad_description", m1_opt, lambda c: c.__setitem__("description", 99), "Optional field 'description' must be a string")
    add("S06_bad_note", m1_opt, lambda c: c.__setitem__("note", True), "Optional field 'note' must be a string")
    add("S07_payload_not_obj", m1_opt, lambda c: c.__setitem__("payload", "bad"), "must be a JSON object")
    add("S08_missing_protocol", m1_opt, lambda c: c.pop("protocol"), "Missing required string field 'protocol'")
    add("S09_bool_as_int", m1_opt, lambda c: c["payload"].__setitem__("m_rows", True), "must be an exact integer")
    add("S10_float_in_dict", m1_opt, lambda c: c["payload"].__setitem__("n_cols", 2.0), "must be an exact integer")
    add("S11_missing_key", m1_opt, lambda c: c["payload"].pop("A"), "Missing keys in Mode 1 payload")
    add("S12_extra_key", m1_opt, lambda c: c["payload"].__setitem__("extra", 1), "Unexpected keys in Mode 1 payload")

    # --- Mode 1 Forgeries (15) ---
    add("M1_01_unknown_branch", m1_opt, lambda c: c["payload"].__setitem__("branch", "mystery"), "Unknown Theorem 1 branch")
    add("M1_02_m_too_small", m1_opt, lambda c: c["payload"].__setitem__("m_rows", 1), "LP dimensions must satisfy")
    add("M1_03_unreduced_frac_b", m1_opt, lambda c: c["payload"]["b"].__setitem__(0, [8, 2]), "lowest terms \\(gcd == 1\\)")
    add("M1_04_primal_infeasible", m1_opt, lambda c: c["payload"]["witness"]["x"].__setitem__(0, [5, 1]), "Primal infeasibility")
    add("M1_05_dual_neg", m1_opt, lambda c: c["payload"]["witness"]["y"].__setitem__(2, [-1, 1]), "Dual infeasibility")
    add("M1_06_dual_stat_broken", m1_opt, lambda c: c["payload"]["witness"]["y"].__setitem__(0, [2, 1]), "Dual stationarity")
    add("M1_07_opt_val_wrong", m1_opt, lambda c: c["payload"]["witness"].__setitem__("claimed_opt_value", [9, 1]), "Strong duality gap nonzero or opt mismatch")
    add("M1_08_strict_comp_violated", m1_opt, lambda c: (c["payload"]["b"].__setitem__(2, [-2, 1]), c["payload"]["witness"]["x"].__setitem__(0, [2, 1])), "Strict complementarity")
    add("M1_09_unb_ray_not_recession", m1_unb, lambda c: c["payload"]["witness"]["r"].__setitem__(0, [5, 1]), "recession condition")
    add("M1_10_unb_ray_nonpos_gain", m1_unb, lambda c: c["payload"]["witness"].__setitem__("r", [[0, 1], [0, 1]]), "Improving ray gain")
    add("M1_11_inf_farkas_neg_y", m1_inf, lambda c: c["payload"]["witness"]["y"].__setitem__(0, [-1, 1]), "Farkas witness violates y >= 0")
    add("M1_12_inf_farkas_ATy_nz", m1_inf, lambda c: c["payload"]["witness"]["y"].__setitem__(3, [1, 1]), "Farkas witness violates A\\^T y == 0")
    add("M1_13_inf_wrong_rank", m1_inf, lambda c: c["payload"]["witness"].__setitem__("claimed_rank_A", 1), "claimed_rank_A=1 != exact rank 2")
    add("M1_14_inf_wrong_rhs", m1_inf, lambda c: c["payload"]["witness"].__setitem__("claimed_farkas_rhs", [-1, 1]), "Farkas witness b\\^T y")
    add("M1_15_rational_mag_cap", m1_opt, lambda c: c["payload"]["b"].__setitem__(0, [10**13 + 1, 1]), "exceed magnitude cap")

    # --- Mode 2 Forgeries (16) ---
    add("M2_01_N_too_small", m2, lambda c: c["payload"].__setitem__("N_dim", 1), "N_dim must lie in")
    add("M2_02_N_too_large", m2, lambda c: c["payload"].__setitem__("N_dim", 9), "N_dim must lie in")
    add("M2_03_empty_C_basis", m2, lambda c: c["payload"].__setitem__("C_basis", []), "must be non-empty lists")
    add("M2_04_dep_C_basis", m2, lambda c: c["payload"]["C_basis"].__setitem__(1, list(c["payload"]["C_basis"][0])), "linearly independent rows")
    add("M2_05_dim_sum_mismatch", m2, lambda c: c["payload"]["C_perp_basis"].pop(), "dim\\(C\\) \\+ dim\\(C_perp\\)")
    add("M2_06_not_orthogonal", m2, lambda c: c["payload"]["C_perp_basis"][0].__setitem__(0, [1, 1]), "are not orthogonal")
    add("M2_07_prices_bad_len", m2, lambda c: c["payload"]["prices_p"].pop(), "length-4 integer list")
    add("M2_08_prices_oob", m2, lambda c: c["payload"]["prices_p"].__setitem__(0, 99), "integer in \\[-16, 16\\]")
    add("M2_09_prices_bool", m2, lambda c: c["payload"]["prices_p"].__setitem__(0, True), "integer in \\[-16, 16\\]")
    add("M2_10_newton_k_small", m2, lambda c: c["payload"].__setitem__("newton_int_matrix", [[1]]), "k x k with 2 <= k <= 5")
    add("M2_11_newton_nonsquare", m2, lambda c: c["payload"]["newton_int_matrix"][0].append(1), "must have length 2")
    add("M2_12_newton_entry_oob", m2, lambda c: c["payload"]["newton_int_matrix"][0].__setitem__(0, 10**9), "\\|x\\| <= 10\\^6")
    add("M2_13_wrong_newton_det", m2, lambda c: c["payload"].__setitem__("claimed_k_fact_det", 0), "claimed_k_fact_det=0 != exact")
    add("M2_14_unreduced_C_entry", m2, lambda c: c["payload"]["C_basis"][0].__setitem__(0, [2, 2]), "lowest terms")
    add("M2_15_zero_prices_vacuous", m2, lambda c: c["payload"].__setitem__("prices_p", [0, 0, 0, 0]), "must be nonzero")
    add("M2_16_dos_rows_early_reject", m2, lambda c: c["payload"].__setitem__("C_basis", [list(c["payload"]["C_basis"][0])] * 500), "dim\\(C\\) \\+ dim\\(C_perp\\)")

    # --- Mode 3 Forgeries (15) ---
    add("M3_01_d_too_small", m3, lambda c: c["payload"].__setitem__("d_dim", 1), "d_dim must lie in")
    add("M3_02_d_too_large", m3, lambda c: c["payload"].__setitem__("d_dim", 9), "d_dim must lie in")
    add("M3_03_not_monotone_neg_diag", m3, lambda c: c["payload"]["M"][0].__setitem__(0, [-1, 1]), "NOT monotone")
    add("M3_04_not_monotone_indef", m3, lambda c: (c["payload"]["M"][0].__setitem__(1, [5, 1]), c["payload"]["M"][1].__setitem__(0, [5, 1])), "NOT monotone")
    add("M3_05_z_star_neg", m3, lambda c: c["payload"]["z_star"].__setitem__(0, [-1, 1]), "must be nonnegative")
    add("M3_06_w_star_mismatch", m3, lambda c: c["payload"]["w_star"].__setitem__(0, [1, 1]), "w_star == M \\* z_star \\+ q violated")
    add("M3_07_comp_violated", m3, lambda c: (c["payload"]["q"].__setitem__(0, [1, 1]), c["payload"]["w_star"].__setitem__(0, [1, 1])), "Complementarity z_star\\[0\\] \\* w_star\\[0\\] != 0")
    add("M3_08_vacuous_probe_eq_star", m3, lambda c: c["payload"].__setitem__("z_probe", list(c["payload"]["z_star"])), "distinct from z_star")
    add("M3_09_z_probe_neg", m3, lambda c: c["payload"]["z_probe"].__setitem__(0, [-1, 1]), "z_probe must be nonnegative")
    add("M3_10_w_probe_neg", m3, lambda c: c["payload"]["z_probe"].__setitem__(2, [0, 1]), "w_probe = M \\* z_probe \\+ q must be nonnegative")
    add("M3_11_wrong_probe_gap", m3, lambda c: c["payload"].__setitem__("claimed_probe_gap", [0, 1]), "claimed_probe_gap disagrees")
    add("M3_12_wrong_quad_term", m3, lambda c: c["payload"].__setitem__("claimed_quad_term", [0, 1]), "claimed_quad_term disagrees")
    add("M3_13_unreduced_q", m3, lambda c: c["payload"]["q"].__setitem__(2, [-10, 2]), "lowest terms")
    add("M3_14_M_bad_row_len", m3, lambda c: c["payload"]["M"][0].pop(), "length-3 rational vector")
    add("M3_15_zero_matrix_vacuous", m3, lambda c: c["payload"].__setitem__("M", [[[0, 1]] * 3 for _ in range(3)]), "cannot be identically zero")

    # --- Mode 4 Forgeries (15) ---
    add("M4_01_n_too_small", m4, lambda c: c["payload"].__setitem__("n_states", 1), "n_states must lie in")
    add("M4_02_bad_owner", m4, lambda c: c["payload"]["owners"].__setitem__(0, "chance"), "must be 'max' or 'min'")
    add("M4_03_gamma_zero", m4, lambda c: c["payload"].__setitem__("gamma_contraction", [0, 1]), "gamma_contraction must lie in")
    add("M4_04_P_not_prob", m4, lambda c: c["payload"]["actions"][0][0]["P"].__setitem__(0, [2, 1]), "valid probability distribution")
    add("M4_05_B_gt_P", m4, lambda c: c["payload"]["actions"][0][0]["B"].__setitem__(1, [1, 4]), "0 <= B_a <= P_a")
    add("M4_06_B_exceeds_beta_max", m4, lambda c: c["payload"].__setitem__("gamma_contraction", [3, 4]), "exceeds 1 - gamma")
    add("M4_07_H_nonpos", m4, lambda c: c["payload"].__setitem__("H_radius", [0, 1]), "H_radius must be positive")
    add("M4_08_bad_fixed_point", m4, lambda c: c["payload"]["v_fixed_point"].__setitem__(0, [2, 1]), "failed exact Bellman equation")
    add("M4_09_x_eq_v_vacuous", m4, lambda c: c["payload"].__setitem__("x_query", list(c["payload"]["v_fixed_point"])), "distinct from v_fixed_point")
    add("M4_10_y_too_far_gt_2H", m4, lambda c: c["payload"].__setitem__("y_center", [[-2, 1], [0, 1]]), "violates \\|\\|v - y\\|\\|_inf <= 2H")
    add("M4_11_u_exceeds_H", m4, lambda c: c["payload"]["u_clipped"].__setitem__(0, [2, 1]), "exceeds clipping box")
    add("M4_12_u_not_clipped_fp", m4, lambda c: c["payload"]["u_clipped"].__setitem__(1, [0, 1]), "failed exact Eq\\. \\(9\\.3\\)")
    add("M4_13_wrong_new_err", m4, lambda c: c["payload"].__setitem__("claimed_new_err", [1, 1]), "claimed_new_err disagrees")
    add("M4_14_unreduced_H", m4, lambda c: c["payload"].__setitem__("H_radius", [2, 2]), "lowest terms")
    add("M4_15_vacuous_y_eq_v", m4, lambda c: c["payload"].__setitem__("y_center", list(c["payload"]["v_fixed_point"])), "distinct from v_fixed_point")

    assert len(cases) == 73
    return cases


FORGERIES = _build_73_forgeries()


@pytest.mark.parametrize("name,cert,expected_regex", FORGERIES, ids=[c[0] for c in FORGERIES])
def test_adversarial_forgery_rejected(name: str, cert: dict, expected_regex: str) -> None:
    with pytest.raises(CertificateError, match=expected_regex):
        verify_certificate_dict(cert)
