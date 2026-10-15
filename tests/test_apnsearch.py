"""Fast checks of the core routines (run with: python3 -m pytest tests -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apnsearch import ccz, families, fast  # noqa: E402
from apnsearch.gf import GF  # noqa: E402
from apnsearch.invariants import algebraic_degree  # noqa: E402
from apnsearch.switching import switch_once  # noqa: E402


def ddt_cost_reference(S):
    N = len(S)
    c = 0
    for a in range(1, N):
        cnt = np.bincount([S[x] ^ S[x ^ a] for x in range(N)], minlength=N)
        c += int((cnt * (cnt - 2)).sum()) // 8
    return c


def test_field_and_cost():
    for n in (3, 5, 6):
        g = GF(n)
        assert all(g.mul(x, g.inv(x)) == 1 for x in range(1, g.N))
        rng = np.random.default_rng(n)
        for _ in range(5):
            S = rng.permutation(g.N)
            assert fast.ddt_cost(S) == ddt_cost_reference(S)
    assert fast.is_apn(GF(7).power_table(3))


def test_swap_deltas_match_recomputation():
    S = np.random.default_rng(1).permutation(64).astype(np.int32)
    D = fast.swap_deltas(S)
    for i, j in [(0, 1), (3, 40), (17, 63)]:
        T = S.copy()
        T[i], T[j] = T[j], T[i]
        assert D[i, j] == fast.ddt_cost(T) - fast.ddt_cost(S)


def test_switching_kernel_elements_are_apn():
    F = GF(6).power_table(3)
    for u in (1, 2):
        K = fast.switching_kernel(F, u)
        rng = np.random.default_rng(u)
        for _ in range(20):
            f = (rng.integers(0, 2, len(K)) @ K) % 2
            assert fast.is_apn(F ^ (u * f))


def test_edel_pott_found_by_switching():
    res = switch_once(families.edel_pott_source6(), 6)
    assert any(nq for _, _, _, nq, _ in res)


def test_dual_matches_exhaustive_n6():
    g = GF(6)
    u, kim = families.kim6()
    for F in (g.power_table(3), kim):
        st_e, _ = ccz.ccz_to_permutation_exhaustive(F, 6)
        st_d, info = ccz.ccz_to_permutation_projection(F, 6, mode="all")
        assert st_e == st_d
        assert info["subspaces"] == ccz.count_walsh_zero_spaces(F, 6)
    st, info = ccz.ccz_to_permutation_projection(kim, 6)
    assert st == "permutation" and info["permutation_is_apn"]
    assert algebraic_degree(np.array(info["permutation"]), 6) == 4


def test_bilinear_filter_agrees_with_ddt():
    g = GF(8)
    rng = np.random.default_rng(0)
    exps = [3, 6, 9, 12, 18, 33, 48, 66]
    Fs = []
    for _ in range(60):
        F = np.zeros(256, dtype=np.int64)
        for e in rng.choice(exps, 2, replace=False):
            F ^= g.mul_vec(np.full(256, int(rng.integers(1, 256))), g.power_table(int(e)))
        Fs.append(F)
    flags = fast.bilinear_apn_filter(np.array([fast.bilinear_matrix(F, 8) for F in Fs]), 8)
    assert list(flags) == [fast.is_apn(F) for F in Fs]


def test_symmetric_subspace_search_matches_plain():
    from apnsearch import symmetry
    g = GF(10)
    mask = np.zeros(g.N, dtype=np.uint8)
    for x in range(1, g.N):
        mask[g.pow(x, 3)] = 1                     # the cubes, invariant under squaring
    squaring = [g.pow(1 << i, 2) for i in range(10)]
    plain = {symmetry.rref(B) for B in fast.find_subspaces(mask, 4)}
    sym = {tuple(int(v) for v in B) for B in symmetry.find_subspaces_sym(mask, 4, squaring, workers=4)}
    assert len(plain) == 341 and sym == plain


def test_ea_equivalence_certificate():
    from apnsearch import equivalence
    rng = np.random.default_rng(3)
    n = 6

    def rand_perm_cols():
        while True:
            cols = [int(x) for x in rng.integers(1, 1 << n, n)]
            if equivalence.rank(cols) == n:
                return cols

    g = GF(n)
    F = g.power_table(3)
    X = np.arange(1 << n)
    G = equivalence.apply(rand_perm_cols(), F[equivalence.apply(rand_perm_cols(), X)])
    G = G ^ equivalence.apply([int(x) for x in rng.integers(0, 1 << n, n)], X) ^ 5
    cert = equivalence.ea_equivalence(F, G, n)
    assert cert is not None and equivalence.check_certificate(F, G, n, *cert)
    # x^3 and the Kim function are inequivalent; the exhaustive search must say so
    assert equivalence.ea_equivalence(F, families.kim6()[1], n) is None
