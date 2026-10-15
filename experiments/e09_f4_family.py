"""
E9: the family F_a(x) = x^3 + a^-1 Tr(a^3 x^9) in dimension n = 2m with m odd (Section 8).

Since F_a(cx) = c^3 F_{a c^3}(x), the class of F_a only depends on a modulo cubes, so
a in {1, alpha, alpha^2} (alpha = 2 primitive) covers every member. For each n and a we
compute the m-dimensional subspaces of the set of non-bent components, in field
coordinates (component x -> Tr(beta F(x))), test whether each one is a line c*GF(2^m) of the
Desarguesian spread and whether it equals a*GF(2^m), and decide CCZ-equivalence to a
permutation (Algorithm 1). We also count the m- and (m-1)-dimensional subspaces of the set
of cubes and check that each lies in a spread line.

    python3 e09_f4_family.py            # n = 6, 10, 14 (n = 18: e10_f4_n18.py)
"""
import itertools
import time

import numpy as np

from common import parser, save
from apnsearch import ccz, fast, families
from apnsearch.gf import GF


def dual_basis_map(g):
    """L with b . y = Tr(L(b) y) for all b, y (bitwise dot product vs trace form)."""
    n = g.n
    beta = []
    for i in range(n):
        for x in range(1, g.N):
            if all(g.trace(g.mul(x, 1 << j)) == (1 if j == i else 0) for j in range(n)):
                beta.append(x)
                break
    def L(b):
        r = 0
        for i in range(n):
            if b >> i & 1:
                r ^= beta[i]
        return r
    return L


def span(B):
    s = {0}
    for v in B:
        s |= {x ^ int(v) for x in s}
    return s


def analyse(n, a, L, sub):
    g = GF(n)
    m = n // 2
    t = time.time()
    F = families.x3_tr_x9(n, a)
    rows = ccz.row_structure_quadratic(F, n)
    mask = np.zeros(g.N, dtype=np.uint8)
    for b in rows:
        mask[b] = 1
    S = fast.find_subspaces(mask, m)
    larger = fast.find_subspaces(mask, m + 1)
    spaces = [{L(w) for w in span(B)} for B in S]
    def line_through(w):
        return {g.mul(w, x) for x in sub}
    is_line = [W == line_through(next(x for x in W if x)) for W in spaces]
    a_line = frozenset(g.mul(a, x) for x in sub)
    has_a_line = any(frozenset(W) == a_line for W in spaces)
    spanning = any(len(x & y) == 1 for x, y in itertools.combinations(spaces, 2))
    st, _ = ccz.ccz_to_permutation_projection(F, n, quadratic=True)
    cubes = {g.pow(x, 3) for x in range(1, g.N)}
    r = {"n": n, "a": a, "a_is_cube": a in cubes, "nonbent": len(rows),
         "subspaces_dim_m": len(S), "subspaces_dim_m_plus_1": 0 if larger is None else len(larger),
         "all_spread_lines": all(is_line), "contains_a_line": has_a_line,
         "condition_inconclusive": spanning, "verdict": st,
         "seconds": round(time.time() - t, 1)}
    print(r, flush=True)
    return r


def cube_subspaces(n, sub):
    g = GF(n)
    m = n // 2
    mask = np.zeros(g.N, dtype=np.uint8)
    for x in range(1, g.N):
        mask[g.pow(x, 3)] = 1
    out = {"n": n}
    for d in (m, m - 1):
        S = fast.find_subspaces(mask, d, cap=5_000_000)
        inside = sum(span(B) <= {g.mul(int(B[0]), x) for x in sub} for B in S)
        out[f"dim_{d}"] = len(S)
        out[f"dim_{d}_inside_a_line"] = inside
    print(out, flush=True)
    return out


def main():
    a = parser(__doc__).parse_args()
    dims = (6, 10, 14)
    rows, cubes = [], []
    for n in dims:
        g = GF(n)
        q = 1 << (n // 2)
        L = dual_basis_map(g)
        sub = [x for x in range(g.N) if g.pow(x, q) == x]
        for aa in (1, 2, 4):          # 1, alpha, alpha^2: all classes modulo cubes
            rows.append(analyse(n, aa, L, sub))
        if n <= 14:
            cubes.append(cube_subspaces(n, sub))
    save("e09_f4_family", {"rows": rows, "cube_subspaces": cubes})


if __name__ == "__main__":
    main()
