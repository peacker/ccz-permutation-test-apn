"""
E2: structured permutation families in even dimension (paper, Section 4).

(a) Open butterflies H(x, y) = (R_t(y), t), t = R_y^{-1}(x), on n = 2k bits with
    R_z(x) = (x + alpha z)^e + beta z^f, x^e a permutation of GF(2^k): exhaustive over
    e (one representative per cyclotomic class of exponents coprime to 2^k - 1), f in
    {0, ..., 2^k - 2} (z^0 is read as 1 for z != 0 and 0 for z = 0), and all alpha, beta in
    GF(2^k), including 0, for k = 3, 4, 5. "count" is the number of parameter tuples; every
    tuple gives a permutation, and equal tables are not merged.
(b) Cyclotomic maps x^r h(x^((2^n-1)/3)) at n = 8: all 24 exponent classes r, h exhaustive.
(c) Switched inverse x^254 + gamma Tr(delta x^d) at n = 8: all d (cyclotomic classes), delta, gamma.
Only permutations are scored; the score is the cost (3 x number of affine 2-flats).
"""
import itertools
import math

import numpy as np

from common import parser, save
from apnsearch import fast
from apnsearch.gf import GF


def butterfly_table(R):
    K = R.shape[0]
    z = np.arange(K)
    Rinv = np.empty_like(R)
    Rinv[z[:, None], R] = z[None, :]
    x, y = np.repeat(z, K), np.tile(z, K)
    t = Rinv[y, x]
    return (R[t, y] * K + t).astype(np.int32)


def classes_mod(q1, k, perm_only):
    seen, reps = set(), []
    for e in range(1, q1 + 1):
        if perm_only and math.gcd(e, q1) != 1:
            continue
        c = frozenset((e << i) % q1 or q1 for i in range(k))
        if c not in seen:
            seen.add(c)
            reps.append(e)
    return reps


def butterflies(k):
    g = GF(k)
    K = g.N
    z = np.arange(K)
    costs = []
    for e in classes_mod(K - 1, k, True):
        pe = g.power_table(e)
        for f in range(K - 1):
            pf = g.power_table(f) if f else (z != 0).astype(np.int64)
            for alpha in range(K):
                sh = pe[z[None, :] ^ g.mul_vec(np.full(K, alpha), z)[:, None]]
                for beta in range(K):
                    H = butterfly_table(sh ^ g.mul_vec(np.full(K, beta), pf)[:, None])
                    costs.append(fast.ddt_cost(H))
    c = np.array(costs)
    return {"k": k, "n": 2 * k, "count": len(c), "apn": int((c == 0).sum()),
            "min_cost": int(c.min()), "min_r": float(c.min() / ((1 << 2 * k) - 1))}


def cyclotomic(n=8, l=3):
    g = GF(n)
    N, q1 = g.N, g.N - 1
    s = q1 // l
    X = np.arange(1, N)
    idx = (g.log[X] * s % q1) // s
    best, perms = {}, 0
    for r in classes_mod(q1, n, False):
        if math.gcd(r, s) != 1:
            continue
        base = (r * g.log[X]) % q1
        mins = None
        for hl in itertools.product(range(q1), repeat=l - 1):
            S = np.zeros(N, dtype=np.int32)
            S[1:] = g.exp[(base + np.array((0,) + hl)[idx]) % q1]
            if np.bincount(S, minlength=N).max() != 1:
                continue
            perms += 1
            c = fast.ddt_cost(S)
            mins = c if mins is None else min(mins, c)
        best[int(r)] = mins
    return {"n": n, "l": l, "exponent_classes": len(best), "permutations": perms,
            "best_cost_per_r": best,
            "min_cost": min(best.values())}


def switched_inverse(n=8):
    g = GF(n)
    N, q1 = g.N, g.N - 1
    inv = g.power_table(q1 - 1)
    hist = {}
    for d in classes_mod(q1, n, False):
        xd = g.power_table(d)
        for delta in range(1, N):
            f = g.trace_vec(g.mul_vec(np.full(N, delta), xd)).astype(bool)
            if not f.any():
                continue
            for gamma in range(1, N):
                S = inv ^ np.where(f, gamma, 0)
                if np.bincount(S, minlength=N).max() != 1:
                    continue
                c = fast.ddt_cost(S.astype(np.int32))
                hist[c] = hist.get(c, 0) + 1
    costs = sorted(hist)
    return {"n": n, "permutations": sum(hist.values()), "lowest_costs": costs[:4],
            "count_at_255": hist.get(N - 1, 0),
            "min_cost_other_than_inverse_value": next(c for c in costs if c != N - 1)}


def main():
    a = parser(__doc__).parse_args()
    out = {"butterflies": [butterflies(k) for k in ((3, 4) if a.quick else (3, 4, 5))]}
    for r in out["butterflies"]:
        print(r)
    out["cyclotomic"] = cyclotomic()
    print({k: v for k, v in out["cyclotomic"].items() if k != "best_cost_per_r"})
    out["switched_inverse"] = switched_inverse()
    print(out["switched_inverse"])
    save("e02_structured_constructions", out)


if __name__ == "__main__":
    main()
