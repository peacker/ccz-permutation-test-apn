"""
E1: the inverse function x^(2^n - 2) for even n (paper, Section 3).

(a) Its cost is 2^n - 1 and its bad 2-flats (2-flats on which it is affine) are exactly
    the GF(4)-lines a*GF(4), a != 0, through 0 (n = 4, 6, 8, 10).
(b) n = 8: effect of all transpositions S -> S o (i j), and of a second transposition
    after each cost-neutral one.
(c) n = 8: simulated annealing started at x^254 (plain and involution-preserving moves).
(d) n = 8: simulated annealing started at random permutations, for comparison.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from common import parser, save
from apnsearch import fast
from apnsearch.gf import GF


def bad_flats_are_gf4_lines(n):
    g = GF(n)
    N = g.N
    S = g.power_table(N - 2)
    w = int(g.exp[(N - 1) // 3])                       # primitive cube root of unity
    X = np.arange(N)
    ok, flats = True, set()
    for a in range(1, N):
        d = S ^ S[X ^ a]
        cnt = np.bincount(d, minlength=N)
        if cnt.max() > 4 or (cnt == 4).sum() != 1:
            ok = False
        b = int(np.argmax(cnt))
        sol = frozenset(np.nonzero(d == b)[0].tolist())
        line = frozenset({0, a, g.mul(a, w), g.mul(a, g.mul(w, w))})
        ok &= sol == line
        flats.add(sol)
    return {"n": n, "cost": fast.ddt_cost(S), "expected_cost": N - 1,
            "bad_flats": len(flats), "gf4_lines": (N - 1) // 3, "bad_flats_are_gf4_lines": bool(ok)}


def main():
    a = parser(__doc__).parse_args()
    part_a = [bad_flats_are_gf4_lines(n) for n in (4, 6, 8, 10)]
    for r in part_a:
        print(r)

    g = GF(8)
    inv = g.power_table(254).astype(np.int32)
    D = fast.swap_deltas(inv)[np.triu_indices(256, 1)]
    iu = np.triu_indices(256, 1)
    pairs = [(int(i), int(j)) for i, j, d in zip(iu[0], iu[1], D) if d == 0]
    second_min, second_neutral = None, 0
    for i, j in pairs:
        Q = inv.copy()
        Q[i], Q[j] = Q[j], Q[i]
        D2 = fast.swap_deltas(Q)[iu]
        second_min = int(D2.min()) if second_min is None else min(second_min, int(D2.min()))
        second_neutral += int((D2 == 0).sum())
    part_b = {"transpositions": int(len(D)), "improving": int((D < 0).sum()),
              "neutral": len(pairs), "neutral_all_involve_0": all(0 in p for p in pairs),
              "min_positive_delta": int(D[D > 0].min()),
              "second_step_min_delta": second_min, "second_step_neutral": second_neutral}
    print(part_b)

    iters = 2_000_000 if a.quick else 20_000_000
    runs = 1 if a.quick else 3
    schedules = [(2.0, 0.1), (20.0, 0.5), (60.0, 1.0)]
    jobs = [(inv_moves, t0, t1, s) for inv_moves in (False, True) for t0, t1 in schedules
            for s in range(runs)]

    def run(job):
        im, t0, t1, s = job
        c, _ = fast.anneal(inv, iters, t0, t1, s + 1, involution=im)
        return {"involution_moves": im, "t0": t0, "t1": t1, "seed": s, "best_cost": c}

    with ThreadPoolExecutor(os.cpu_count()) as ex:
        part_c = list(ex.map(run, jobs))
    print("annealing best costs:", sorted({r["best_cost"] for r in part_c}))

    def run_random(seed):
        P = np.random.default_rng(seed).permutation(256).astype(np.int32)
        c, _ = fast.anneal(P, iters, 8.0, 0.2, 1000 + seed)
        return {"seed": seed, "start_cost": fast.ddt_cost(P), "best_cost": c}

    with ThreadPoolExecutor(os.cpu_count()) as ex:
        part_d = list(ex.map(run_random, range(6 if a.quick else 18)))
    print("random starts, best costs:", sorted(r["best_cost"] for r in part_d))
    save("e01_inverse_function", {"flats": part_a, "transpositions_n8": part_b,
                                  "annealing_n8": {"iters": iters, "runs": part_c},
                                  "annealing_random_n8": {"iters": iters, "runs": part_d}})


if __name__ == "__main__":
    main()
