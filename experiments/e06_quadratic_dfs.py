"""
E6: randomized depth-first search for quadratic APN functions over the alternating
bilinear form (apnsearch.fast.quadratic_apn_search): class coverage in dimensions 6 and 7,
and the failure to find any solution in dimension 8 within the node budget.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from common import parser, save
from apnsearch import fast
from apnsearch.invariants import ortho_invariant


def coverage(n, restarts, node_limit):
    t = time.time()
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        sols = list(ex.map(lambda s: fast.quadratic_apn_search(n, s, node_limit=node_limit),
                           range(restarts)))
    found = [F for r in sols for F in r]
    assert all(fast.is_apn(F) for F in found)
    classes = {ortho_invariant(F, n) for F in found}
    r = {"n": n, "restarts": restarts, "node_limit": node_limit, "functions": len(found),
         "classes": len(classes), "seconds": round(time.time() - t, 1)}
    print(r, flush=True)
    return r


def main():
    a = parser(__doc__).parse_args()
    rows = [coverage(6, 3000, 10**6),
            coverage(7, 100 if a.quick else 400, 10**6),
            coverage(8, 18 if a.quick else 36, 3 * 10**6 if a.quick else 3 * 10**7)]
    save("e06_quadratic_dfs", {"rows": rows})


if __name__ == "__main__":
    main()
