"""
Known APN functions used in the experiments, as lookup tables over GF(2^n) with the
primitive moduli of gf.PRIMITIVE (alpha = 2 is primitive).

Quadratic families with members at n = 10 = 2m (q = 2^m), following Kaleyski
[Cryptogr. Commun. 2022, Table 3]:
  F3  (Budaghyan-Carlet)          s x^(q+1) + x^(2^i+1) + x^(q(2^i+1)) + c x^(2^i q+1) + c^q x^(2^i+q)
  F4  (Budaghyan-Carlet-Leander)  x^3 + a^-1 Tr(a^3 x^9)
  F12 (Taniguchi)                 u(u^q x + x^q u)(x^q + x) + (u^q x + x^q u)^(2^(2i) + 2^(3i))
                                  + a (u^q x + x^q u)^(2^(2i)) (x^q + x)^(2^i) + b (x^q + x)^(2^i+1)
  F13 (Budaghyan-Helleseth-Kaleyski)
                                  x^3 + a (x^(2^i+1))^(2^k) + b x^(3 * 2^m) + c (x^(2^(i+m)+2^m))^(2^k)
"""

import math
import os
import re
import urllib.request

import numpy as np

from .gf import GF
from . import fast

POWER_APN = {  # exponent -> family, for the dimensions used in the paper
    6: {3: "Gold"},
    8: {3: "Gold", 9: "Gold", 57: "Kasami"},
    9: {3: "Gold", 5: "Gold", 17: "Gold", 13: "Kasami", 241: "Kasami", 19: "Welch/Niho",
        510: "inverse"},
    10: {3: "Gold", 9: "Gold", 57: "Kasami", 339: "Dobbertin"},
}

BL_URL = "https://zenodo.org/records/4738942/files/apn_10bit.txt?download=1"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def power_maps(n):
    g = GF(n)
    return [(f"x^{d} ({fam})", g.power_table(d)) for d, fam in POWER_APN.get(n, {}).items()]


def x3_tr_x9(n, a=1):
    """F4: x^3 + a^-1 Tr(a^3 x^9)."""
    g = GF(n)
    X = np.arange(g.N)
    t = g.trace_vec(g.mul_vec(np.full(g.N, g.pow(a, 3)), g.pow_vec(X, 9)))
    return g.pow_vec(X, 3) ^ g.mul_vec(np.full(g.N, g.inv(a)), t)


def kim6():
    """An APN instance of the Kim function x^3 + x^10 + u x^24 over GF(2^6) (first such u)."""
    g = GF(6)
    for u in range(1, 64):
        F = g.power_table(3) ^ g.power_table(10) ^ g.mul_vec(np.full(64, u), g.power_table(24))
        if fast.is_apn(F):
            return u, F
    raise RuntimeError("no APN Kim instance")


def edel_pott_source6():
    """x^3 + alpha (x^17 + x^18 + x^20 + x^24) over GF(2^6), the quadratic function whose
    switching neighbourhood contains the non-quadratic APN function of Edel and Pott."""
    g = GF(6)
    T = g.power_table(17) ^ g.power_table(18) ^ g.power_table(20) ^ g.power_table(24)
    return g.power_table(3) ^ g.mul_vec(np.full(64, 2), T)


# --------------------------------------------------------------------------- n = 10 families

def _ops(n):
    g = GF(n)
    X = np.arange(g.N)

    def P(e):
        return g.pow_vec(X, e)

    def mul(a, T):
        return g.mul_vec(np.full(g.N, int(a)), T) if np.isscalar(a) else g.mul_vec(a, T)

    return g, X, P, mul


def f3_instance(rng, n=10):
    g, X, P, mul = _ops(n)
    q = 1 << (n // 2)
    outside = [x for x in range(g.N) if g.pow(x, q) != x]
    i = rng.choice([i for i in range(1, n // 2) if math.gcd(i, n // 2) == 1])
    c, s = rng.randrange(g.N), rng.choice(outside)
    F = (mul(s, P(q + 1)) ^ P((1 << i) + 1) ^ P(q * ((1 << i) + 1)) ^ mul(c, P((1 << i) * q + 1))
         ^ mul(g.pow(c, q), P((1 << i) + q)))
    return F, f"F3 i={i} c={c} s={s}"


def f12_instance(rng, n=10):
    g, X, P, mul = _ops(n)
    m = n // 2
    q = 1 << m
    sub = [x for x in range(g.N) if g.pow(x, q) == x]
    prim = [int(g.exp[k]) for k in range(1, g.N - 1) if math.gcd(k, g.N - 1) == 1]
    i = rng.choice([i for i in range(1, m) if math.gcd(i, m) == 1])
    u, a, b = rng.choice(prim), rng.choice(sub), rng.choice(sub)
    xq = P(q)
    y = mul(g.pow(u, q), X) ^ mul(u, xq)
    z = xq ^ X
    F = (mul(u, g.mul_vec(y, z)) ^ g.pow_vec(y, (1 << (2 * i)) + (1 << (3 * i)))
         ^ mul(a, g.mul_vec(g.pow_vec(y, 1 << (2 * i)), g.pow_vec(z, 1 << i)))
         ^ mul(b, g.pow_vec(z, (1 << i) + 1)))
    return F, f"F12 i={i} u={u} a={a} b={b}"


def f13_all(n=10):
    """All F13-type functions with a, b, c in GF(4) and all i, k (n = 2m)."""
    g, X, P, mul = _ops(n)
    m = n // 2
    gf4 = [x for x in range(g.N) if g.pow(x, 4) == x]
    out = []
    for i in range(1, n):
        for k in range(n):
            e2 = ((1 << (i + m)) + (1 << m)) % (g.N - 1) or (g.N - 1)
            A = g.pow_vec(P((1 << i) + 1), 1 << k)
            C = g.pow_vec(P(e2), 1 << k)
            for a in gf4:
                for b in gf4:
                    for c in gf4:
                        F = P(3) ^ mul(a, A) ^ mul(b, P(3 << m)) ^ mul(c, C)
                        out.append((F, f"F13 i={i} k={k} a={a} b={b} c={c}"))
    return out


def beierle_leander_10(path=None):
    """The five quadratic APN functions in dimension 10 of Beierle and Leander (Zenodo
    10.5281/zenodo.4738942). Downloaded on first use into data/."""
    path = path or os.path.join(DATA, "apn_10bit.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(BL_URL, path)
    txt = open(path).read()
    luts = [[int(v) for v in re.findall(r"\d+", b)] for b in re.findall(r"\{([^}]*)\}", txt)]
    return [(f"Beierle-Leander #{i + 1}", np.array(l)) for i, l in enumerate(luts) if len(l) == 1024]


# ------------------------------------------------- n = 10 instances from more recent papers

def zheng_instances_10(alpha):
    """The three explicit APN instances over GF(2^10) of Zheng, Kan, Li, Peng and Tang (arXiv
    2101.11535v1, Table III, rows F15 and the two sporadic ones), for a primitive element alpha:
    a Tr(b x^3) + a^q Tr(...), q = 2^5, Tr(y) = y + y^q (relative trace to GF(2^5))."""
    g, X, P, mul = _ops(10)
    q = 32

    def tr(T):
        return T ^ g.pow_vec(T, q)

    aq = g.pow(alpha, q)
    return [
        ("Zheng et al. F15", mul(alpha, tr(mul(alpha, P(3)))) ^ mul(aq, tr(mul(g.pow(alpha, 3), P(9))))),
        ("Zheng et al. Remark 3.6", mul(alpha, tr(P(3))) ^ mul(aq, tr(mul(g.pow(alpha, 11), P(9))))),
        ("Zheng et al. Example 1", mul(alpha, tr(P(3)))
         ^ mul(aq, tr(mul(alpha, P(5)) ^ mul(g.pow(alpha, 369), P(4 * q + 1))))),
    ]


# Table 2 of Budaghyan, Ivkovic and Kaleyski, "Triplicate functions" (Cryptogr. Commun. 15, 2023):
# terms (k, e) stand for beta^k x^e, beta primitive in GF(4); the last entry is the differential
# spectrum of the ortho-derivative printed there (multiplicities of the values 0, 2, 4, ...).
BIK_TABLE2 = {
    1: ([(0, 3), (0, 33), (2, 36), (1, 66), (1, 96), (0, 129)],
        [634041, 320166, 78420, 13020, 1830, 60, 15]),
    2: ([(0, 3), (1, 9), (1, 36), (0, 96), (2, 129), (0, 768)],
        [636306, 315018, 82335, 11715, 2145, 33]),
    3: ([(0, 3), (0, 6), (2, 24), (1, 48), (1, 72), (0, 132), (0, 288)],
        [631911, 323421, 78495, 11775, 1725, 210, 15]),
    4: ([(0, 3), (0, 9), (2, 12), (0, 36), (0, 48), (1, 264), (1, 516)],
        [632286, 322566, 78540, 12675, 1320, 165]),
    5: ([(0, 3), (0, 18), (2, 24), (2, 36), (0, 48), (0, 72), (2, 288)],
        [634746, 318081, 79920, 13485, 1215, 90, 15]),
    6: ([(0, 3), (0, 9), (1, 24), (0, 33), (0, 66), (1, 72), (1, 258)],
        [636591, 316371, 78720, 13740, 1935, 165, 30]),
}


def bik_instance_10(i, beta):
    g, X, P, mul = _ops(10)
    F = np.zeros(g.N, dtype=np.int64)
    for k, e in BIK_TABLE2[i][0]:
        F ^= mul(g.pow(beta, k), P(e))
    return F


def yu_f2_x3_tr_x9_10():
    """x^513 + x^192 + x^96 + x^48 + x^33 + x^18 + x^12, the representative of the class of
    x^3 + Tr(x^9) given by Yu, Li, Ichanska and Kaleyski (ePrint 2024/1778, Table 1)."""
    g, X, P, mul = _ops(10)
    F = np.zeros(g.N, dtype=np.int64)
    for e in (513, 192, 96, 48, 33, 18, 12):
        F ^= P(e)
    return F


# --------------------------------------------- bivariate families on GF(2^5) x GF(2^5), n = 10
# A pair (f(x, y), g(x, y)) is identified with the (10,10)-function (x + 32 y) -> f + 32 g; this
# is an F_2-linear identification of GF(2^5)^2 with F_2^10, so EA-classes are preserved.

def _biv_ops():
    g5 = GF(5)
    X = np.arange(1024, dtype=np.int64)
    xs, ys = X & 31, X >> 5

    def p(V, e):
        e %= 31
        return g5.pow_vec(V, e or 31)            # V^e with e reduced mod 31 (e > 0)

    def mul(*vs):
        r = vs[0]
        for v in vs[1:]:
            r = g5.mul_vec(r, v)
        return r

    def c(a, V):
        return g5.mul_vec(np.full(len(V), int(a), dtype=np.int64), V)

    def proj(t, Q):
        """(a, b, c, d)_Q = a x^(Q+1) + b x^Q y + c x y^Q + d y^(Q+1)."""
        a, b, cc, d = t
        return (c(a, p(xs, Q + 1)) ^ c(b, mul(p(xs, Q), ys)) ^ c(cc, mul(xs, p(ys, Q)))
                ^ c(d, p(ys, Q + 1)))

    def pair(f, h):
        return (f | (h << 5)).astype(np.int64)

    def no_root(Q, b, cc, d):
        """x^(Q+1) + b x^Q + c x + d has no root in GF(2^5)."""
        V = np.arange(32, dtype=np.int64)
        return not np.any((p(V, Q + 1) ^ c(b, p(V, Q)) ^ c(cc, V) ^ d) == 0)

    return g5, xs, ys, p, mul, c, proj, pair, no_root


def gologlu_10():
    """Goloğlu's biprojective families F1 = ((1,0,1,1)_q, (1,1,0,1)_{q^2}) and
    F2 = ((1,0,1,1)_q, (0,1,1,0)_{q^3}), q = 2^k, gcd(3k, 5) = 1 (Goloğlu-Kölsch, Table 2.1)."""
    g5, xs, ys, p, mul, c, proj, pair, no_root = _biv_ops()
    out = []
    for k in range(1, 5):
        q = 1 << k
        out.append((f"Gologlu F1 k={k}", pair(proj((1, 0, 1, 1), q), proj((1, 1, 0, 1), q * q))))
        out.append((f"Gologlu F2 k={k}", pair(proj((1, 0, 1, 1), q), proj((0, 1, 1, 0), q ** 3))))
    return out


def clv_10():
    """Calderini-Li-Villa (arXiv 2204.07462v2), Theorem 3 (family F1, all k and alpha) and
    Theorem 4 (family F2, all alpha), for m = 5; alpha such that x^(q+1) + x + alpha
    (resp. x^3 + x + alpha) has no root in GF(2^5)."""
    g5, xs, ys, p, mul, c, proj, pair, no_root = _biv_ops()
    out = []
    for k in range(1, 5):
        q = 1 << k
        for al in range(32):
            if not no_root(q, 0, 1, al):
                continue
            f = p(xs, q + 1) ^ mul(xs, p(ys, q)) ^ c(al, p(ys, q + 1))
            h = (p(xs, q * q + 1) ^ c(al, mul(p(xs, q * q), ys)) ^ c(g5.pow(1 ^ al, q), mul(xs, p(ys, q * q)))
                 ^ c(al, p(ys, q * q + 1)))
            out.append((f"CLV family 1 k={k} alpha={al}", pair(f, h)))
    for al in range(32):
        if not no_root(2, 0, 1, al):
            continue
        f = p(xs, 3) ^ mul(xs, ys) ^ mul(xs, p(ys, 2)) ^ c(al, p(ys, 3))
        h = (p(xs, 5) ^ mul(xs, ys) ^ c(al, mul(p(xs, 2), p(ys, 2))) ^ c(al, mul(p(xs, 4), ys))
             ^ c(g5.pow(1 ^ al, 2), mul(xs, p(ys, 4))) ^ c(al, p(ys, 5)))
        out.append((f"CLV family 2 alpha={al}", pair(f, h)))
    return out


def lzlq_10():
    """Li-Zhou-Li-Qu (arXiv 2105.08464v1, Theorem 5):
    (x^3 + x y^2 + y^3 + x y, x^5 + x^4 y + y^5 + x y + x^2 y^2), m = 5."""
    g5, xs, ys, p, mul, c, proj, pair, no_root = _biv_ops()
    f = p(xs, 3) ^ mul(xs, p(ys, 2)) ^ p(ys, 3) ^ mul(xs, ys)
    h = p(xs, 5) ^ mul(p(xs, 4), ys) ^ p(ys, 5) ^ mul(xs, ys) ^ mul(p(xs, 2), p(ys, 2))
    return [("Li-Zhou-Li-Qu Thm 5", pair(f, h))]


def carlet_10(rng, count):
    """Carlet's family (xy, (1,b,c,d)_q), q = 2^k, x^(q+1) + b x^q + c x + d without roots in
    GF(2^5) (Goloğlu-Kölsch, Table 2.1): `count` random parameter choices."""
    g5, xs, ys, p, mul, c, proj, pair, no_root = _biv_ops()
    out = []
    while len(out) < count:
        k = rng.randrange(1, 5)
        b, cc, d = rng.randrange(32), rng.randrange(32), rng.randrange(32)
        if no_root(1 << k, b, cc, d):
            out.append((f"Carlet k={k} b={b} c={cc} d={d}", pair(mul(xs, ys), proj((1, b, cc, d), 1 << k))))
    return out


def zheng_family_10(rng, count):
    """Zheng et al., Theorem 3.3 shape a Tr(b x^3) + a^q Tr(c x^(2^s+1)), a not in GF(2^5), s odd:
    `count` random parameter choices (APN or not)."""
    g, X, P, mul = _ops(10)
    out = []
    while len(out) < count:
        s = rng.choice([1, 3, 5, 7, 9])
        a, b, cc = rng.randrange(1, 1024), rng.randrange(1, 1024), rng.randrange(1, 1024)
        if g.pow(a, 32) == a:
            continue
        tr = lambda T: T ^ g.pow_vec(T, 32)
        F = mul(a, tr(mul(b, P(3)))) ^ mul(g.pow(a, 32), tr(mul(cc, P((1 << s) + 1))))
        out.append((f"Zheng et al. Thm 3.3 s={s} a={a} b={b} c={cc}", F))
    return out
