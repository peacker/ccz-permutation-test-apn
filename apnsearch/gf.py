"""
Helper library available to candidate programs (import apn_lib).

Everything is pure Python + numpy so it can run inside the sandbox.
"""

import numpy as np

# Primitive polynomials over GF(2), bit i = coefficient of x^i.
PRIMITIVE = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10000011,
    8: 0b100011101,
    9: 0b1000010001,
    10: 0b10000001001,
    11: 0b100000000101,
    12: 0b1000001010011,
    13: 0x201B,
    14: 0x4443,
    15: 0x8003,
    16: 0x1100B,
    17: 0x20009,
    18: 0x40081,
    19: 0x80027,
    20: 0x100009,
    22: 0x400003,
}


class GF:
    """GF(2^n) with elements encoded as integers 0 .. 2^n - 1 (polynomial basis).

    g = GF(n)
    g.mul(a, b), g.pow(a, e), g.inv(a), g.trace(a)       scalars
    g.mul_vec(A, B), g.pow_vec(A, e), g.trace_vec(A)     numpy arrays
    g.power_table(e)                                    array x -> x^e for all x
    g.exp[i] = alpha^i, g.log[x] (alpha = 2 is primitive), g.N = 2^n
    """

    def __init__(self, n):
        self.n, self.N = n, 1 << n
        mod = PRIMITIVE[n]
        exp = np.zeros(2 * self.N, dtype=np.int64)
        log = np.zeros(self.N, dtype=np.int64)
        x = 1
        for i in range(self.N - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x >> n & 1:
                x ^= mod
        assert x == 1, "modulus is not primitive"
        exp[self.N - 1:2 * self.N - 1] = exp[:self.N]
        self.exp, self.log = exp, log
        self._trace = self._build_trace()

    def _build_trace(self):
        t = np.zeros(self.N, dtype=np.int64)
        for x in range(self.N):
            acc, y = 0, x
            for _ in range(self.n):
                acc ^= y
                y = self.mul(y, y)
            t[x] = acc
        return t

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self.exp[(self.log[a] + self.log[b]) % (self.N - 1)])

    def pow(self, a, e):
        if e == 0:
            return 1
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] * e) % (self.N - 1)])

    def inv(self, a):
        """Multiplicative inverse, with inv(0) = 0."""
        return 0 if a == 0 else int(self.exp[(-self.log[a]) % (self.N - 1)])

    def trace(self, a):
        return int(self._trace[a])

    def mul_vec(self, A, B):
        A, B = np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64)
        out = self.exp[(self.log[A] + self.log[B]) % (self.N - 1)]
        return np.where((A == 0) | (B == 0), 0, out)

    def pow_vec(self, A, e):
        A = np.asarray(A, dtype=np.int64)
        if e == 0:
            return np.ones_like(A)
        out = self.exp[(self.log[A] * e) % (self.N - 1)]
        return np.where(A == 0, 0, out)

    def trace_vec(self, A):
        return self._trace[np.asarray(A, dtype=np.int64)]

    def power_table(self, e):
        return self.pow_vec(np.arange(self.N), e)


def is_permutation(S):
    S = list(S)
    return sorted(S) == list(range(len(S)))


def inverse_permutation(S):
    S = np.asarray(S, dtype=np.int64)
    inv = np.empty_like(S)
    inv[S] = np.arange(len(S))
    return inv


def ddt(S):
    """Full DDT as a numpy array D[a, b] (fine for n <= 9)."""
    S = np.asarray(S, dtype=np.int64)
    N = len(S)
    X = np.arange(N)
    b = S[X[None, :]] ^ S[X[None, :] ^ X[:, None]]
    return np.stack([np.bincount(row, minlength=N) for row in b])


def ddt_cost(S):
    """Sum of D(D-2)/8 over the DDT rows a != 0: 0 iff APN (same as the evaluator)."""
    D = ddt(S)[1:]
    return int((D * (D - 2)).sum()) // 8


def parity(x):
    return bin(int(x)).count("1") & 1
