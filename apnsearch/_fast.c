/*
 * Performance-critical routines of apnsearch, loaded through ctypes by fast.py
 * (compiled automatically on first import).
 *
 *   ddt_cost, swap_deltas, anneal_swaps   DDT cost and local search on permutations
 *   find_subspaces(_pair), find_complementary_pair
 *                                         exhaustive search for vector spaces in a set
 *   switching_kernel                      exact Edel-Pott switching (linear algebra)
 *   quadratic_apn_search / _extend        randomized DFS for quadratic APN functions
 *   bilinear_apn_filter                   APN test for quadratic functions from B(e_k, e_l)
 *
 * Cost of S on n bits: sum over a != 0 and all b of D(D - 2) / 8 with D = DDT[a][b];
 * it is 0 exactly when S is APN, and equals 3 times the number of 2-flats on which
 * S is affine.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static __thread uint64_t rng_state = 88172645463325252ULL;

static inline uint64_t rnd(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

static inline double rndu(void)
{
    return (rnd() >> 11) * (1.0 / 9007199254740992.0);
}

static inline long long entry_cost(int d)
{
    long long h = d / 2;
    return h * (h - 1) / 2;
}

static void build_ddt(int N, const int *S, int *D)
{
    memset(D, 0, sizeof(int) * N * N);
    for (int a = 0; a < N; a++)
        for (int x = 0; x < N; x++)
            D[a * N + (S[x] ^ S[x ^ a])]++;
}

static long long total_cost(int N, const int *D)
{
    long long c = 0;
    for (int k = N; k < N * N; k++)
        c += entry_cost(D[k]);
    return c;
}

long long ddt_cost(int n, const int *S)
{
    int N = 1 << n;
    int *D = malloc(sizeof(int) * N * N);
    build_ddt(N, S, D);
    long long c = total_cost(N, D);
    free(D);
    return c;
}

/* Swap S[i] and S[j], update D incrementally and return the cost change.
 * Calling it again with the same (i, j) undoes it exactly. */
static long long apply_swap(int N, int *S, int *D, int i, int j)
{
    long long delta = 0;
    int si = S[i], sj = S[j], aij = i ^ j;
    for (int a = 1; a < N; a++) {
        if (a == aij)
            continue;
        int *row = D + a * N;
        int ti = S[i ^ a], tj = S[j ^ a], c;
        c = si ^ ti; delta += entry_cost(row[c] - 2) - entry_cost(row[c]); row[c] -= 2;
        c = sj ^ tj; delta += entry_cost(row[c] - 2) - entry_cost(row[c]); row[c] -= 2;
        c = sj ^ ti; delta += entry_cost(row[c] + 2) - entry_cost(row[c]); row[c] += 2;
        c = si ^ tj; delta += entry_cost(row[c] + 2) - entry_cost(row[c]); row[c] += 2;
    }
    S[i] = sj;
    S[j] = si;
    return delta;
}

/* Cost change of every transposition (i, j), i < j, written to out[i * N + j]. */
void swap_deltas(int n, const int *S0, long long *out)
{
    int N = 1 << n;
    int *S = malloc(sizeof(int) * N);
    int *D = malloc(sizeof(int) * N * N);
    memcpy(S, S0, sizeof(int) * N);
    build_ddt(N, S, D);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            out[i * N + j] = apply_swap(N, S, D, i, j);
            apply_swap(N, S, D, i, j);
        }
    free(S);
    free(D);
}

/* Build an involution-preserving move on S: pick x, y and re-pair the cycles
 * {x, S[x]} and {y, S[y]} as {x, y} and {S[x], S[y]} (fixed points allowed);
 * if y == S[x] != x, split that 2-cycle into two fixed points. The move is
 * written as a list of transpositions of positions (S <- S o (p q)). */
static int involution_move(int N, const int *S, int moves[][2])
{
    int x = rnd() % N, y = rnd() % N;
    if (x == y)
        return 0;
    int P[4], T[4], np = 0;
    int xs = S[x], ys = S[y];
    if (y == xs) {
        P[np] = x; T[np++] = x;
        P[np] = y; T[np++] = y;
    } else {
        P[np] = x; T[np++] = y;
        P[np] = y; T[np++] = x;
        if (xs != x && ys != y) {
            P[np] = xs; T[np++] = ys;
            P[np] = ys; T[np++] = xs;
        } else if (xs != x) {
            P[np] = xs; T[np++] = xs;
        } else if (ys != y) {
            P[np] = ys; T[np++] = ys;
        }
    }
    /* selection sort on a local copy to express S' as position swaps */
    int cur[4], nm = 0;
    for (int k = 0; k < np; k++)
        cur[k] = S[P[k]];
    for (int k = 0; k < np; k++) {
        if (cur[k] == T[k])
            continue;
        int q = k + 1;
        while (cur[q] != T[k])
            q++;
        int tmp = cur[k]; cur[k] = cur[q]; cur[q] = tmp;
        moves[nm][0] = P[k];
        moves[nm][1] = P[q];
        nm++;
    }
    return nm;
}

/* Simulated annealing starting from S (modified in place). Moves are single
 * transpositions, or involution-preserving re-pairings if involution != 0
 * (S must then start as an involution). The best permutation seen is written
 * to best and its cost returned. */
long long anneal_swaps(int n, int *S, long long iters, double t0, double t1,
                       uint64_t seed, int involution, int *best)
{
    int N = 1 << n;
    int *D = malloc(sizeof(int) * N * N);
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 1;
    build_ddt(N, S, D);
    long long cur = total_cost(N, D), bestc = cur;
    memcpy(best, S, sizeof(int) * N);
    double T = t0, r = pow(t1 / t0, 1.0 / (double)iters);
    int moves[4][2];

    for (long long it = 0; it < iters && bestc > 0; it++, T *= r) {
        int nm;
        if (involution) {
            nm = involution_move(N, S, moves);
        } else {
            int i = rnd() % N, j = rnd() % (N - 1);
            if (j >= i)
                j++;
            moves[0][0] = i;
            moves[0][1] = j;
            nm = 1;
        }
        if (nm == 0)
            continue;
        long long d = 0;
        for (int k = 0; k < nm; k++)
            d += apply_swap(N, S, D, moves[k][0], moves[k][1]);
        if (d <= 0 || rndu() < exp(-(double)d / T)) {
            cur += d;
            if (cur < bestc) {
                bestc = cur;
                memcpy(best, S, sizeof(int) * N);
            }
        } else {
            for (int k = nm - 1; k >= 0; k--)
                apply_swap(N, S, D, moves[k][0], moves[k][1]);
        }
    }
    free(D);
    return bestc;
}

/* ---------------------------------------------------------------------------
 * Enumerate all d-dimensional linear subspaces of F_2^m contained in
 * Z u {0}, where Z is given as a byte mask over [0, 2^m). Each subspace is
 * found once, through its reduced echelon basis (pivot = highest set bit,
 * pivots strictly decreasing). Bases are written to out (d ints each).
 * Returns the number of subspaces found, or -1 if cap or node_limit is hit.
 */
typedef struct {
    int d, m;
    int *elems[33];
    int count[33];
    uint8_t *mask[33];
    int basis[33];
    int *out;
    long long cap, found, nodes, node_limit;
    int aborted;
    int stop_on_pair, pair_found;
    long long pair[2];
    int part, nparts;          /* level-0 branches k with k % nparts == part only */
} SubCtx;

static int gf2_rank(int *v, int k);

static inline int highest_bit(int x)
{
    return 31 - __builtin_clz((unsigned)x);
}

static void sub_rec(SubCtx *c, int level, int pivots)
{
    if (c->aborted)
        return;
    if (level == c->d) {
        if (c->found >= c->cap) {
            c->aborted = 1;
            return;
        }
        memcpy(c->out + c->found * c->d, c->basis, sizeof(int) * c->d);
        if (c->stop_on_pair) {
            int v[64];
            for (long long b = 0; b < c->found; b++) {
                memcpy(v, c->out + b * c->d, sizeof(int) * c->d);
                memcpy(v + c->d, c->basis, sizeof(int) * c->d);
                if (gf2_rank(v, 2 * c->d) == 2 * c->d) {
                    c->pair[0] = b;
                    c->pair[1] = c->found;
                    c->pair_found = 1;
                    c->aborted = 1;
                    break;
                }
            }
        }
        c->found++;
        return;
    }
    if (++c->nodes > c->node_limit) {
        c->aborted = 1;
        return;
    }
    int *E = c->elems[level], m = c->count[level];
    uint8_t *mask = c->mask[level];
    int need = (1 << (c->d - level - 1)) - 1;
    for (int k = 0; k < m && !c->aborted; k++) {
        if (level == 0 && c->nparts > 1 && k % c->nparts != c->part)
            continue;
        int z = E[k], h = highest_bit(z);
        if (pivots >> h & 1)
            continue;
        if ((1 << h) - 1 < need)
            continue;
        int *E2 = c->elems[level + 1], m2 = 0;
        uint8_t *mask2 = c->mask[level + 1];
        int lim = 1 << h;
        for (int t = 0; t < m && E[t] < lim; t++)
            if (mask[E[t] ^ z])
                E2[m2++] = E[t];
        if (m2 < need)
            continue;
        for (int t = 0; t < m2; t++)
            mask2[E2[t]] = 1;
        c->count[level + 1] = m2;
        c->basis[level] = z;
        sub_rec(c, level + 1, pivots | z);
        for (int t = 0; t < m2; t++)
            mask2[E2[t]] = 0;
    }
}

static long long find_subspaces_impl(int m, const uint8_t *Z, int d, int *out, long long cap,
                                     long long node_limit, long long *nodes_used,
                                     int stop_on_pair, long long *pair, int part, int nparts)
{
    SubCtx c;
    memset(&c, 0, sizeof(c));
    c.stop_on_pair = stop_on_pair;
    c.part = part;
    c.nparts = nparts;
    c.d = d;
    c.m = m;
    c.out = out;
    c.cap = cap;
    c.node_limit = node_limit;
    int M = 1 << m, total = 0;
    for (int x = 1; x < M; x++)
        total += Z[x] != 0;
    for (int l = 0; l <= d; l++) {
        c.elems[l] = malloc(sizeof(int) * (total + 1));
        c.mask[l] = calloc(M, 1);
    }
    for (int x = 1; x < M; x++)
        if (Z[x]) {
            c.elems[0][c.count[0]++] = x;
            c.mask[0][x] = 1;
        }
    sub_rec(&c, 0, 0);
    for (int l = 0; l <= d; l++) {
        free(c.elems[l]);
        free(c.mask[l]);
    }
    *nodes_used = c.nodes;
    if (c.pair_found) {
        pair[0] = c.pair[0];
        pair[1] = c.pair[1];
        return -2;
    }
    return c.aborted ? -1 : c.found;
}

long long find_subspaces(int m, const uint8_t *Z, int d, int *out, long long cap,
                         long long node_limit, long long *nodes_used)
{
    long long pair[2];
    return find_subspaces_impl(m, Z, d, out, cap, node_limit, nodes_used, 0, pair, 0, 1);
}

/* The part-th of nparts disjoint slices of find_subspaces (split at the first level). */
long long find_subspaces_part(int m, const uint8_t *Z, int d, int *out, long long cap,
                              long long node_limit, long long *nodes_used, int part, int nparts)
{
    long long pair[2];
    return find_subspaces_impl(m, Z, d, out, cap, node_limit, nodes_used, 0, pair, part, nparts);
}

/* Like find_subspaces, but stops as soon as two found subspaces intersect
 * trivially: returns -2 and sets pair, -1 on cap/node limit, else the count. */
long long find_subspaces_pair(int m, const uint8_t *Z, int d, int *out, long long cap,
                              long long node_limit, long long *nodes_used, long long *pair)
{
    return find_subspaces_impl(m, Z, d, out, cap, node_limit, nodes_used, 1, pair, 0, 1);
}

/* Rank over GF(2) of k vectors of m bits. */
static int gf2_rank(int *v, int k)
{
    int rank = 0;
    for (int bit = 31; bit >= 0 && rank < k; bit--) {
        int p = -1;
        for (int r = rank; r < k; r++)
            if (v[r] >> bit & 1) { p = r; break; }
        if (p < 0)
            continue;
        int tmp = v[p]; v[p] = v[rank]; v[rank] = tmp;
        for (int r = 0; r < k; r++)
            if (r != rank && (v[r] >> bit & 1))
                v[r] ^= v[rank];
        rank++;
    }
    return rank;
}

/* Find two of the given d-dim subspaces (bases, d ints each) with trivial
 * intersection. Returns 1 and sets pair[0], pair[1], or 0 if none exists. */
int find_complementary_pair(int d, long long count, const int *bases, long long *pair)
{
    int v[64];
    for (long long a = 0; a < count; a++)
        for (long long b = a + 1; b < count; b++) {
            memcpy(v, bases + a * d, sizeof(int) * d);
            memcpy(v + d, bases + b * d, sizeof(int) * d);
            if (gf2_rank(v, 2 * d) == 2 * d) {
                pair[0] = a;
                pair[1] = b;
                return 1;
            }
        }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Switching kernel (Edel-Pott). For an APN function F on n bits (N = 2^n <= 4096)
 * and u != 0, G = F + u f is APN iff f(x) + f(x^a) + f(y) + f(y^a) = 0 for all
 * a != 0 and all x, y with F(x)^F(x^a)^F(y)^F(y^a) = u. These equations are
 * linear in the truth table of f. Writes a basis of the solution space to basis
 * (each vector N bits, packed into N/64 uint64 words, bit x = f(x)) and returns
 * its dimension, or -1 if F is not APN or max_basis is too small.
 */
int switching_kernel(int n, const int *F, int u, uint64_t *basis, int max_basis)
{
    int N = 1 << n, W = (N + 63) / 64;
    uint64_t *rows = calloc((size_t)N * W, sizeof(uint64_t));
    int *piv_row = malloc(sizeof(int) * N); /* pivot bit -> row index, or -1 */
    int *pos = malloc(sizeof(int) * N);
    int *d = malloc(sizeof(int) * N);
    uint64_t *r = malloc(sizeof(uint64_t) * W);
    int rank = 0, ok = 1;
    for (int i = 0; i < N; i++)
        piv_row[i] = -1;

    for (int a = 1; a < N && ok; a++) {
        for (int v = 0; v < N; v++)
            pos[v] = -1;
        for (int x = 0; x < N; x++) {
            d[x] = F[x] ^ F[x ^ a];
            if (x < (x ^ a)) {
                if (pos[d[x]] != -1) { ok = 0; break; } /* not APN */
                pos[d[x]] = x;
            }
        }
        if (!ok)
            break;
        for (int x = 0; x < N; x++) {
            if (x > (x ^ a))
                continue;
            int y = pos[d[x] ^ u];
            if (y < 0 || y < x)
                continue;
            memset(r, 0, sizeof(uint64_t) * W);
            int pts[4] = {x, x ^ a, y, y ^ a};
            for (int k = 0; k < 4; k++)
                r[pts[k] >> 6] ^= 1ULL << (pts[k] & 63);
            /* reduce by existing pivots (pivot = highest set bit) */
            for (;;) {
                int h = -1;
                for (int w = W - 1; w >= 0; w--)
                    if (r[w]) { h = w * 64 + 63 - __builtin_clzll(r[w]); break; }
                if (h < 0)
                    break;
                if (piv_row[h] < 0) {
                    memcpy(rows + (size_t)rank * W, r, sizeof(uint64_t) * W);
                    piv_row[h] = rank++;
                    break;
                }
                uint64_t *p = rows + (size_t)piv_row[h] * W;
                for (int w = 0; w < W; w++)
                    r[w] ^= p[w];
            }
        }
    }

    int dim = -1;
    if (ok) {
        /* full reduction: clear each pivot bit from all other rows */
        for (int h = N - 1; h >= 0; h--) {
            if (piv_row[h] < 0)
                continue;
            uint64_t *p = rows + (size_t)piv_row[h] * W;
            for (int j = 0; j < rank; j++) {
                if (j == piv_row[h])
                    continue;
                uint64_t *q = rows + (size_t)j * W;
                if (q[h >> 6] >> (h & 63) & 1)
                    for (int w = 0; w < W; w++)
                        q[w] ^= p[w];
            }
        }
        dim = N - rank;
        if (dim > max_basis) {
            dim = -1;
        } else {
            int k = 0;
            memset(basis, 0, sizeof(uint64_t) * W * (size_t)(N - rank));
            for (int j = 0; j < N; j++) {
                if (piv_row[j] >= 0)
                    continue; /* j is a free variable */
                uint64_t *b = basis + (size_t)k * W;
                b[j >> 6] |= 1ULL << (j & 63);
                for (int h = 0; h < N; h++) {
                    if (piv_row[h] < 0)
                        continue;
                    uint64_t *p = rows + (size_t)piv_row[h] * W;
                    if (p[j >> 6] >> (j & 63) & 1)
                        b[h >> 6] |= 1ULL << (h & 63);
                }
                k++;
            }
        }
    }
    free(rows); free(piv_row); free(pos); free(d); free(r);
    return dim;
}

/* ---------------------------------------------------------------------------
 * Randomized depth-first search for quadratic APN functions on n <= 8 bits.
 * F is determined by its alternating bilinear form B(e_i, e_j) = v[i][j];
 * F is APN iff B(x, y) != 0 for all linearly independent x, y. B is built
 * on V_k = span(e_0..e_{k-1}) for growing k; adding e_k means choosing
 * L(x) = B(x, e_k) linear on V_k with L(x) not in S_x = B(x, V_k) for all
 * nonzero x in V_k. Solutions F(x) = sum_{i<j} x_i x_j v[i][j] go to out
 * (N ints each). Returns the number of solutions found.
 */
typedef struct {
    int n, N;
    int v[8][8];
    uint64_t S[9][256][4];  /* S[k][x]: bitmap of S_x for x in V_k */
    int Lval[9][256];       /* Lval[k][x] = L(x) for the column being built */
    long long nodes, node_limit;
    int found, max_found;
    int *out;
} QCtx;

static int bform(QCtx *c, int x, int i)
{
    int r = 0;
    for (int t = 0; t < c->n; t++)
        if (x >> t & 1)
            r ^= c->v[t][i];
    return r;
}

static void q_build_S(QCtx *c, int k)
{
    int V = 1 << k;
    for (int x = 1; x < V; x++) {
        uint64_t *s = c->S[k][x];
        s[0] = 1; s[1] = s[2] = s[3] = 0;
        for (int i = 0; i < k; i++) {
            int gi = bform(c, x, i);
            if (s[gi >> 6] >> (gi & 63) & 1)
                continue; /* already in the span */
            uint64_t add[4] = {0, 0, 0, 0};
            for (int e = 0; e < c->N; e++)
                if (s[e >> 6] >> (e & 63) & 1) {
                    int f = e ^ gi;
                    add[f >> 6] |= 1ULL << (f & 63);
                }
            for (int w = 0; w < 4; w++)
                s[w] |= add[w];
        }
    }
}

static void q_rec(QCtx *c, int k, int j)
{
    if (c->found >= c->max_found || c->nodes > c->node_limit)
        return;
    c->nodes++;
    if (k == c->n) {
        int *F = c->out + (size_t)c->found * c->N;
        for (int x = 0; x < c->N; x++) {
            int y = 0;
            for (int i = 0; i < c->n; i++)
                for (int jj = i + 1; jj < c->n; jj++)
                    if ((x >> i & 1) && (x >> jj & 1))
                        y ^= c->v[i][jj];
            F[x] = y;
        }
        c->found++;
        return;
    }
    if (j == 0)
        q_build_S(c, k);
    if (j == k) {
        q_rec(c, k + 1, 0);
        return;
    }
    int start = rnd() % c->N, stride = (rnd() % (c->N / 2)) * 2 + 1;
    int lo = 1 << j;
    for (int t = 0; t < c->N; t++) {
        int w = (start + t * stride) & (c->N - 1);
        int ok = 1;
        /* new x in span(e_0..e_j) containing e_j */
        for (int x = lo; x < 2 * lo; x++) {
            int L = (x == lo ? 0 : c->Lval[k][x ^ lo]) ^ w;
            if (c->S[k][x][L >> 6] >> (L & 63) & 1) { ok = 0; break; }
            c->Lval[k][x] = L;
        }
        if (!ok)
            continue;
        c->v[j][k] = c->v[k][j] = w;
        q_rec(c, k, j + 1);
        if (c->found >= c->max_found || c->nodes > c->node_limit)
            return;
    }
}

int quadratic_apn_search(int n, uint64_t seed, long long node_limit, int max_found, int *out)
{
    QCtx *c = calloc(1, sizeof(QCtx));
    c->n = n;
    c->N = 1 << n;
    c->node_limit = node_limit;
    c->max_found = max_found;
    c->out = out;
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 7;
    q_rec(c, 1, 0);
    int found = c->found;
    free(c);
    return found;
}

/* Extend a quadratic APN-type form on n-1 bits (v_init: (n-1) x (n-1) matrix of
 * B(e_i, e_j) values in F_2^n) by one more basis vector, i.e. search only the
 * last column of B. Returns the number of APN functions on n bits found. */
int quadratic_apn_extend(int n, const int *v_init, uint64_t seed, long long node_limit,
                         int max_found, int *out)
{
    QCtx *c = calloc(1, sizeof(QCtx));
    c->n = n;
    c->N = 1 << n;
    c->node_limit = node_limit;
    c->max_found = max_found;
    c->out = out;
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - 1; j++)
            c->v[i][j] = v_init[i * (n - 1) + j];
    rng_state = seed * 0x9E3779B97F4A7C15ULL + 11;
    q_rec(c, n - 1, 0);
    int found = c->found;
    free(c);
    return found;
}

/* ---------------------------------------------------------------------------
 * APN test for quadratic functions from their bilinear matrix. v[k*n + l] =
 * B(e_k, e_l) (n <= 16). F is APN iff for every a != 0 the n vectors
 * r_i(a) = B(a, e_i) = XOR_{k in a} v[k][i] have rank n - 1. Rows are updated
 * along a Gray code. flags[t] = 1 if the t-th matrix of the batch is APN.
 */
static int rank_small(int *r, int k)
{
    int rank = 0;
    for (int bit = 31; bit >= 0 && rank < k; bit--) {
        int p = -1;
        for (int i = rank; i < k; i++)
            if (r[i] >> bit & 1) { p = i; break; }
        if (p < 0)
            continue;
        int t = r[p]; r[p] = r[rank]; r[rank] = t;
        for (int i = rank + 1; i < k; i++)
            if (r[i] >> bit & 1)
                r[i] ^= r[rank];
        rank++;
    }
    return rank;
}

void bilinear_apn_filter(int n, long long count, const int *vs, uint8_t *flags)
{
    int N = 1 << n;
    for (long long t = 0; t < count; t++) {
        const int *v = vs + t * n * n;
        int row[16] = {0}, tmp[16];
        int ok = 1;
        int prev = 0;
        for (int g = 1; g < N && ok; g++) {
            int a = g ^ (g >> 1);          /* Gray code: differs from prev in one bit */
            int k = __builtin_ctz((unsigned)(a ^ prev));
            prev = a;
            for (int i = 0; i < n; i++)
                row[i] ^= v[k * n + i];
            memcpy(tmp, row, sizeof(int) * n);
            if (rank_small(tmp, n) != n - 1)
                ok = 0;
        }
        flags[t] = (uint8_t)ok;
    }
}
