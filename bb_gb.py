"""
Algebraic-geometry (Groebner basis) decoder for Bivariate Bicycle (BB) codes.

BACKGROUND
----------
A BB code is defined by two polynomials A, B in F2[x,y]/(x^l-1, y^m-1),
each a sum of monomials x^a y^b. Physical qubits sit on two copies (L, R)
of an l x m grid, n = 2*l*m qubits total. With

    x = S_l (x) I_m       (kron product, S_l = cyclic shift of size l)
    y = I_l (x) S_m

the check matrices are

    H_X = [ A | B ]     (lm x 2lm)
    H_Z = [ B^T | A^T ] (lm x 2lm)

Decoding H_X (or H_Z) means: given a syndrome s, find an error vector e in
F2^n with H e = s (mod 2), ideally the lowest-weight one.

THE ALGEBRAIC FORMULATION (your original idea, corrected)
-----------------------------------------------------------
For each qubit i:            e_i^2 + e_i = 0            (Boolean constraint)
For each check row r:        sum_{i in row r} e_i - s_r = 0

The variety of the resulting ideal I = <field eqs, syndrome eqs> is EXACTLY
the coset {e0 + k : k in ker(H)} of valid corrections. A lex Groebner basis
of I does not collapse this to one point in general -- it triangularizes it
into a set of "pivot" variables expressed in terms of "free" Boolean
variables (this is the honest algebraic content of Buchberger's algorithm
here). Picking values for the free variables is a *separate* combinatorial
step -- that's where "which correction is most likely" actually gets
decided, and it's the part that makes decoding hard in general.

This module gives you both:
  * groebner_decode(...)   -- does it the way you described, via an actual
                               Groebner basis (sympy). Great for small/demo
                               sizes and for *seeing* the free-variable
                               structure explicitly. Gets slow fast.
  * gf2_decode(...)         -- the scalable version: GF(2) Gaussian
                               elimination to get a particular solution +
                               an explicit kernel basis, then a weight-
                               minimization step over the free dimensions
                               (brute force if small, else a greedy
                               heuristic). This is what you'd actually run
                               on a real BB code.

Both return the same kind of object: a particular solution plus a basis for
the free/kernel directions, so you can see the equivalence directly.
"""

from __future__ import annotations
import itertools
import numpy as np
import sympy
from sympy import symbols, groebner, Poly
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# 1. Building H_X, H_Z for a BB code from its defining polynomials
# ----------------------------------------------------------------------

def _cyclic_shift(n: int) -> np.ndarray:
    """S_n: the n x n cyclic shift permutation matrix, entries mod 2 (0/1)."""
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def _monomial_matrix(l: int, m: int, a: int, b: int) -> np.ndarray:
    """Matrix for x^a y^b acting on the l*m dimensional qubit-index space,
    with x = S_l (x) I_m and y = I_l (x) S_m."""
    Sl_a = np.linalg.matrix_power(_cyclic_shift(l), a % l)
    Sm_b = np.linalg.matrix_power(_cyclic_shift(m), b % m)
    return np.kron(Sl_a, Sm_b) % 2


def poly_matrix(l: int, m: int, terms: List[Tuple[int, int]]) -> np.ndarray:
    """Sum (mod 2) of monomial matrices x^a y^b for (a, b) in terms."""
    lm = l * m
    M = np.zeros((lm, lm), dtype=np.uint8)
    for (a, b) in terms:
        M = (M + _monomial_matrix(l, m, a, b)) % 2
    return M


@dataclass
class BBCode:
    l: int
    m: int
    A_terms: List[Tuple[int, int]]
    B_terms: List[Tuple[int, int]]
    HX: np.ndarray = field(init=False)
    HZ: np.ndarray = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        A = poly_matrix(self.l, self.m, self.A_terms)
        B = poly_matrix(self.l, self.m, self.B_terms)
        self.HX = np.concatenate([A, B], axis=1) % 2        # lm x 2lm
        self.HZ = np.concatenate([B.T, A.T], axis=1) % 2     # lm x 2lm
        self.n = 2 * self.l * self.m

    def syndrome(self, error: np.ndarray, check: str = "X") -> np.ndarray:
        H = self.HX if check == "X" else self.HZ
        return (H @ (error.astype(np.uint8) % 2)) % 2


# ----------------------------------------------------------------------
# 2. GF(2) linear algebra: particular solution + kernel basis
# ----------------------------------------------------------------------

def gf2_rref_augmented(H: np.ndarray, s: np.ndarray):
    """Row-reduce [H | s] over GF(2). Returns (R, s_r, pivot_cols)."""
    R = H.copy().astype(np.uint8) % 2
    sr = s.copy().astype(np.uint8) % 2
    rows, cols = R.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if R[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        R[[r, piv]] = R[[piv, r]]
        sr[[r, piv]] = sr[[piv, r]]
        for i in range(rows):
            if i != r and R[i, c] == 1:
                R[i, :] ^= R[r, :]
                sr[i] ^= sr[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return R, sr, pivot_cols, r


def gf2_solve(H: np.ndarray, s: np.ndarray):
    """
    Solve H e = s over GF(2).
    Returns (particular_solution, kernel_basis, free_cols, consistent).
    kernel_basis: list of vectors spanning ker(H) restricted to the
    columns that ended up free after elimination (i.e. a basis of ker(H)).
    """
    n = H.shape[1]
    R, sr, pivot_cols, rank = gf2_rref_augmented(H, s)

    # consistency check: any all-zero row of R with sr=1 -> no solution
    for i in range(rank, H.shape[0]):
        if R[i].sum() == 0 and sr[i] == 1:
            return None, None, None, False

    free_cols = [c for c in range(n) if c not in pivot_cols]

    # particular solution: free vars = 0
    e0 = np.zeros(n, dtype=np.uint8)
    for row_idx, c in enumerate(pivot_cols):
        e0[c] = sr[row_idx]

    # kernel basis: for each free column, set it to 1, others free = 0,
    # solve for pivot columns from the homogeneous system
    kernel_basis = []
    for fc in free_cols:
        k = np.zeros(n, dtype=np.uint8)
        k[fc] = 1
        for row_idx, c in enumerate(pivot_cols):
            k[c] = R[row_idx, fc]
        kernel_basis.append(k)

    return e0, kernel_basis, free_cols, True


def min_weight_over_coset(e0: np.ndarray, kernel_basis: List[np.ndarray],
                           max_bruteforce_dim: int = 20) -> np.ndarray:
    """
    Find (or approximate) the minimum-weight vector in {e0 + sum c_i k_i}.
    Exact brute force if the kernel dimension is small; otherwise a greedy
    local-search heuristic (flip whichever kernel vector reduces weight
    most, repeat). For real BB codes you generally want BP+OSD instead --
    see the note in demo_realistic_size().
    """
    dim = len(kernel_basis)
    if dim == 0:
        return e0
    if dim <= max_bruteforce_dim:
        best = e0
        best_w = int(e0.sum())
        for bits in itertools.product([0, 1], repeat=dim):
            v = e0.copy()
            for bit, k in zip(bits, kernel_basis):
                if bit:
                    v ^= k
            w = int(v.sum())
            if w < best_w:
                best_w, best = w, v
        return best

    # greedy heuristic for large kernel dimension
    cur = e0.copy()
    improved = True
    while improved:
        improved = False
        cur_w = int(cur.sum())
        for k in kernel_basis:
            cand = cur ^ k
            if int(cand.sum()) < cur_w:
                cur = cand
                improved = True
                break
    return cur


def gf2_decode(H: np.ndarray, s: np.ndarray, minimize_weight: bool = True):
    """The scalable decoder: Gaussian elimination + weight minimization."""
    e0, kernel_basis, free_cols, ok = gf2_solve(H, s)
    if not ok:
        raise ValueError("syndrome is not in the row space of H -- no valid correction exists")
    if minimize_weight:
        return min_weight_over_coset(e0, kernel_basis)
    return e0


# ----------------------------------------------------------------------
# 3. The Groebner-basis decoder (your original approach, made explicit
#    about the free-variable structure it actually produces)
# ----------------------------------------------------------------------

def groebner_decode(H: np.ndarray, s: np.ndarray, minimize_weight: bool = True,
                     verbose: bool = False):
    """
    Builds I = <e_i^2 - e_i, syndrome rows> and computes a lex Groebner
    basis over GF(2). Returns the same kind of (solution, free_vars) info
    as gf2_decode, but derived via Buchberger's algorithm, so you can see
    the two agree. Only practical for small H (say n <= ~40) -- generic
    Groebner-basis software does not exploit the fact that these
    particular equations are linear, so it scales far worse than the
    Gaussian-elimination version above for the same problem.
    """
    m_rows, n = H.shape
    e = symbols(f"e0:{n}")
    field_eqs = [ei**2 - ei for ei in e]
    synd_eqs = []
    for r in range(m_rows):
        cols = [e[i] for i in range(n) if H[r, i] == 1]
        lhs = sum(cols) if cols else sympy.Integer(0)
        synd_eqs.append(lhs - int(s[r]))

    G = groebner(field_eqs + synd_eqs, *e, order='lex', modulus=2)

    if verbose:
        print("Groebner basis:")
        for g in G.polys:
            print(" ", g.as_expr())

    # Parse the reduced GB: every polynomial here is either
    #   e_i + (const)                     -> pivot var, forced value
    #   e_i + (sum of other e_j) + const  -> pivot var, expressed via free vars
    #   e_i^2 + e_i                       -> e_i is free (0 or 1)
    pivot_value = {}      # var index -> {'const': 0/1, 'free_terms': {var_idx: coeff}}
    free_vars = set()

    for g in G.polys:
        expr = g.as_expr()
        p = Poly(expr, *e, modulus=2)
        if p.total_degree() == 2 and len(p.free_symbols) == 1:
            # e_i^2 + e_i  -> e_i is a genuinely free Boolean variable
            (v,) = p.free_symbols
            free_vars.add(e.index(v))
            continue
        # linear relation: leading var (lowest index under lex, first symbol
        # appearing) is the pivot; everything else on the RHS is "free" so far
        involved = sorted(e.index(v) for v in p.free_symbols)
        if not involved:
            continue
        pivot = involved[0]
        const_expr = expr.subs({v: 0 for v in p.free_symbols})
        const = int(const_expr) % 2
        free_terms = {j: 1 for j in involved[1:]}  # GF(2): all coeffs are 1
        pivot_value[pivot] = {"const": const, "free_terms": free_terms}

    all_idx = set(range(n))
    pivots = set(pivot_value.keys())
    free_idx = sorted(all_idx - pivots)  # anything not pinned down is free

    # particular solution: free vars = 0
    e0 = np.zeros(n, dtype=np.uint8)
    # resolve in order so dependencies (free vars) are already filled in
    for i in free_idx:
        e0[i] = 0
    changed = True
    remaining = dict(pivot_value)
    while remaining:
        progressed = False
        for pv, info in list(remaining.items()):
            if all((j in free_idx) or (j not in remaining) for j in info["free_terms"]):
                val = info["const"]
                for j in info["free_terms"]:
                    val ^= int(e0[j])
                e0[pv] = val
                del remaining[pv]
                progressed = True
        if not progressed:
            break  # shouldn't happen for a consistent triangular system

    # kernel basis: for each free var, set it to 1 (others 0), propagate
    kernel_basis = []
    for fc in free_idx:
        k = np.zeros(n, dtype=np.uint8)
        k[fc] = 1
        remaining = dict(pivot_value)
        while remaining:
            progressed = False
            for pv, info in list(remaining.items()):
                if all((j in free_idx) or (j not in remaining) for j in info["free_terms"]):
                    val = 0  # homogeneous part only (const dropped for kernel dir)
                    for j in info["free_terms"]:
                        val ^= int(k[j])
                    k[pv] = val
                    del remaining[pv]
                    progressed = True
            if not progressed:
                break
        kernel_basis.append(k)

    if minimize_weight:
        sol = min_weight_over_coset(e0, kernel_basis)
    else:
        sol = e0

    return sol, e0, kernel_basis, free_idx


# ----------------------------------------------------------------------
# 4. Bounded-distance Groebner decoding: makes Buchberger's algorithm
#    itself return a genuinely triangular (zero-free-variable) basis,
#    over GF(q) for any prime q -- not just GF(2).
#
#    This is the missing ingredient from the original algorithm sketch:
#    the raw ideal in section 3 is underdetermined because ker(H) is
#    nontrivial. Here we additionally constrain the ideal to "weight of e
#    is at most t", which collapses the variety to a single point whenever
#    t is within the code's unique-decoding radius. This is the technique
#    the literature calls "Cooper's philosophy" / bounded-distance
#    decoding with Groebner bases (Cooper; Fitzgerald-Lax 1998; Bulygin &
#    Pellikaan 2009) -- the same family of methods implemented in
#    Singular's decodegb.lib.
#
#    THE TRICK: for e_i ranging over GF(q), e_i^(q-1) is exactly the
#    indicator "1 if e_i != 0 else 0" (Fermat: e_i^q = e_i for all
#    e_i in GF(q), so e_i^(q-1) = 1 when e_i != 0 and = 0 when e_i = 0).
#    So weight(e) <= t  <=>  every elementary symmetric polynomial
#    E_k(e_1^(q-1), ..., e_n^(q-1)) vanishes for k = t+1, ..., n
#    (if weight(e) = w, E_k is a sum over k-subsets that are all nonzero;
#    there are none once k > w). Adding these as ideal generators, on top
#    of the field equations and syndrome equations, is what makes the
#    reduced Groebner basis triangular.
# ----------------------------------------------------------------------

from sympy.polys.polyfuncs import symmetric_poly


def _format_linear(coeffs: dict, q: int) -> str:
    """coeffs: {var_index: nonzero coeff}. Render as 'c*e_i + ...' LHS string."""
    terms = []
    for i in sorted(coeffs):
        c = coeffs[i] % q
        terms.append(f"e{i}" if c == 1 else f"{c}*e{i}")
    return " + ".join(terms) if terms else "0"


@dataclass
class DecodeReport:
    """
    Everything about one bounded_distance_decode() call, in a form that can
    be printed as a full worked derivation (report.pretty(), or just
    print(report)) or consumed programmatically (report.status,
    report.solution, ...).
    """
    status: str                              # 'unique' | 'no_solution' | 'underdetermined'
    n: int
    q: int
    t: int
    syndrome_equations: List[str]
    field_equation_note: str
    weight_equation_note: str
    groebner_basis: List[str]
    solution: Optional[np.ndarray] = None
    free_vars: Optional[List[str]] = None
    alternatives: Optional[List[np.ndarray]] = None
    truncated: bool = False

    def pretty(self) -> str:
        lines = [f"--- setup: n={self.n} symbols over GF({self.q}), target weight t<={self.t} ---",
                 "syndrome equations:"]
        lines += [f"  {eq}" for eq in self.syndrome_equations]
        lines += [self.field_equation_note, self.weight_equation_note,
                  f"reduced Groebner basis ({len(self.groebner_basis)} elements):"]
        lines += [f"  {g}" for g in self.groebner_basis]

        if self.status == 'unique':
            w = int(np.count_nonzero(self.solution))
            lines.append(f"triangular -> unique solution: {self.solution.tolist()}  (weight {w})")
        elif self.status == 'no_solution':
            lines.append("basis reduced to {1} -> NO error of weight <= t reproduces this syndrome")
        else:
            lines.append(f"NOT triangular -> free variable(s): {self.free_vars}")
            if self.alternatives is not None:
                prefix = "first " if self.truncated else ""
                lines.append(f"all {prefix}{len(self.alternatives)} solutions consistent with weight<=t:")
                for v in self.alternatives:
                    w = int(np.count_nonzero(v))
                    lines.append(f"  {v.tolist()}  (weight {w})")
                if self.truncated:
                    lines.append("  ... (truncated)")
        return "\n".join(lines)

    def __str__(self):
        return self.pretty()


def bounded_distance_decode(H: np.ndarray, s: np.ndarray, t: int, q: int = 2,
                             max_enumerate: int = 64) -> DecodeReport:
    """
    Decode syndrome s for check matrix H (entries in GF(q)) assuming
    weight(e) <= t, via a single Groebner basis computation. Returns a
    DecodeReport carrying the actual equations, the actual reduced
    Groebner basis, and the read-off answer (or, if the basis isn't
    triangular, every alternative consistent with the constraints, up to
    max_enumerate) -- print(report) shows the full worked derivation.

    Only practical for small n (the weight constraints have up to C(n,t)
    terms) -- this is an exact small-scale / verification tool, matching
    the scope of "basic correctness" testing, not a production decoder
    for full-size BB codes.
    """
    m_rows, n = H.shape
    e = symbols(f"e0:{n}")

    field_eqs = [ei**q - ei for ei in e]
    synd_eqs_poly, synd_eqs_str = [], []
    for r in range(m_rows):
        coeffs = {i: int(H[r, i]) % q for i in range(n) if int(H[r, i]) % q != 0}
        rhs = int(s[r]) % q
        synd_eqs_poly.append(sum(c * e[i] for i, c in coeffs.items()) - rhs)
        synd_eqs_str.append(f"{_format_linear(coeffs, q)} = {rhs}")

    indicators = [ei**(q - 1) for ei in e]     # 1[e_i != 0], exactly (Fermat)
    weight_eqs = [symmetric_poly(k, *indicators) for k in range(t + 1, n + 1)]

    G = groebner(field_eqs + synd_eqs_poly + weight_eqs, *e, order='lex', modulus=q)
    gb_strs = [str(g.as_expr()) for g in G.polys]

    field_note = f"field equations: e_i^{q} - e_i = 0 for i=0..{n-1}  (forces e_i in GF({q}))"
    weight_note = (f"weight<=t constraint: E_k(ind_0..ind_{n-1}) = 0 for k={t+1}..{n}, "
                   f"where ind_i = e_i^{q-1} (=1 iff e_i != 0)")
    base_args = (n, q, t, synd_eqs_str, field_note, weight_note, gb_strs)

    if len(G.polys) == 1 and G.polys[0].is_one:
        return DecodeReport('no_solution', *base_args)

    # Parse the reduced basis: each element is either a genuinely free
    # variable's field equation (e_i^2+e_i / e_i^2-1, degree>=2, one symbol),
    # or a pivot relation e_pivot = const + sum_j coeff_j * e_j.
    pivot_info = {}          # var_index -> (const, {other_var_index: coeff})
    free_symbol_indices = set()
    for g in G.polys:
        expr = g.as_expr()
        p = Poly(expr, *e, modulus=q)
        syms = p.free_symbols
        if len(syms) == 1 and p.total_degree() >= 2:
            free_symbol_indices.add(e.index(next(iter(syms))))
            continue
        involved = sorted(e.index(v) for v in syms)
        if not involved:
            continue
        pivot = involved[0]
        coeffs = {j: c for j in involved[1:] if (c := p.coeff_monomial(e[j]) % q) != 0}
        const = int((-expr.subs({v: 0 for v in syms})) % q)
        pivot_info[pivot] = (const, coeffs)

    pivots = set(pivot_info.keys())
    free_idx = sorted((set(range(n)) - pivots) | free_symbol_indices)

    def resolve(free_values: dict) -> np.ndarray:
        vec = [None] * n
        for i in free_idx:
            vec[i] = free_values[i]
        remaining = dict(pivot_info)
        while remaining:
            progressed = False
            for pv, (const, coeffs) in list(remaining.items()):
                if all(vec[j] is not None for j in coeffs):
                    val = const
                    for j, c in coeffs.items():
                        val = (val - c * vec[j]) % q
                    vec[pv] = val % q
                    del remaining[pv]
                    progressed = True
            if not progressed:
                break
        return np.array(vec, dtype=np.uint8)

    if not free_idx:
        return DecodeReport('unique', *base_args, solution=resolve({}))

    total = q ** len(free_idx)
    truncated = total > max_enumerate
    alts = []
    for combo in itertools.islice(itertools.product(range(q), repeat=len(free_idx)), max_enumerate):
        alts.append(resolve(dict(zip(free_idx, combo))))
    return DecodeReport('underdetermined', *base_args,
                         free_vars=[f"e{i}" for i in free_idx],
                         alternatives=alts, truncated=truncated)


def classical_min_distance(H: np.ndarray, q: int = 2, brute_force_max_n: int = 16) -> int:
    """
    Exact minimum weight of a nonzero vector in ker(H) over GF(q), by brute
    force. Only for small n -- this is what determines the unique-decoding
    radius t <= floor((d-1)/2) that bounded_distance_decode can guarantee.
    """
    n = H.shape[1]
    if n > brute_force_max_n:
        raise ValueError(f"n={n} too large for brute force (limit {brute_force_max_n})")
    best = None
    for bits in itertools.product(range(q), repeat=n):
        v = np.array(bits, dtype=np.uint8)
        if not v.any():
            continue
        if not ((H @ v) % q).any():
            w = int(np.count_nonzero(v))
            if best is None or w < best:
                best = w
    return best