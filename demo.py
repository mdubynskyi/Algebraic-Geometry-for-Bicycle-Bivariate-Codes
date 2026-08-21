import time
import numpy as np
from bb_gb import BBCode, groebner_decode, gf2_decode

np.random.seed(0)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# Part 1: your original toy example, run through both decoders
# ------------------------------------------------------------------
hr("PART 1: your toy example (6 qubits, 3 checks)")

H = np.array([
    [1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
], dtype=np.uint8)
s = np.array([1, 1, 0], dtype=np.uint8)

sol, e0, kernel_basis, free_idx = groebner_decode(H, s, verbose=True)
print(f"\nfree variables exposed by the Groebner basis: {free_idx}")
print(f"(there are 2^{len(free_idx)} = {2**len(free_idx)} corrections consistent with this syndrome)")
print(f"particular solution (free vars = 0): {e0}")
print(f"minimum-weight solution chosen from the coset: {sol}  (weight {sol.sum()})")

sol2 = gf2_decode(H, s)
print(f"gf2_decode agrees: {np.array_equal(sol, sol2)}  -> {sol2}")
assert np.array_equal((H @ sol) % 2, s), "sanity check failed"


# ------------------------------------------------------------------
# Part 2: a genuine (tiny) BB code, both decoders on a random error
# ------------------------------------------------------------------
hr("PART 2: a small real BB code")

# l=2, m=3 -> n = 2*2*3 = 12 physical qubits, 6 X-checks, 6 Z-checks.
# (Toy-sized purely so the Groebner-basis version stays fast; real BB
#  codes from the literature use e.g. l=12, m=6 -> n=144.)
code = BBCode(l=2, m=3, A_terms=[(0, 0), (1, 0), (0, 1)],
                        B_terms=[(0, 0), (1, 1), (0, 2)])
print(f"n = {code.n} qubits, H_X shape {code.HX.shape}")

error = np.zeros(code.n, dtype=np.uint8)
error[[2, 7]] = 1  # a real weight-2 error
s = code.syndrome(error, check="X")
print(f"true error (weight {error.sum()}): {error}")
print(f"syndrome: {s}")

t0 = time.time()
sol_gb, e0, kernel_basis, free_idx = groebner_decode(code.HX, s, verbose=False)
t_gb = time.time() - t0

t0 = time.time()
sol_fast = gf2_decode(code.HX, s)
t_fast = time.time() - t0

print(f"\nGroebner-basis decode : {sol_gb}  (weight {sol_gb.sum()}, {t_gb*1000:.1f} ms, "
      f"{len(free_idx)} free vars -> {2**len(free_idx)} corrections in the coset)")
print(f"GF(2) linear-algebra  : {sol_fast}  (weight {sol_fast.sum()}, {t_fast*1000:.1f} ms)")
print(f"both agree: {np.array_equal(sol_gb, sol_fast)}")
print(f"both fix the syndrome: "
      f"{np.array_equal((code.HX @ sol_gb) % 2, s)}, {np.array_equal((code.HX @ sol_fast) % 2, s)}")
print(f"recovered the true error exactly: {np.array_equal(sol_fast, error)}")


# ------------------------------------------------------------------
# Part 3: why this matters -- scaling
# ------------------------------------------------------------------
hr("PART 3: scaling -- why gf2_decode is the one you actually run")

for (l, m) in [(2, 3), (3, 3), (4, 3)]:
    A_terms = [(0, 0), (1, 0), (0, 1)]
    B_terms = [(0, 0), (1, 1), (0, 2)]
    code = BBCode(l=l, m=m, A_terms=A_terms, B_terms=B_terms)
    error = np.zeros(code.n, dtype=np.uint8)
    error[np.random.choice(code.n, size=3, replace=False)] = 1
    s = code.syndrome(error, check="X")

    t0 = time.time()
    try:
        sol_gb, _, _, free_idx = groebner_decode(code.HX, s)
        t_gb = time.time() - t0
        gb_str = f"{t_gb*1000:8.1f} ms  ({len(free_idx)} free vars)"
    except Exception as ex:
        gb_str = f"failed/too slow ({ex})"

    t0 = time.time()
    sol_fast = gf2_decode(code.HX, s)
    t_fast = time.time() - t0

    print(f"n={code.n:4d}  Groebner: {gb_str:35s}  GF(2) linear algebra: {t_fast*1000:6.1f} ms")
