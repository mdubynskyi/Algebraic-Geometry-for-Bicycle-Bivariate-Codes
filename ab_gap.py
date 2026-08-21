"""
Test design: at each orbit_distance level (1, 2, 3), draw MANY random B's
(not one anecdotal path) and record the actual resulting minimum weight
found (up to 6), to get a distribution, not a single point estimate.
Repeated across two different lattices and two different base
polynomials for robustness.
"""
import itertools
import numpy as np
from bb_gb import BBCode, find_low_weight_codeword


def all_shifted_termsets(A_terms, l, m):
    return [frozenset(((a + s) % l, (b + t) % m) for (a, b) in A_terms)
            for s in range(l) for t in range(m)]


def orbit_distance(B_terms, shifted_sets):
    B = frozenset(B_terms)
    return min(len(B.symmetric_difference(S)) // 2 for S in shifted_sets)


def random_B_at_distance(A_terms, l, m, shifted_sets, target_dist, rng, all_positions):
    """Build a random 3-term B at EXACTLY the given orbit_distance from A."""
    base = list(list(shifted_sets[rng.integers(len(shifted_sets))]))
    if target_dist == 0:
        return base
    keep = rng.choice(len(base), size=3 - target_dist, replace=False)
    kept_terms = [base[i] for i in keep]
    remaining_pool = [p for p in all_positions if p not in kept_terms]
    new_terms = rng.choice(len(remaining_pool), size=target_dist, replace=False)
    B = kept_terms + [remaining_pool[i] for i in new_terms]
    # verify we actually landed at the intended distance (guards against
    # accidentally re-hitting a different orbit point)
    if orbit_distance(B, shifted_sets) != target_dist:
        return None
    return B


def run_experiment(l, m, A_terms, n_per_level, max_t, seed):
    rng = np.random.default_rng(seed)
    shifted_sets = all_shifted_termsets(A_terms, l, m)
    all_positions = [(a, b) for a in range(l) for b in range(m)]

    print(f"--- lattice ({l},{m}), A={A_terms}, {n_per_level} trials/level, checking weight<={max_t} ---")
    for dist in [0, 1, 2, 3]:
        weights_found = []
        n_valid = 0
        attempts = 0
        while n_valid < n_per_level and attempts < n_per_level * 20:
            attempts += 1
            B = random_B_at_distance(A_terms, l, m, shifted_sets, dist, rng, all_positions)
            if B is None:
                continue
            n_valid += 1
            code = BBCode(l=l, m=m, A_terms=A_terms, B_terms=B)
            t, _ = find_low_weight_codeword(code.HX, max_t=max_t)
            weights_found.append(t if t is not None else f">{max_t}")

        n_trapped = sum(1 for w in weights_found if w == 2)
        n_none_found = sum(1 for w in weights_found if isinstance(w, str))
        finite = [w for w in weights_found if isinstance(w, int)]
        mean_w = f"{np.mean(finite):.2f}" if finite else "n/a"
        print(f"  orbit_distance={dist}: {n_trapped}/{n_valid} exactly trapped (w=2), "
              f"{n_none_found}/{n_valid} found nothing <=6, mean weight (when found)={mean_w}")


run_experiment(l=12, m=6, A_terms=[(3,0),(0,1),(0,2)], n_per_level=20, max_t=6, seed=0)
print()
run_experiment(l=9, m=4, A_terms=[(2,0),(1,1),(0,3)], n_per_level=20, max_t=6, seed=1)

print()
print("=== filling the orbit_distance=3 gap with direct rejection sampling ===")
def run_distance3(l, m, A_terms, n_target, max_t, seed, max_attempts=20000):
    rng = np.random.default_rng(seed)
    shifted_sets = all_shifted_termsets(A_terms, l, m)
    all_positions = [(a, b) for a in range(l) for b in range(m)]
    weights_found = []
    n_valid = 0
    for _ in range(max_attempts):
        if n_valid >= n_target:
            break
        B = list(rng.choice(len(all_positions), size=3, replace=False))
        B = [all_positions[i] for i in B]
        if orbit_distance(B, shifted_sets) != 3:
            continue
        n_valid += 1
        code = BBCode(l=l, m=m, A_terms=A_terms, B_terms=B)
        t, _ = find_low_weight_codeword(code.HX, max_t=max_t)
        weights_found.append(t if t is not None else f">{max_t}")
    n_trapped = sum(1 for w in weights_found if w == 2)
    n_none = sum(1 for w in weights_found if isinstance(w, str))
    print(f"  ({l},{m}) orbit_distance=3: got {n_valid}/{n_target} valid samples, "
          f"{n_trapped} trapped, {n_none} found nothing <={max_t}")

run_distance3(l=12, m=6, A_terms=[(3,0),(0,1),(0,2)], n_target=20, max_t=6, seed=2)
run_distance3(l=9, m=4, A_terms=[(2,0),(1,1),(0,3)], n_target=20, max_t=6, seed=3)