"""
Test 1: Basic Correctness (Unit Tests), from BB_Codes_Explained validation spec.

Verifies that the decoder recovers known errors on small, hand-verified codes.

    Test  Code                                             Error pattern
    1.1   6-qubit BB code (l=3, m=1, A=1+x, B=1+x+x^2)      single error at v1
    1.2   same code                                          single error at v3
    1.3   same code                                          two errors at v1, v2
    1.4   [11,6,5] ternary code                              errors at positions 2,4

Conventions used here (not specified in the spec, so stated explicitly):
  * v_k (1-indexed, as in the spec) <-> e_{k-1} (0-indexed array position).
  * Tests 1.1-1.3 decode via H_X = [A | B]. (H_Z gives the mirror-image
    result -- see the report.)
  * Test 1.4 uses the standard [11,6,5]_3 ternary Golay code parity check
    matrix
"""
import unittest
import numpy as np

from bb_gb import BBCode, bounded_distance_decode, classical_min_distance

VALUES_1_4 = [1, 2]  # edit if you get the actual intended error values for 1.4

# standard [11,6,5]_3 ternary Golay parity check matrix (Wikipedia / MacWilliams & Sloane)
H_GOLAY = np.array([
    [2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0],
    [2, 2, 1, 2, 0, 1, 0, 1, 0, 0, 0],
    [2, 1, 2, 0, 2, 1, 0, 0, 1, 0, 0],
    [2, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0],
    [2, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1],
], dtype=np.uint8)


def v(n, positions_1indexed, values=None):
    """Build an error vector; v(6, [1,3]) or v(6,[2,4],[1,2]) for weighted errors."""
    e = np.zeros(n, dtype=np.uint8)
    if values is None:
        values = [1] * len(positions_1indexed)
    for p, val in zip(positions_1indexed, values):
        e[p - 1] = val
    return e


def show(title, report):
    print(f"\n=== {title} ===")
    print(report)


class Test1BasicCorrectness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.code = BBCode(l=3, m=1, A_terms=[(0, 0), (1, 0)],
                                     B_terms=[(0, 0), (1, 0), (2, 0)])
        cls.d_HX = classical_min_distance(cls.code.HX, q=2)
        cls.d_golay = classical_min_distance(H_GOLAY, q=3, brute_force_max_n=11)

    def test_1_1_single_error_v1(self):
        """Single error at v1 must be recovered exactly, with a triangular
        (zero free-variable) Groebner basis."""
        err = v(6, [1])
        s = (self.code.HX @ err) % 2
        r = bounded_distance_decode(self.code.HX, s, t=1, q=2)
        show("Test 1.1: single error at v1", r)
        self.assertEqual(r.status, 'unique', f"expected a triangular GB, got {r.status}, free vars: {r.free_vars}")
        np.testing.assert_array_equal(r.solution, err)

    def test_1_2_single_error_v3(self):
        """Single error at v3 must be recovered exactly."""
        err = v(6, [3])
        s = (self.code.HX @ err) % 2
        r = bounded_distance_decode(self.code.HX, s, t=1, q=2)
        show("Test 1.2: single error at v3", r)
        self.assertEqual(r.status, 'unique', f"expected a triangular GB, got {r.status}, free vars: {r.free_vars}")
        np.testing.assert_array_equal(r.solution, err)

    def test_1_3_two_errors_v1_v2(self):
        self.assertEqual(self.d_HX, 2, "if this ever changes, re-derive the test below")

        err = v(6, [1, 2])
        s = (self.code.HX @ err) % 2
        r = bounded_distance_decode(self.code.HX, s, t=2, q=2)
        show("Test 1.3: two errors at v1, v2 (expect ambiguity, not a unique answer)", r)

        self.assertEqual(r.status, 'underdetermined',
                          "if this now resolves uniquely, the code changed -- re-check")
        sols = [tuple(a.tolist()) for a in r.alternatives]
        self.assertIn(tuple(err.tolist()), sols, "the true error should still be among the alternatives")
        self.assertIn(tuple(v(6, [3]).tolist()), sols, "v3 alone should be the lower-weight alternative")

    def test_1_4_ternary_golay_two_errors(self):
        """Errors at positions 2 and 4 on the [11,6,5]_3 ternary Golay code,
        t=2 (within its guaranteed radius since d=5 -> floor((5-1)/2)=2)."""
        self.assertEqual(self.d_golay, 5)

        err = v(11, [2, 4], values=VALUES_1_4)
        s = (H_GOLAY @ err) % 3
        r = bounded_distance_decode(H_GOLAY, s, t=2, q=3)
        show("Test 1.4: ternary Golay, errors at positions 2, 4", r)
        self.assertEqual(r.status, 'unique', f"expected a triangular GB, got {r.status}, free vars: {r.free_vars}")
        np.testing.assert_array_equal(r.solution, err)


if __name__ == "__main__":
    print(__doc__)
    unittest.main(verbosity=2)