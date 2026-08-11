from simplicial_cup_product.simp_homology import SimpHomology
from simplicial_cup_product.simplicial_complex import SimplicialComplex
import numpy as np
import pytest
from sympy import GF, Matrix
from sympy.polys.matrices import DomainMatrix


# The boundary of the 3-simplex: the vertex-minimal S^2, four triangles glued
# along all six of their edges. V - E + F = 4 - 6 + 4 = 2.
SPHERE = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

# Moebius' 7-vertex torus, the vertex-minimal triangulation: the triangles
# {i, i+1, i+3} and {i, i+2, i+3} taken mod 7. V - E + F = 7 - 21 + 14 = 0.
TORUS = [(i % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)] \
      + [(i % 7, (i + 2) % 7, (i + 3) % 7) for i in range(7)]

# The 6-vertex RP^2, the vertex-minimal triangulation: the quotient of the
# icosahedron by the antipodal map, so every pair of vertices is an edge and
# exactly one of each complementary pair of triples is a face.
# V - E + F = 6 - 15 + 10 = 1.
RP2 = [(1, 2, 5), (1, 2, 6), (1, 3, 4), (1, 3, 6), (1, 4, 5),
       (2, 3, 4), (2, 3, 5), (2, 4, 6), (3, 5, 6), (4, 5, 6)]

# The 9-vertex Klein bottle: the 3x3 square grid on vertices (i, j), labelled
# 3i + j, with the vertical edges glued straight and the horizontal ones glued
# with a flip, (i, j) ~ (i + 3, -j). Every grid square is cut along a diagonal.
# V - E + F = 9 - 27 + 18 = 0.
KLEIN = [(0, 1, 4), (0, 1, 8), (0, 2, 3), (0, 2, 6), (0, 3, 4), (0, 6, 8),
         (1, 2, 5), (1, 2, 7), (1, 4, 5), (1, 7, 8), (2, 3, 5), (2, 6, 7),
         (3, 4, 7), (3, 5, 6), (3, 6, 7), (4, 5, 8), (4, 7, 8), (5, 6, 8)]

# Kuehnel's 9-vertex CP^2, the vertex-minimal triangulation: the unique
# 3-neighborly 9-vertex combinatorial 4-manifold, with a symmetry group of
# order 54 acting transitively on the vertices. Its f-vector is
# (9, 36, 84, 90, 36), so V - E + F - T - P = 3, and H_* = (Z, 0, Z, 0, Z).
# W. Kuehnel and T. F. Banchoff, "The 9-vertex complex projective plane",
# Math. Intelligencer 5 (1983), 11-22.
CP2 = [(1, 2, 3, 7, 8), (1, 2, 3, 7, 9), (1, 2, 3, 8, 9),
       (1, 2, 4, 5, 6), (1, 2, 4, 5, 9), (1, 2, 4, 6, 7),
       (1, 2, 4, 7, 9), (1, 2, 5, 6, 8), (1, 2, 5, 8, 9),
       (1, 2, 6, 7, 8), (1, 3, 4, 5, 6), (1, 3, 4, 5, 7),
       (1, 3, 4, 6, 8), (1, 3, 4, 7, 8), (1, 3, 5, 6, 9),
       (1, 3, 5, 7, 9), (1, 3, 6, 8, 9), (1, 4, 5, 7, 9),
       (1, 4, 6, 7, 8), (1, 5, 6, 8, 9), (2, 3, 4, 5, 6),
       (2, 3, 4, 5, 8), (2, 3, 4, 6, 9), (2, 3, 4, 8, 9),
       (2, 3, 5, 6, 7), (2, 3, 5, 7, 8), (2, 3, 6, 7, 9),
       (2, 4, 5, 8, 9), (2, 4, 6, 7, 9), (2, 5, 6, 7, 8),
       (3, 4, 5, 7, 8), (3, 4, 6, 8, 9), (3, 5, 6, 7, 9),
       (4, 5, 7, 8, 9), (4, 6, 7, 8, 9), (5, 6, 7, 8, 9)]

# H_d(K; Z) for d = 0, 1, 2, each as (betti number, torsion coefficients).
# All four surfaces are closed and connected, so H_0 = Z throughout, and
# H_2 = Z exactly for the orientable ones.
#   S^2:   (Z, 0, Z)
#   torus: (Z, Z^2, Z)
#   RP^2:  (Z, Z/2, 0)      -- non-orientable, so H_2 vanishes
#   Klein: (Z, Z + Z/2, 0)  -- non-orientable, so H_2 vanishes
SURFACES = [
    pytest.param(SPHERE, ((1, ()), (0, ()), (1, ())), id="sphere"),
    pytest.param(TORUS, ((1, ()), (2, ()), (1, ())), id="torus"),
    pytest.param(RP2, ((1, ()), (0, (2,)), (0, ())), id="rp2"),
    pytest.param(KLEIN, ((1, ()), (1, (2,)), (0, ())), id="klein_bottle"),
]

SURFACES_MOD2 = [
    pytest.param(SPHERE, ((1, ()), (0, ()), (1, ())), id="sphere"),
    pytest.param(TORUS, ((1, ()), (2, ()), (1, ())), id="torus"),
    pytest.param(RP2, ((1, ()), (1, ()), (1, ())), id="rp2"),
    pytest.param(KLEIN, ((1, ()), (2, ()), (1, ())), id="klein_bottle"),
]

@pytest.mark.parametrize("triangles, homology_nums", SURFACES)
def test_homology_ZZ(triangles, homology_nums):
    K = SimplicialComplex()
    for triangle in triangles:
        K.add_simplex(triangle)
    H = SimpHomology(K)

    assert tuple(H.d_homology(d) for d in range(0, 3)) == homology_nums

@pytest.mark.parametrize("triangles, homology_nums", SURFACES_MOD2)
def test_homology_Z2(triangles, homology_nums):
    K = SimplicialComplex()
    for triangle in triangles:
        K.add_simplex(triangle)
    H = SimpHomology(K, 2)

    assert tuple(H.d_homology(d) for d in range(0, 3)) == homology_nums

@pytest.mark.parametrize("triangles, homology_nums", SURFACES_MOD2)
def test_cohomology_Z2(triangles, homology_nums):
    K = SimplicialComplex()
    for triangle in triangles:
        K.add_simplex(triangle)
    H = SimpHomology(K, 2)

    assert tuple(H.d_cohomology(d) for d in range(0, 3)) == tuple(rank for rank, torsion in homology_nums)

@pytest.mark.parametrize("triangles",
                         [pytest.param(p.values[0], id=p.id) for p in SURFACES])
def test_homology_ZZ_euler_characteristic(triangles):
    # the Euler characteristic is the alternating sum of the betti numbers,
    # and for a closed surface it agrees with V - E + F.
    K = SimplicialComplex()
    for triangle in triangles:
        K.add_simplex(triangle)
    H = SimpHomology(K)

    betti = [H.d_homology(d)[0] for d in range(0, 3)]
    faces = [len(K.sorted_simplices[d]) for d in range(0, 3)]

    assert sum((-1)**d * b for d, b in enumerate(betti)) \
        == sum((-1)**d * f for d, f in enumerate(faces))


@pytest.mark.parametrize("d", [-1, 3])
def test_homology_ZZ_rejects_out_of_range_dimension(d):
    K = SimplicialComplex()
    for triangle in TORUS:
        K.add_simplex(triangle)

    with pytest.raises(ValueError):
        SimpHomology(K).d_homology(d)


# --------------------------------------------------------------------------
# cycle_reps: explicit representatives for a basis of H_d(K; Z_p). a basis of
# the cycles Z_d is sieved against the echelon basis of the boundaries B_d
# produced by _sieve_basis(d + 1), and whatever survives with a new leading
# term generates the quotient -- so there should be dim H_d(K; Z_p) of them.
# --------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7]

# dim H_d(K; Z_p) for d = 0 .. dim K. only the non-orientable surfaces feel p:
# their Z/2 in H_1 is invisible mod odd p, but mod 2 it both survives in H_1
# and, by the universal coefficient theorem, contributes a class to H_2 -- so
# every closed surface looks orientable over Z_2, with the sum of all faces as
# its fundamental class. CP^2 is torsion-free, so its betti numbers are the
# integral ones for every p.
MOD_P_BETTI = {
    "sphere":       lambda p: (1, 0, 1),
    "torus":        lambda p: (1, 2, 1),
    "rp2":          lambda p: (1, 1, 1) if p == 2 else (1, 0, 0),
    "klein_bottle": lambda p: (1, 2, 1) if p == 2 else (1, 1, 0),
    "cp2":          lambda p: (1, 0, 1, 0, 1),
}

COMPLEXES = [("sphere", SPHERE), ("torus", TORUS), ("rp2", RP2),
             ("klein_bottle", KLEIN), ("cp2", CP2)]

# every (complex, prime) pair, carrying the mod p betti numbers expected of it.
CYCLE_REPS = [
    pytest.param(faces, p, MOD_P_BETTI[name](p), id=f"{name}-{p}")
    for name, faces in COMPLEXES for p in PRIMES
]

# the same pairs split out one degree at a time. shared with the cocycle tests
# below, where the same grid of complexes, primes and degrees applies.
REPS_DEGREES = [
    pytest.param(faces, p, d, id=f"{name}-{p}-{d}")
    for name, faces in COMPLEXES for p in PRIMES
    for d in range(len(MOD_P_BETTI[name](p)))
]


def build(faces) -> SimplicialComplex:
    K = SimplicialComplex()
    for face in faces:
        K.add_simplex(face)
    return K


def rank_mod_p(matrix, p: int) -> int:
    return DomainMatrix.from_Matrix(Matrix(matrix)).convert_to(GF(p)).rank()


# cycle_reps hands back a dict keyed by leading term when there are boundaries
# to quotient by, and a bare tuple of vectors when there are none. either way
# it is just a list of representatives as far as these tests are concerned.
def reps_list(reps) -> list:
    if isinstance(reps, dict):
        return [reps[pivot] for pivot in sorted(reps)]
    return list(reps)


# the representatives as the columns of a single matrix, with the right number
# of rows even when there are none of them.
def reps_matrix(reps, rows: int) -> np.array:
    vectors = reps_list(reps)
    if not vectors:
        return np.zeros((rows, 0), dtype=int)
    return np.column_stack(vectors)


# the columns spanning the d-boundaries B_d. nothing bounds in the top
# dimension, where there are no (d + 1)-chains to take the boundary of.
def boundaries_matrix(H: SimpHomology, d: int) -> np.array:
    if d < H.K.dim:
        return H.C.d_boundary_matrix(d + 1)
    return np.zeros((H.C.d_rank(d), 0), dtype=int)


@pytest.mark.parametrize("faces, p, betti", CYCLE_REPS)
def testcycle_reps_count_is_the_mod_p_betti_number(faces, p, betti):
    # the headline claim: one representative per generator of H_d(K; Z_p), in
    # every degree, checked against the betti numbers of the space rather than
    # against another run of the same code.
    H = SimpHomology(build(faces), p)

    assert tuple(len(reps_list(H.cycle_reps(d)))
                 for d in range(0, H.K.dim + 1)) == betti


@pytest.mark.parametrize("faces, p, d", REPS_DEGREES)
def testcycle_reps_count_agrees_with_d_homology(faces, p, d):
    # the same count, but against what d_homology reports for this complex, so
    # the two routes to dim H_d(K; Z_p) cannot drift apart even if the
    # hand-written table above is wrong.
    H = SimpHomology(build(faces), p)

    assert len(reps_list(H.cycle_reps(d))) == H.d_homology(d)[0]


@pytest.mark.parametrize("faces, p, d", REPS_DEGREES)
def testcycle_reps_are_cycles(faces, p, d):
    # a representative of a homology class has to be a cycle to begin with: a
    # d-chain, reduced mod p and not the zero one, that d_d kills.
    H = SimpHomology(build(faces), p)
    boundary = H.C.d_boundary_matrix(d)

    for rep in reps_list(H.cycle_reps(d)):
        assert len(rep) == H.C.d_rank(d)
        assert np.array_equal(rep, rep % p)
        assert np.any(rep % p)
        assert not np.any((boundary @ rep) % p)


@pytest.mark.parametrize("faces, p, d", REPS_DEGREES)
def testcycle_reps_are_independent_modulo_the_boundaries(faces, p, d):
    # being the right number of cycles is not enough -- they have to be
    # distinct, nonzero classes in Z_d / B_d. adjoining them to a spanning set
    # of B_d must therefore raise its rank by exactly their number, which
    # together with the count above makes them a basis of H_d(K; Z_p).
    H = SimpHomology(build(faces), p)
    B = boundaries_matrix(H, d)
    R = reps_matrix(H.cycle_reps(d), H.C.d_rank(d))

    if not R.shape[1]:
        return                                    # H_d(K; Z_p) is trivial

    if not B.shape[1]:
        assert rank_mod_p(R, p) == R.shape[1]     # nothing bounds in degree d
        return

    assert rank_mod_p(np.hstack([B, R]), p) == rank_mod_p(B, p) + R.shape[1]


@pytest.mark.parametrize("p", PRIMES)
def testcycle_reps_on_a_solid_simplex(p):
    # the solid 3-simplex is contractible: a single generator, the class of a
    # point, in degree 0 and nothing above it.
    H = SimpHomology(SimplicialComplex((0, 1, 2, 3)), p)

    assert tuple(len(reps_list(H.cycle_reps(d)))
                 for d in range(0, 4)) == (1, 0, 0, 0)


@pytest.mark.parametrize("p", PRIMES)
def testcycle_reps_on_a_disconnected_complex(p):
    # two disjoint solid triangles: H_0 = Z_p^2, one generator per component,
    # and both pieces are contractible, so nothing survives above degree 0.
    H = SimpHomology(SimplicialComplex((0, 1, 2), (3, 4, 5)), p)

    assert tuple(len(reps_list(H.cycle_reps(d)))
                 for d in range(0, 3)) == (2, 0, 0)


@pytest.mark.parametrize("d", [-1, 3])
def testcycle_reps_rejects_out_of_range_dimension(d):
    with pytest.raises(ValueError):
        SimpHomology(build(TORUS), 2).cycle_reps(d)


@pytest.mark.parametrize("p", [0, 1, 4])
def testcycle_reps_rejects_non_prime_coefficients(p):
    # the sieve inverts leading terms mod p, and the nullspace is taken with a
    # zero test that reads x % p; neither means anything unless Z_p is a field.
    with pytest.raises(ValueError):
        SimpHomology(SimplicialComplex((0, 1, 2)), p).cycle_reps(1)


# --------------------------------------------------------------------------
# cocycle_reps: the same construction one arrow the other way. the cocycles
# Z^d are the kernel of the coboundary map d^d, the coboundaries B^d are the
# image of d^(d-1), and there should be one representative per generator of
# H^d(K; Z_p). the coboundary matrices are the transposed boundary matrices,
# so d^d = d_coboundary_matrix(d) has the d-simplices as its columns and the
# (d + 1)-simplices as its rows.
# --------------------------------------------------------------------------


# the columns spanning the d-coboundaries B^d. nothing is a coboundary in
# degree 0, where there are no (d - 1)-cochains to take the coboundary of.
def coboundaries_matrix(H: SimpHomology, d: int) -> np.array:
    if d:
        return H.C.d_coboundary_matrix(d - 1)
    return np.zeros((H.C.d_rank(d), 0), dtype=int)


@pytest.mark.parametrize("faces, p, d", REPS_DEGREES)
def testcocycle_reps_count_is_the_cohomology_rank(faces, p, d):
    # the headline claim: one representative per generator of H^d(K; Z_p). over
    # a field the universal coefficient theorem makes H^d the dual of H_d, so
    # this is the mod p betti number again -- which the cycle tests above have
    # already pinned to the table at the top of the section.
    H = SimpHomology(build(faces), p)

    assert len(reps_list(H.cocycle_reps(d))) == H.d_cohomology(d)


@pytest.mark.parametrize("faces, p, d", REPS_DEGREES)
def testcocycle_reps_are_cocycles_independent_modulo_the_coboundaries(faces, p, d):
    # the right number of vectors is only meaningful if they are the right kind
    # of vector: d-cochains the coboundary map kills, spanning a subspace that
    # meets B^d only in zero. that plus the count makes them a basis of H^d.
    H = SimpHomology(build(faces), p)
    coboundary = H.C.d_coboundary_matrix(d)
    B = coboundaries_matrix(H, d)
    R = reps_matrix(H.cocycle_reps(d), H.C.d_rank(d))

    for rep in reps_list(H.cocycle_reps(d)):
        assert len(rep) == H.C.d_rank(d)
        assert np.array_equal(rep, rep % p)
        assert np.any(rep % p)
        assert not np.any((coboundary @ rep) % p)

    if not R.shape[1]:
        return                                    # H^d(K; Z_p) is trivial

    if not B.shape[1]:
        assert rank_mod_p(R, p) == R.shape[1]     # degree 0: B^d is trivial
        return

    assert rank_mod_p(np.hstack([B, R]), p) == rank_mod_p(B, p) + R.shape[1]


if __name__ == "__main__":
    #pytest.main([__file__])
    K = SimplicialComplex()
    for triangle in RP2:
        K.add_simplex(triangle)

    print(SimpHomology(K, 2).d_homology(1))