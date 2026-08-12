from simplicial_cup_product.cup_product import CupProduct
from simplicial_cup_product.simplicial_complex import SimplicialComplex
import itertools
import numpy as np
import pytest

# the triangulations, their expected mod p betti numbers, and the two small
# helpers all live next door; there is no reason to keep a second copy of
# Kuehnel's CP^2 in this file. tests/ is a package, so this is an absolute
# import from the project root.
from tests.test_simp_homology import (SPHERE, TORUS, RP2, KLEIN, CP2,
                                      MOD_P_BETTI, build, rank_mod_p)


PRIMES = [2, 3, 5]

SURFACES = [("sphere", SPHERE), ("torus", TORUS),
            ("rp2", RP2), ("klein_bottle", KLEIN)]
COMPLEXES = SURFACES + [("cp2", CP2)]

# every (complex, prime) pair, for the tests that hold in any characteristic.
CUP_CASES = [
    pytest.param(name, faces, p, id=f"{name}-{p}")
    for name, faces in COMPLEXES for p in PRIMES
]

# the same for the surfaces alone, where Poincare duality pins the form on H^1.
SURFACE_CASES = [
    pytest.param(name, faces, id=name) for name, faces in SURFACES
]


def cup_product(faces, p) -> CupProduct:
    return CupProduct(build(faces), p)


# the cocycle representatives of H^d, ordered by leading term, so that index i
# in a coefficient tuple returned by cup_from_rep() refers to reps(cp, d)[i].
def reps(cp: CupProduct, d: int) -> list:
    return [cp.cocycle_reps[d][pivot] for pivot in sorted(cp.cocycle_reps[d])]


# cup_from_rep() answers with (target degree, coefficients in the basis of that
# degree). most of what follows only cares about the coefficients.
def coefficients(cp: CupProduct, d: int, phi, e: int, psi) -> tuple:
    return cp.cup_from_rep((d, phi), (e, psi))[1]


# the matrix of the cup product H^d x H^e -> H^{d+e} in those bases. entry
# [i][j][k] is the coefficient of the k-th generator of H^{d+e} in the product
# of the i-th generator of H^d with the j-th of H^e.
def cup_matrix(cp: CupProduct, d: int, e: int) -> list:
    return [[coefficients(cp, d, phi, e, psi) for psi in reps(cp, e)]
            for phi in reps(cp, d)]


# the form H^1 x H^1 -> H^2 on a closed surface, as a plain scalar matrix:
# H^2 is one dimensional over Z_p whenever H^1 is nonzero here, so each
# product is a one-entry tuple.
def surface_form(cp: CupProduct) -> np.array:
    return np.array([[entry[0] for entry in row]
                     for row in cup_matrix(cp, 1, 1)], dtype=int)


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_lands_in_the_right_degree(name, faces, p):
    # a product of classes in degrees d and e is a class in degree d + e:
    # cup_from_rep() reports that degree, and one coefficient per generator.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            for phi in reps(cp, d):
                for psi in reps(cp, e):
                    degree, coeffs = cp.cup_from_rep((d, phi), (e, psi))

                    assert degree == d + e
                    assert len(coeffs) == len(cp.cocycle_reps[d + e])


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_unit_acts_as_the_identity(name, faces, p):
    # every complex here is connected, so H^0 is spanned by the cocycle taking
    # the value 1 on each vertex -- the unit of the ring. cupping with it must
    # return each generator unchanged, i.e. the i-th standard basis vector.
    cp = cup_product(faces, p)
    unit = reps(cp, 0)[0]

    for d in range(0, cp.K.dim + 1):
        for i, phi in enumerate(reps(cp, d)):
            basis_vector = tuple(int(j == i) for j in range(len(reps(cp, d))))

            assert coefficients(cp, 0, unit, d, phi) == basis_vector
            assert coefficients(cp, d, phi, 0, unit) == basis_vector


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_above_the_top_dimension_vanishes(name, faces, p):
    # there are no simplices to evaluate on, so the product is zero. this exit
    # still reports its degree, but hands back an empty tuple: there is no
    # basis of H^{d+e} to give coefficients in.
    cp = cup_product(faces, p)
    pairs = [(d, e)
             for d in range(0, cp.K.dim + 1) for e in range(0, cp.K.dim + 1)
             if d + e > cp.K.dim and cp.cocycle_reps[d] and cp.cocycle_reps[e]]

    # vacuous where the cohomology above the middle vanishes, as it does for
    # the non-orientable surfaces mod odd p.
    for d, e in pairs:
        degree, coeffs = cp.cup_from_rep((d, reps(cp, d)[0]),
                                         (e, reps(cp, e)[0]))

        assert degree == d + e
        assert not np.any(np.asarray(coeffs, dtype=int))


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_of_a_zero_cochain_vanishes(name, faces, p):
    # the zero cochain represents the zero class, and cup_from_rep() sends it
    # straight to zero without asking whether it is one of the stored
    # representatives -- it is a coboundary, so no reduction is needed.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            zero = np.zeros(cp.C.d_rank(d), dtype=int)

            for psi in reps(cp, e):
                degree, coeffs = cp.cup_from_rep((d, zero), (e, psi))

                assert degree == d + e
                assert coeffs == (0, ) * len(cp.cocycle_reps[d + e])

                degree, coeffs = cp.cup_from_rep((e, psi), (d, zero))

                assert degree == d + e
                assert coeffs == (0, ) * len(cp.cocycle_reps[d + e])


# --------------------------------------------------------------------------
# the ring structure mod 2, where every closed surface has a fundamental class
# and the cup product on H^1 is the mod 2 intersection form.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, faces", SURFACE_CASES)
def test_cup_from_rep_form_on_a_surface_is_symmetric_mod_2(name, faces):
    # graded commutativity: phi cup psi = (-1)^(de) psi cup phi, and mod 2 the
    # sign is invisible, so the form on H^1 is symmetric.
    form = surface_form(cup_product(faces, 2))

    assert np.array_equal(form, form.T)


@pytest.mark.parametrize("name, faces", SURFACE_CASES)
def test_cup_from_rep_form_on_a_surface_is_nondegenerate_mod_2(name, faces):
    # Poincare duality over Z_2 holds for every closed surface, orientable or
    # not: the pairing H^1 x H^1 -> H^2 = Z_2 has no radical.
    cp = cup_product(faces, 2)
    form = surface_form(cp)

    if not form.size:
        return                                  # H^1 vanishes, as for S^2

    assert rank_mod_p(form, 2) == len(reps(cp, 1))


@pytest.mark.parametrize("name, faces, orientable", [
    pytest.param("sphere", SPHERE, True, id="sphere"),
    pytest.param("torus", TORUS, True, id="torus"),
    pytest.param("rp2", RP2, False, id="rp2"),
    pytest.param("klein_bottle", KLEIN, False, id="klein_bottle"),
])
def test_cup_from_rep_diagonal_detects_orientability_mod_2(name, faces,
                                                           orientable):
    # x -> x cup x is linear over Z_2, since the cross terms appear twice, so
    # "the form is even" does not depend on the basis. an even form is exactly
    # an orientable surface: on a non-orientable one some class has nonzero
    # self-intersection, coming from a cross-cap.
    form = surface_form(cup_product(faces, 2))
    diagonal = np.diag(form) if form.size else np.array([], dtype=int)

    assert (not np.any(diagonal)) == orientable


def test_cup_from_rep_square_generates_the_top_class_of_rp2_mod_2():
    # H^*(RP^2; Z_2) = Z_2[x]/(x^3): the square of the H^1 generator is the
    # fundamental class, the one product that makes mod 2 coefficients worth
    # the trouble here.
    cp = cup_product(RP2, 2)
    x = reps(cp, 1)[0]

    assert coefficients(cp, 1, x, 1, x) == (1,)


@pytest.mark.parametrize("p", PRIMES)
def test_cup_from_rep_square_generates_the_top_class_of_cp2(p):
    # H^*(CP^2; Z_p) = Z_p[x]/(x^3) with x in degree 2, in every
    # characteristic. the coefficient depends on how the generators of H^2 and
    # H^4 were chosen independently, so only its nonvanishing is meaningful.
    cp = cup_product(CP2, p)
    x = reps(cp, 2)[0]

    assert coefficients(cp, 2, x, 2, x) != (0,)


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_agrees_with_the_dimensions_of_the_cohomology(name, faces,
                                                                   p):
    # a consistency check tying the ring back to the groups: the number of
    # generators cup_from_rep() reports in each degree is the mod p betti
    # number.
    cp = cup_product(faces, p)

    assert tuple(len(reps(cp, d)) for d in range(0, cp.K.dim + 1)) \
        == MOD_P_BETTI[name](p)


# --------------------------------------------------------------------------
# graded commutativity in odd characteristic
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [3, 5, 7])
def test_cup_from_rep_is_graded_commutative_on_the_torus_mod_odd_p(p):
    # phi cup psi = (-1)^(de) psi cup phi. in degree 1 x 1 that is a genuine
    # sign, so on the torus a cup b and b cup a must be negatives of each
    # other -- a constraint that mod 2 cannot see, since 1 = -1 there. this is
    # what fails whenever the reduction against the coboundaries renormalizes
    # the running vector and loses its scale.
    cp = cup_product(TORUS, p)
    a, b = reps(cp, 1)

    assert coefficients(cp, 1, a, 1, b)[0] \
        == (-coefficients(cp, 1, b, 1, a)[0]) % p


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_from_rep_of_even_degree_classes_commutes(name, faces, p):
    # the sign (-1)^(de) is +1 whenever either degree is even, so these
    # products commute in every characteristic.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            if d % 2 and e % 2:
                continue
            for phi in reps(cp, d):
                for psi in reps(cp, e):
                    assert cp.cup_from_rep((d, phi), (e, psi)) \
                        == cp.cup_from_rep((e, psi), (d, phi))


# --------------------------------------------------------------------------
# rejected input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0, 1, 4, 9])
def test_cup_product_rejects_non_prime_coefficients(p):
    with pytest.raises(ValueError):
        CupProduct(build(RP2), p)


@pytest.mark.parametrize("d", [-1, 3])
def test_cup_from_rep_rejects_out_of_range_dimension(d):
    cp = cup_product(RP2, 2)
    unit = reps(cp, 0)[0]

    with pytest.raises(ValueError):
        cp.cup_from_rep((d, unit), (0, unit))
    with pytest.raises(ValueError):
        cp.cup_from_rep((0, unit), (d, unit))


@pytest.mark.parametrize("d", [0, 1, 2])
def test_cup_from_rep_rejects_a_chain_that_is_not_a_representative(d):
    # cup_from_rep() reads its arguments as generators of H^d and reports the
    # answer in that basis, so a cochain it does not recognize has to be turned
    # away. the stranger has to be nonzero to reach the check: the zero cochain
    # has its own exit, tested above.
    cp = cup_product(RP2, 2)
    unit = reps(cp, 0)[0]
    stranger = np.zeros(cp.C.d_rank(d), dtype=int)
    stranger[0] = 1
    assert not any(np.array_equal(stranger, rep) for rep in reps(cp, d))

    with pytest.raises(ValueError):
        cp.cup_from_rep((d, stranger), (0, unit))
    with pytest.raises(ValueError):
        cp.cup_from_rep((0, unit), (d, stranger))


# --------------------------------------------------------------------------
# cup(), which takes a class as its coefficients over the generators of H^d
# rather than as a literal cocycle, and expands the product bilinearly over
# cup_from_rep. everything above pins the generators; what is left to check
# is the expansion.
# --------------------------------------------------------------------------

# the standard basis of the coefficient space of H^d: basis(cp, d)[i] is the
# class of reps(cp, d)[i].
def basis(cp: CupProduct, d: int) -> list:
    n = len(cp.cocycle_reps[d])
    return [tuple(int(j == i) for j in range(n)) for i in range(n)]


# every class in H^d, as a coefficient tuple over Z_p. these spaces are tiny
# here -- 25 elements at the largest, on the torus mod 5 -- so the tests below
# can afford to quantify over all of them rather than sample.
def classes(cp: CupProduct, d: int) -> list:
    n = len(cp.cocycle_reps[d])
    return list(itertools.product(range(cp.p), repeat=n))


# phi cup psi worked out by hand from the products of the generators: the
# bilinear expansion sum_ij phi_i psi_j (x_i cup y_j), computed from the table
# cup_from_rep produces. this is what cup() has to agree with.
def expand(cp: CupProduct, table: list, d: int, e: int, phi, psi) -> tuple:
    total = np.zeros(len(cp.cocycle_reps[d + e]), dtype=int)

    for i, a in enumerate(phi):
        for j, b in enumerate(psi):
            total += a * b * np.asarray(table[i][j], dtype=int)

    return tuple(int(c) for c in total % cp.p)


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_agrees_with_cup_from_rep_on_the_generators(name, faces, p):
    # the i-th standard basis vector is the class of the i-th representative,
    # so on those the two entry points have to give the same answer. this is
    # what fixes the order of the coefficient tuple: sorted by pivot.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            for i, phi in enumerate(basis(cp, d)):
                for j, psi in enumerate(basis(cp, e)):
                    assert cp.cup((d, phi), (e, psi)) \
                        == cp.cup_from_rep((d, reps(cp, d)[i]),
                                           (e, reps(cp, e)[j]))


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_is_bilinear(name, faces, p):
    # the whole point of cup(): a product of arbitrary classes is the bilinear
    # expansion over the generators. checked against a table computed by
    # cup_from_rep, so the two ways of getting there stay independent.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            table = cup_matrix(cp, d, e)

            for phi in classes(cp, d):
                for psi in classes(cp, e):
                    assert cp.cup((d, phi), (e, psi))[1] \
                        == expand(cp, table, d, e, phi, psi)


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_distributes_over_addition_of_classes(name, faces, p):
    # bilinearity again, but stated the way it is used: adding classes before
    # multiplying is adding the products. cup() drops zero coefficients from
    # the expansion, so this also checks that a class and the same class with
    # a zero coordinate spliced in behave alike. the second argument only
    # ranges over the generators; the test above already quantifies over every
    # class on both sides.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            for phi, chi in itertools.product(classes(cp, d), repeat=2):
                total = tuple((a + b) % p for a, b in zip(phi, chi))

                for psi in basis(cp, e):
                    left = np.asarray(cp.cup((d, phi), (e, psi))[1], dtype=int)
                    right = np.asarray(cp.cup((d, chi), (e, psi))[1], dtype=int)

                    assert cp.cup((d, total), (e, psi))[1] \
                        == tuple(int(c) for c in (left + right) % p)


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_lands_in_the_right_degree(name, faces, p):
    # as for cup_from_rep: degree d + e, one coefficient per generator of it.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            for phi in classes(cp, d):
                for psi in classes(cp, e):
                    degree, coeffs = cp.cup((d, phi), (e, psi))

                    assert degree == d + e
                    assert len(coeffs) == len(cp.cocycle_reps[d + e])


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_unit_acts_as_the_identity(name, faces, p):
    # H^0 is spanned by the cocycle taking the value 1 on every vertex, so the
    # class (1,) is the unit of the ring: it returns every class unchanged,
    # not just the generators.
    cp = cup_product(faces, p)
    unit = basis(cp, 0)[0]

    for d in range(0, cp.K.dim + 1):
        for phi in classes(cp, d):
            assert cp.cup((0, unit), (d, phi))[1] == phi
            assert cp.cup((d, phi), (0, unit))[1] == phi


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_of_a_zero_class_vanishes(name, faces, p):
    # the zero class annihilates, and the early exit that says so has to give
    # back a coefficient tuple of the usual width rather than a bare zero.
    cp = cup_product(faces, p)

    for d in range(0, cp.K.dim + 1):
        for e in range(0, cp.K.dim + 1 - d):
            zero = (0, ) * len(cp.cocycle_reps[d])

            for psi in classes(cp, e):
                assert cp.cup((d, zero), (e, psi)) \
                    == (d + e, (0, ) * len(cp.cocycle_reps[d + e]))
                assert cp.cup((e, psi), (d, zero)) \
                    == (d + e, (0, ) * len(cp.cocycle_reps[d + e]))


@pytest.mark.parametrize("name, faces, p", CUP_CASES)
def test_cup_above_the_top_dimension_vanishes(name, faces, p):
    # no simplices to evaluate on, so the product is zero whatever the classes
    # were. the exit still reports its degree.
    cp = cup_product(faces, p)
    pairs = [(d, e)
             for d in range(0, cp.K.dim + 1) for e in range(0, cp.K.dim + 1)
             if d + e > cp.K.dim and cp.cocycle_reps[d] and cp.cocycle_reps[e]]

    for d, e in pairs:
        degree, coeffs = cp.cup((d, basis(cp, d)[0]), (e, basis(cp, e)[0]))

        assert degree == d + e
        assert not np.any(np.asarray(coeffs, dtype=int))


def test_cup_square_generates_the_top_class_of_rp2_mod_2():
    # H^*(RP^2; Z_2) = Z_2[x]/(x^3), read off through the coefficient API:
    # the only nonzero class of H^1 squares to the fundamental class.
    cp = cup_product(RP2, 2)

    assert cp.cup((1, (1,)), (1, (1,)))[1] == (1,)


@pytest.mark.parametrize("p", PRIMES)
def test_cup_square_generates_the_top_class_of_cp2(p):
    # H^*(CP^2; Z_p) = Z_p[x]/(x^3) with x in degree 2. squaring c*x scales the
    # top class by c^2, which is what a bilinear expansion has to produce.
    cp = cup_product(CP2, p)
    square = cp.cup((2, (1,)), (2, (1,)))[1][0]

    assert square

    for c in range(0, p):
        assert cp.cup((2, (c,)), (2, (c,)))[1] == ((c * c * square) % p, )


@pytest.mark.parametrize("p", [3, 5, 7])
def test_cup_is_graded_commutative_on_the_torus_mod_odd_p(p):
    # phi cup psi = -psi cup phi in degree 1 x 1. over the generators this is
    # already covered; here it is checked on every class of H^1, where the
    # cross terms of the expansion have to cancel correctly.
    cp = cup_product(TORUS, p)

    for phi in classes(cp, 1):
        for psi in classes(cp, 1):
            assert cp.cup((1, phi), (1, psi))[1][0] \
                == (-cp.cup((1, psi), (1, phi))[1][0]) % p


@pytest.mark.parametrize("p", [3, 5, 7])
def test_cup_square_vanishes_on_the_torus_mod_odd_p(p):
    # a consequence of the above that mod 2 cannot see: 2 (x cup x) = 0, and 2
    # is invertible, so every class of H^1 squares to zero.
    cp = cup_product(TORUS, p)

    for phi in classes(cp, 1):
        assert cp.cup((1, phi), (1, phi))[1] == (0,)


@pytest.mark.parametrize("d", [-1, 3])
def test_cup_rejects_out_of_range_dimension(d):
    cp = cup_product(RP2, 2)
    unit = basis(cp, 0)[0]

    with pytest.raises(ValueError):
        cp.cup((d, unit), (0, unit))
    with pytest.raises(ValueError):
        cp.cup((0, unit), (d, unit))


@pytest.mark.parametrize("d", [0, 1, 2])
def test_cup_rejects_a_coefficient_tuple_of_the_wrong_length(d):
    # a class is read against the generators of H^d, so a tuple that does not
    # have one entry per generator cannot be interpreted. every H^d of RP^2
    # mod 2 is one dimensional, so both of these are the wrong shape.
    cp = cup_product(RP2, 2)
    unit = basis(cp, 0)[0]

    for stranger in [(), (1, 1)]:
        with pytest.raises(ValueError):
            cp.cup((d, stranger), (0, unit))
        with pytest.raises(ValueError):
            cp.cup((0, unit), (d, stranger))
