from simplicial_cup_product.simp_homology import SimpHomology
from simplicial_cup_product.simplicial_complex import SimplicialComplex
import pytest
from sympy import GF


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

# H_d(K; Z) for d = 0, 1, 2, each as (betti number, torsion coefficients).
# All three surfaces are closed and connected, so H_0 = Z throughout, and
# H_2 = Z exactly for the orientable one.
#   torus: (Z, Z^2, Z)
#   RP^2:  (Z, Z/2, 0)      -- non-orientable, so H_2 vanishes
#   Klein: (Z, Z + Z/2, 0)  -- non-orientable, so H_2 vanishes
SURFACES = [
    pytest.param(TORUS, ((1, ()), (2, ()), (1, ())), id="torus"),
    pytest.param(RP2, ((1, ()), (0, (2,)), (0, ())), id="rp2"),
    pytest.param(KLEIN, ((1, ()), (1, (2,)), (0, ())), id="klein_bottle"),
]

SURFACES_MOD2 = [
    pytest.param(TORUS, ((1, ()), (2, ()), (1, ())), id="torus"),
    pytest.param(RP2, ((1, ()), (1, ()), (1, ())), id="rp2"),
    pytest.param(KLEIN, ((1, ()), (2, ()), (1, ())), id="klein_bottle"),
]

@pytest.mark.parametrize("triangles, homology_nums", SURFACES)
def test_homology_ZZ(triangles, homology_nums):
    K = SimplicialComplex(triangles)
    H = SimpHomology(K)

    assert tuple(H.d_homology(d) for d in range(0, 3)) == homology_nums

@pytest.mark.parametrize("triangles, homology_nums", SURFACES_MOD2)
def test_homology_Z2(triangles, homology_nums):
    K = SimplicialComplex(triangles)
    H = SimpHomology(K)

    assert tuple(H.d_homology(K, d, 2) for d in range(0, 3)) == homology_nums

@pytest.mark.parametrize("triangles, homology_nums", SURFACES_MOD2)
def test_cohomology_Z2(triangles, homology_nums):
    K = SimplicialComplex(triangles)
    H = SimpHomology(K)

    assert tuple(H.d_cohomology(K, d, 2) for d in range(0, 3)) == tuple(rank for rank, torsion in homology_nums)

@pytest.mark.parametrize("triangles",
                         [pytest.param(p.values[0], id=p.id) for p in SURFACES])
def test_homology_ZZ_euler_characteristic(triangles):
    # the Euler characteristic is the alternating sum of the betti numbers,
    # and for a closed surface it agrees with V - E + F.
    K = SimplicialComplex(triangles)
    H = SimpHomology(K)

    betti = [H.d_homology(K, d)[0] for d in range(0, 3)]
    faces = [len(K.sorted_simplices[d]) for d in range(0, 3)]

    assert sum((-1)**d * b for d, b in enumerate(betti)) \
        == sum((-1)**d * f for d, f in enumerate(faces))


@pytest.mark.parametrize("d", [-1, 3])
def test_homology_ZZ_rejects_out_of_range_dimension(d):
    K = SimplicialComplex()
    for triangle in TORUS:
        K.add_simplex(triangle)
    H = SimpHomology(K)

    with pytest.raises(ValueError):
        SimpHomology(K).d_homology(K, d)


if __name__ == "__main__":
    #pytest.main([__file__])
    K = SimplicialComplex()
    for triangle in RP2:
        K.add_simplex(triangle)

    print(SimpHomology(K, 2).d_homology(1))