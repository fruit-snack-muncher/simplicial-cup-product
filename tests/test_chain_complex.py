from simplicial_cup_product.simplicial_complex import SimplicialComplex
from simplicial_cup_product.chain_complex import ChainComplex
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Closed surfaces: canonical triangulations of the torus, RP^2 and the Klein
# bottle. Each is given by its list of top-dimensional (2-)simplices; the
# faces are filled in by SimplicialComplex.
# ---------------------------------------------------------------------------

# Moebius' 7-vertex torus, the vertex-minimal triangulation: the triangles
# {i, i+1, i+3} and {i, i+2, i+3} taken mod 7. V - E + F = 7 - 21 + 14 = 0.
TORUS = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), 
         (5, 6, 1), (6, 0, 2), (0, 2, 3), (1, 3, 4), (2, 4, 5), 
         (3, 5, 6), (4, 6, 0), (5, 0, 1), (6, 1, 2)]

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

SURFACES = [
    pytest.param(TORUS, (7, 21, 14), id="torus"),
    pytest.param(RP2, (6, 15, 10), id="rp2"),
    pytest.param(KLEIN, (9, 27, 18), id="klein_bottle"),
]

@pytest.mark.parametrize("triangles, expected_ranks", SURFACES)
def test_surface_chain_ranks(triangles, expected_ranks):
    K = SimplicialComplex()
    K.add_simplex(*triangles)
    C = ChainComplex(K)
    assert tuple(C.chain_ranks()) == expected_ranks

@pytest.mark.parametrize("triangles, expected_ranks", SURFACES)
def test_surface_d_boundary_matrix_shapes(triangles, expected_ranks):
    K = SimplicialComplex()
    K.add_simplex(*triangles)
    C = ChainComplex(K)
    n_0, n_1, n_2 = expected_ranks

    assert C.d_boundary_matrix(1).shape == (n_0, n_1)
    assert C.d_boundary_matrix(2).shape == (n_1, n_2)

    # every d-simplex has exactly d + 1 faces, each appearing with sign +-1.
    assert (np.abs(C.d_boundary_matrix(1)).sum(axis=0) == 2).all()
    assert (np.abs(C.d_boundary_matrix(2)).sum(axis=0) == 3).all()

    # every edge of a closed surface lies in exactly two triangles.
    assert (np.abs(C.d_boundary_matrix(2)).sum(axis=1) == 2).all()

@pytest.mark.parametrize("triangles, expected_ranks", SURFACES)
def test_surface_d_boundary_composition_vanishes(triangles, expected_ranks):
    K = SimplicialComplex()
    K.add_simplex(*triangles)
    C = ChainComplex(K)
    n_0, n_1, n_2 = expected_ranks

    # boundary_{d} boundary_{d+1} = 0 for each consecutive pair.
    assert np.array_equal(C.d_boundary_matrix(0) @ C.d_boundary_matrix(1),
                          np.zeros((n_0, n_1), dtype=int))
    assert np.array_equal(C.d_boundary_matrix(1) @ C.d_boundary_matrix(2),
                          np.zeros((n_0, n_2), dtype=int))

@pytest.mark.parametrize("triangles, expected_ranks", SURFACES)
def test_surface_boundary_matrix_squared_vanishes(triangles, expected_ranks):
    K = SimplicialComplex()
    K.add_simplex(*triangles)
    C = ChainComplex(K)
    B = C.boundary_matrix()

    n = sum(expected_ranks)
    assert B.shape == (n, n)
    assert np.array_equal(B @ B, np.zeros((n, n), dtype=int))


@pytest.mark.parametrize("triangles, expected_ranks", SURFACES)
def test_surface_d_coboundary_matrix_shapes(triangles, expected_ranks):
    K = SimplicialComplex()
    K.add_simplex(*triangles)
    C = ChainComplex(K)
    n_0, n_1, n_2 = expected_ranks

    assert C.d_coboundary_matrix(0).shape == (n_1, n_0)
    assert C.d_coboundary_matrix(1).shape == (n_2, n_1)

    assert C.coboundary_matrix().shape == (sum(expected_ranks), sum(expected_ranks))

if __name__ == "__main__":
    pass
