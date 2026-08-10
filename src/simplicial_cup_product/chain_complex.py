from simplicial_cup_product.simplicial_complex import SimplicialComplex
import numpy as np
from sympy import Matrix

class ChainComplex:
    def __init__(self, K: SimplicialComplex):
        self.K = K

    def _dim_check(self, d: int):
        if not d in range(self.K.dim + 1):
            raise ValueError('d must be a non-negative integer at most the dimension of the complex.')

        
    # returns the boundary of any d-simplex as a numpy array, where indices
    # of the array correspond to the sorted indices of the d-1 simplices,
    # and nonzero entries represent the signs of each d-1 simplex in the
    # boundary as a d-1 chain.
    def simplex_boundary(self, s: tuple) -> np.array:
        d = len(s) - 1
        if d < 0:
            return ValueError("Dimension of the simplex must be at least 0.")

        if d == 0:
            return np.zeros(len(self.K.sorted_simplices[0]))

        boundary_simplices = {s[:i] + s[i+1:]: (-1)**i  for i in range(len(s))} # simplex and sign
        boundary_array = [boundary_simplices[simplex] if simplex in boundary_simplices else 0  for simplex in self.K.sorted_simplices[d-1] ]

        return np.array(boundary_array)

    # rank of C_d for particular non-negative d at most the dimension of the complex.
    def d_rank(self, d: int) -> int:
        self._dim_check(d)
        return len(self.K.sorted_simplices[d])

    # rank of C_d for each non-negative d at most the dimension of the complex, where index = d.
    def chain_ranks(self) -> np.array:
        d_rank = [len(self.K.sorted_simplices[d]) for d in sorted(self.K.sorted_simplices.keys())]
        return np.array(d_rank)

    # Euler characteristic
    def euler_char(self) -> int:
        return sum([(-1)**i * rank for i, rank in enumerate(self.chain_ranks)])

    # boundary matrix of d-simplices, where the columns represent d-simplices, rows represent d-1 simplices,
    # columns represent the boundary d-1 chains of each d-simplex.
    # all simplices are sorted according to the lexicographic ordering on their vertices in sorted_simplices,
    # where rightward columns and downward rows are higher in the order.
    def d_boundary_matrix(self, d: int = 0) -> np.array:
        self._dim_check(d)
    
        # square 0 matrix for d = 0; only used to verify boundary boundary = 0 for all boundary maps.
        if d == 0:
            return np.zeros( (len(self.K.sorted_simplices[0]), len(self.K.sorted_simplices[0])), dtype=int )

        boundary_matrix = np.array([self.simplex_boundary(s) for s in self.K.sorted_simplices[d]], dtype=int)
        return boundary_matrix.T

    def boundary_matrix(self) -> np.array:
        ranks = self.chain_ranks()

        boundary_matrix = np.zeros( (sum(ranks), sum(ranks)),  dtype=int) # 0 matrix; to be filled in.

        for d in range(1, len(ranks)): # since ranks include C_0
            row1, col1 = sum(ranks[:d-1]), sum(ranks[:d])
            boundary_matrix[row1 : row1 + ranks[d-1] , col1 : col1 + ranks[d]] = self.d_boundary_matrix(d)

        return boundary_matrix

    # coboundary matrix of d-cochains.
    # the columns represent the dual d-cochain corresponding to each d-simplex S, where phi sends S to 1 and all 
    # other simplices to 0. the rows represent d+1-simplices. each column reveals where a particular d-simplex
    # appeared in the boundary of a d+1-simplex, encoding orientation information as +/-.
    def d_coboundary_matrix(self, d: int = 0) -> np.array:
        self._dim_check(d)

        # square 0 matrix for d = K.dim; only used to verify boundary boundary = 0 for all boundary maps.
        if d == self.K.dim:
            return np.zeros( (len(self.K.sorted_simplices[d]), len(self.K.sorted_simplices[d])), dtype=int )

        # the d coboundary matrix is the transpose of the d+1 boundary matrix.
        coboundary_matrix = np.array([self.simplex_boundary(s) for s in self.K.sorted_simplices[d+1]], dtype=int)
        return coboundary_matrix

    def coboundary_matrix(self) -> np.array:
        ranks = self.chain_ranks()

        coboundary_matrix = np.zeros( (sum(ranks), sum(ranks)),  dtype=int) # 0 matrix; to be filled in.

        for d in range(0, len(ranks)-1): # since ranks include C_dim
            col1, row1 = sum(ranks[:d]), sum(ranks[:d+1])
            coboundary_matrix[row1 : row1 + ranks[d+1] , col1 : col1 + ranks[d]] = self.d_coboundary_matrix(d)

        return coboundary_matrix


if __name__ == "__main__":
    KLEIN = SimplicialComplex((0, 1, 4), (0, 1, 8), (0, 2, 3), (0, 2, 6), (0, 3, 4), (0, 6, 8),
         (1, 2, 5), (1, 2, 7), (1, 4, 5), (1, 7, 8), (2, 3, 5), (2, 6, 7),
         (3, 4, 7), (3, 5, 6), (3, 6, 7), (4, 5, 8), (4, 7, 8), (5, 6, 8))
    K = ChainComplex(KLEIN)
    