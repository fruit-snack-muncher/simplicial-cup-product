from simplicial_cup_product.chain_complex import ChainComplex
from simplicial_cup_product.simplicial_complex import SimplicialComplex
import numpy as np
from sympy import Matrix, ZZ, GF, isprime, symbols
from sympy.matrices.normalforms import smith_normal_form
from sympy.polys.matrices import DomainMatrix

class SimpHomology:
    def __init__(self, K: SimplicialComplex, p: int = 0):
        self.K = K
        self.C = ChainComplex(K)
        self.p = abs(p)

    def _p_prime(self):
        if not self.p or not isprime(self.p):
            raise ValueError('p in SimpHomology initialization must be a prime.')

    # the rank of a boundary matrix over the coefficient ring, together with the
    # elementary divisors of its SNF (the units dropped, as they contribute no torsion).
    def _rank_and_divisors(self, boundary_matrix, coeff_ring) -> tuple[int, tuple]:
        M = Matrix(boundary_matrix)

        if coeff_ring.is_Field:
            return DomainMatrix.from_Matrix(M).convert_to(coeff_ring).rank(), tuple()

        snf = smith_normal_form(M, domain=coeff_ring)
        # the diagonal of the SNF runs to min(rows, cols); units are not torsion.
        divisors = tuple(snf[i, i] for i in range(0, min(snf.shape)) if snf[i, i] not in {0, 1, -1})
        return snf.rank(), divisors

    # use the SNF on d-boundary matrices to compute homology over Z. broken into betti 
    # numbers and torsion numbers. uses sympy, as SNF is not implemented as a part of 
    # the numpy library. formula via Omar Antolin Camarena, "Using the Smith normal form
    # to compute homology" (https://www.matem.unam.mx/~omar/mathX27/smith-form.html)

    # ZZmod = 0 means integer coefficients; a positive prime p means coefficients in GF(p),
    # where the field has no torsion and the second entry is always empty.
    def d_homology(self, d: int) -> tuple[int, tuple]: # b_d, and torsion coefficients in a tuple. no torsion -> empty tuple.
        coeff_ring = ZZ
        if self.p:
            self._p_prime()
        
        C = ChainComplex(K)
        m = C.d_rank(d)

        r, _ = self._rank_and_divisors(C.d_boundary_matrix(d), coeff_ring)

        if d < self.K.dim:
            s, elem_divisors = self._rank_and_divisors(C.d_boundary_matrix(d+1), coeff_ring)

            return (m-r-s, elem_divisors)

        return (m-r, tuple())

    # by the UCT. rather uninteresting, as we are always working over Z_p. 
    # returns rank of cohomology group over field Z_p.
    def d_cohomology(self, d: int) -> tuple[int]:
        self._p_prime()

        return self.d_homology(d)[0]

    # finds the cokernel of a LT, via the nullspace of the transpose of its matrix. 
    # works since the orthogonal complement of a matrix (where the spanned space is
    # isomorphic to the cokernel) with the nullspace of A^T.
    # requires p prime.
    def _matrix_cokernel(self, M: Matrix) -> tuple:
        self._p_prime()

        M_T = DomainMatrix.from_Matrix(M).convert_to(GF(self.p))
        M_T = M_T.transpose()
        return M_T.nullspace()

    # finds explicit representatives for the d-homology group. requires working over Z_p.
    def _cycle_reps(self, d: int) -> tuple[int]:
        self._p_prime()

        





        


if __name__ == "__main__":
    K = SimplicialComplex((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), 
         (5, 6, 1), (6, 0, 2), (0, 2, 3), (1, 3, 4), (2, 4, 5), 
         (3, 5, 6), (4, 6, 0), (5, 0, 1), (6, 1, 2))
    C = ChainComplex(K)
    H = SimpHomology(K, 2)

    print(H._matrix_cokernel(Matrix(C.d_boundary_matrix(2))))

    
