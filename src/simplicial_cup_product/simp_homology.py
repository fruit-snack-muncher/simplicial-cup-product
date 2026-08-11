from simplicial_cup_product.chain_complex import ChainComplex
from simplicial_cup_product.simplicial_complex import SimplicialComplex
import numpy as np
import itertools
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

    def _rank_and_divisors(self, boundary_matrix, coeff_ring) -> tuple[int, tuple]:
        """The rank over coeff_ring, with the torsion elementary divisors of the SNF.

        Units are dropped, contributing no torsion; over a field there is none at all and
        the second entry is empty.
        """
        M = Matrix(boundary_matrix)

        if coeff_ring.is_Field:
            return DomainMatrix.from_Matrix(M).convert_to(coeff_ring).rank(), tuple()

        snf = smith_normal_form(M, domain=coeff_ring)
        # the diagonal of the SNF runs to min(rows, cols); units are not torsion.
        divisors = tuple(snf[i, i] for i in range(0, min(snf.shape)) if snf[i, i] not in {0, 1, -1})
        return snf.rank(), divisors

    def d_homology(self, d: int) -> tuple[int, tuple]:
        """H_d(K) as (betti number, torsion coefficients), over Z or over GF(p).

        Both are read off the Smith normal forms of the d- and (d+1)-boundary matrices,
        following Omar Antolin Camarena, "Using the Smith normal form to compute
        homology". SymPy does the work, as NumPy has no SNF. p = 0 means integer
        coefficients; over GF(p) there is no torsion, so the tuple is always empty.
        """
        coeff_ring = ZZ
        if self.p:
            self._p_prime()
            coeff_ring = GF(self.p)
        
        C = ChainComplex(self.K)
        m = C.d_rank(d)

        r, _ = self._rank_and_divisors(C.d_boundary_matrix(d), coeff_ring)

        if d < self.K.dim:
            s, elem_divisors = self._rank_and_divisors(C.d_boundary_matrix(d+1), coeff_ring)

            return (m-r-s, elem_divisors)

        return (m-r, tuple())

    def d_cohomology(self, d: int) -> tuple[int]:
        """dim H^d(K; Z_p), which the UCT makes equal to dim H_d over a field."""
        self._p_prime()

        return self.d_homology(d)[0]

    def _rem(self, x:int) -> int:
        """The multiplicative inverse of x mod self.p."""
        if x % self.p == 0:
            pass
        for i in range(1, self.p):
            if (i * x) % self.p == 1:
                return i

    def _scale(self, n: np.array, factor_only = False) -> tuple[int, np.array]:
        """Rescale n mod p so that its leading (first nonzero) entry is 1.

        Returns (index of that entry, rescaled vector), or the leading coefficient alone
        if factor_only. An all-zero vector has no leading term: it yields (-2, zeros), or
        0 under factor_only.
        """
        mod_n = np.mod(n, self.p)
        if not np.any(mod_n):
            if factor_only:
                return 0
            return (-2, mod_n) # :3

        lt_idx = int( np.nonzero(n)[0][0] )
        lt = int( n[lt_idx] )
        lt_inv = self._rem(lt)
        if factor_only:
            return lt
        
        return lt_idx, np.mod(lt_inv * n, self.p)
    
    def _sieve_basis(self, d: int, cocycles: bool = False) -> dict:
        """An echelon basis for the image of the d-boundary map, or of the d-coboundary.

        Keyed by the index of each vector's leading term, which is normalized to 1;
        distinct keys are what make the vectors independent. A column reducing to zero is
        already spanned, and contributes nothing.
        """
        self._p_prime()

        if cocycles:
            M = self.C.d_coboundary_matrix(d)
        else:
            M = self.C.d_boundary_matrix(d)

        boundaries = tuple( boundary for boundary in M.transpose() )
        basis = {}

        for boundary in boundaries:
            idx, scaled = self._scale(boundary)

            # add elements: passes the first boundary in for free.
            if not basis:
                basis[idx] = scaled
                continue

            # runs each boundary against a sieve of boundary vectors, with at most corresponding
            # to each possible leading term position. we run the boundary by removing leading
            # terms from left to right.
            pivots = sorted(basis.keys())
            iter = 0

            while iter >= 0 and idx in pivots and iter < len(pivots):
                pivot = pivots[iter]
                if scaled[pivot] != 0:
                    idx, scaled = self._scale(scaled - basis[pivot])
                iter += 1

            if idx < 0:
                continue

            basis[idx] = scaled

        return {key : value for key, value in basis.items() if key >= 0}

    def _cycle_sieve(self, cycles: dict, cycle: np.array) -> dict:
        """Reduce cycle against the echelon basis `cycles`, adding it if it survives.

        Leading terms are cleared left to right. A vector reducing to zero is already
        spanned and is dropped; otherwise its new leading term becomes a fresh pivot.
        Mutates `cycles` and returns it.
        """
        idx, scaled = self._scale(cycle)
        pivots = sorted(cycles.keys())
        iter = 0
    
        while iter >= 0 and idx in pivots and iter<len(pivots):
            pivot = pivots[iter]
            if scaled[pivot] != 0:
                cleared = scaled - cycles[pivot]
                idx, scaled = self._scale(cleared)
            iter += 1

        if idx >= 0:
            cycles[idx] = scaled

        return cycles
    
    def cycle_reps(self, d: int) -> dict:
        """Representatives for a basis of H_d(K; Z_p), keyed by leading term.

        A basis of the cycles Z_d is sieved against the boundaries B_d; those surviving
        with a leading term of their own generate the quotient, one entry per generator.
        """
        self._p_prime()

        sieve = self._sieve_basis(d+1) if d < self.K.dim else {}
        pivots = frozenset(sieve)

        # a basis for the cycles Z_d, taken over Z_p. 
        boundary_M = DomainMatrix.from_Matrix(Matrix(self.C.d_boundary_matrix(d))).convert_to(GF(self.p))
        cycle_basis = tuple( np.array([int(entry) % self.p for entry in row]) for row in boundary_M.nullspace().to_list() )
        cycle_basis = tuple( self._scale(cycle)[1] for cycle in cycle_basis )

        for cycle in cycle_basis:
            sieve = self._cycle_sieve(sieve, cycle)

        # boundaries already have taken unique pivots.
        cycle_reps = {key: value for key, value in sieve.items() if key not in pivots}

        return cycle_reps

    def cocycle_reps(self, d: int) -> dict:
        """Representatives for a basis of H^d(K; Z_p), keyed by leading term.

        The same construction as cycle_reps with the arrows reversed: the cocycles Z^d
        are sieved against the coboundaries B^d.
        """
        self._p_prime()

        sieve = self._sieve_basis(d-1, cocycles=True) if d > 0 else {}
        pivots = frozenset(sieve)

        # a basis for the cycles Z_d, taken over Z_p. 
        boundary_M = DomainMatrix.from_Matrix(Matrix(self.C.d_coboundary_matrix(d))).convert_to(GF(self.p))
        cycle_basis = tuple( np.array([int(entry) % self.p for entry in row]) for row in boundary_M.nullspace().to_list() )
        cycle_basis = tuple( self._scale(cycle)[1] for cycle in cycle_basis )

        for cycle in cycle_basis:
            sieve = self._cycle_sieve(sieve, cycle)

        # boundaries already have taken unique pivots.
        cycle_reps = {key: value for key, value in sieve.items() if key not in pivots}

        return cycle_reps


if __name__ == "__main__":
    K = SimplicialComplex((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), 
         (5, 6, 1), (6, 0, 2), (0, 2, 3), (1, 3, 4), (2, 4, 5), 
         (3, 5, 6), (4, 6, 0), (5, 0, 1), (6, 1, 2))
    KK = SimplicialComplex((1, 2, 5), (1, 2, 6), (1, 3, 4), (1, 3, 6), (1, 4, 5),
       (2, 3, 4), (2, 3, 5), (2, 4, 6), (3, 5, 6), (4, 5, 6))
    C = ChainComplex(K)
    H = SimpHomology(K, 5)
    CC = ChainComplex(KK)
    HH = SimpHomology(KK, 2)
    M = np.array([[1,2,3],
                  [1,2,3],
                  [1,2,3]])

    test = SimplicialComplex((0,1,2,3))

    testH = SimpHomology(test, 3)

    print(testH._cycle_reps(2))