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
            coeff_ring = GF(self.p)
        
        C = ChainComplex(self.K)
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

    # returns the multiplicative inverse of x, mod self.p
    def _rem(self, x:int) -> int:
        if x % self.p == 0:
            pass
        for i in range(1, self.p):
            if (i * x) % self.p == 1:
                return i

    # obtains an np.array, finds the first nonzero coordinate, scales the
    # array to normalize the first nonzero position, and returns the scaled array
    # and the index of the first nonzero position.
    # returns (-2, n) if n is all zeros... due to implementation using an iterator.
    def _scale(self, n: np.array) -> tuple[int, np.array]:
        mod_n = np.mod(n, self.p)
        if not np.any(mod_n):
            return (-2, mod_n) # :3

        lt_idx = int( np.nonzero(n)[0][0] )
        lt = int( n[lt_idx] )
        lt_inv = self._rem(lt)
        return lt_idx, np.mod(lt_inv * n, self.p)
    
    # returns a basis for the image of the boundary matrix as a dict, where the leading
    # term of each basis vector is one. 
    # keys are the indices of the leading terms, and values are the associated vector.
    def _sieve_basis(self, d: int, cocycles: bool = False) -> dict:
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

            basis[idx] = scaled

        return {key : value for key, value in basis.items() if key >= 0}

    # given a type of dictionary as produced by _image_basis -- a dictionary
    # with (pivot index, pivot vector) where pivots are unique -- runs a vector 
    # through the dictionary vectors in order of the position of their LT, and 
    # updates the dictionary only if the vector survives.
    def _cycle_sieve(self, cycles: dict, cycle: np.array) -> dict:
        idx, scaled = self._scale(cycle)
        pivots = sorted(cycles.keys())
        iter = 0

        while iter >= 0 and idx in pivots and iter<len(pivots):
            pivot = pivots[iter]
            if scaled[pivot] != 0:
                idx, scaled = self._scale(scaled - cycles[pivot])
            iter += 1

        if idx >= 0:
            cycles[idx] = scaled

        return cycles

    
    # finds explicit cycle representatives for the d-homology group. requires working over Z_p.
    # keys are the indices of the leading terms, as in _sieve_basis; there is one entry per
    # generator of H_d(K; Z_p).
    def cycle_reps(self, d: int) -> dict:
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

    # finds explicit cocycle representatives for the d-homology group. requires working over Z_p.
    # keys are the indices of the leading terms, as in _sieve_basis; there is one entry per
    # generator of H^d(K; Z_p).
    def cocycle_reps(self, d: int) -> dict:
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