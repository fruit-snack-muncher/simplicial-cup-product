from simplicial_cup_product.simplicial_complex import SimplicialComplex
from simplicial_cup_product.chain_complex import ChainComplex
from simplicial_cup_product.simp_homology import SimpHomology
from sympy import Matrix, GF, isprime
import numpy as np


class CupProduct():
    # implicit to be a connected simplicial complex. will bug if not!
    # do not add simplices to the complex once initialized. create a separate
    # CupProduct instance for a new complex.
    def __init__(self, K: SimplicialComplex, p: int = 2):
        self.p = abs(p)
        self._p_prime()

        self.K = K
        self.C = ChainComplex(K)
        self.H = SimpHomology(K, self.p)

        self._coboundary_sieves = {d: self.H._sieve_basis(d, cocycles=True) for d in range(0, self.K.dim)}
        self._face_maps = {d: {face: i for i, face in enumerate(self.K.sorted_simplices[d])} for d in range(0, self.K.dim+1)}
        self.cocycle_reps = {d: self.H.cocycle_reps(d) for d in range(0, self.K.dim + 1)}
        self.sorted_cocycle_reps = {d: tuple( self.cocycle_reps[d][key] for key in sorted(self.cocycle_reps[d]) )  
                                    for d in range(0, self.K.dim+1)}
        self.non_trivial_dim = frozenset(d for d, reps in self.cocycle_reps.items() if reps)

        # filled on demand by _gather, _reducer and _product_block: the per-bidegree
        # face lookups, the per-degree echelon basis, and the products of the
        # generators, all of which are reused by every product of that shape.
        self._gathers, self._reducers, self._products = {}, {}, {}

    def _p_prime(self):
        if not isprime(self.p):
            raise ValueError('p must be a prime number.')

    def _gather(self, d: int, e: int) -> tuple[np.array, np.array]:
        """The front and back face indices of every (d+e)-simplex.

        A cup product reads phi off the front d-face and psi off the back e-face of
        each simplex. Which faces those are depends only on the bidegree, not on the
        cochains, so the lookups are done once here and every later product in the
        block is a single gather.
        """
        if (d, e) not in self._gathers:
            simplices = self.K.sorted_simplices[d + e]
            n = len(simplices)

            self._gathers[(d, e)] = (
                np.fromiter((self._face_maps[d][simplex[:d + 1]] for simplex in simplices),
                            dtype=np.int64, count=n),
                np.fromiter((self._face_maps[e][simplex[d:]] for simplex in simplices),
                            dtype=np.int64, count=n))

        return self._gathers[(d, e)]

    def _reducer(self, d: int) -> tuple[dict, dict, int]:
        """The echelon basis of the d-cocycles, keyed by pivot, with the rep columns.

        The coboundaries and the cocycle representatives together span Z^d, each with
        a leading term of its own, so any cocycle reduces against them uniquely. The
        second entry says which pivots are representatives, and where their
        coefficients belong in the answer.
        """
        if d not in self._reducers:
            sieve = self._coboundary_sieves[d - 1] if d >= 1 else {}
            reps = self.cocycle_reps[d]
            pivots = {pivot: np.asarray(vector).astype(int)
                      for pivot, vector in {**sieve, **reps}.items()}

            self._reducers[d] = (pivots, {pivot: column for column, pivot in enumerate(sorted(reps))},
                                 len(reps))

        return self._reducers[d]

    def _rep_index(self, d: int, cochain: np.array) -> int:
        """Which generator of H^d this cochain is, or None if it is not one of them."""
        for i, rep in enumerate(self.sorted_cocycle_reps[d]):
            if np.array_equal(cochain, rep):
                return i

        return None

    def _cup_indices(self, d: int, i: int, e: int, j: int) -> tuple:
        """The coefficients of the i-th generator of H^d cup the j-th of H^e.

        Unchecked: d + e must not exceed the dimension and both indices must be in
        range. cup_from_rep is the way in from outside.
        """
        p = self.p
        front, back = self._gather(d, e)
        phi = np.asarray(self.sorted_cocycle_reps[d][i]).astype(int)
        psi = np.asarray(self.sorted_cocycle_reps[e][j]).astype(int)
        residue = (phi[front] * psi[back]) % p

        pivots, columns, width = self._reducer(d + e)
        coeffs = [0] * width

        # in an echelon basis the leading term of anything in the span is itself a
        # pivot, so this visits only the pivots that actually turn up - a handful,
        # where a sweep over all of them would run the length of the cochain. a
        # product of cocycles is a cocycle, hence always in the span.
        nonzero = np.flatnonzero(residue)
        while nonzero.size:
            pivot = int(nonzero[0])
            c = int(residue[pivot])

            residue = (residue - c * pivots[pivot]) % p
            if pivot in columns:
                coeffs[columns[pivot]] = c

            nonzero = np.flatnonzero(residue)

        return tuple(coeffs)

    # cup will produce an error if the inputs are not dimension less than or equal to the 
    # dimension of the complex!
    def cup_from_rep(self, PHI: tuple[int, np.array], PSI: tuple[int, np.array]) -> tuple:
        """The cup product of two classes, each given as (degree, cocycle representative).

        Returns the degree of the product and its coefficients in the basis of that
        degree, ordered by pivot. Each representative must be one of self.cocycle_reps.
        """
        phiD, phi = PHI
        psiD, psi = PSI

        if not {phiD, psiD}.issubset(set([i for i in range(0, self.K.dim+1)])):
            raise ValueError('Dimension error on cocycle representatives.')

        # phiD + psiD exceeds the dimension of our complex, and will output
        # a zero class.
        if phiD + psiD > self.K.dim:
            return phiD + psiD, (0, )
        
        # sends any cycle that is all zeros straight to zero.
        # represents multiplication by a boundary; placed early to avoid catch in following lines.
        if not any(phi) or not any(psi):
            return phiD + psiD, (0, ) * len( self.cocycle_reps[phiD+psiD] )

        # locating each representative is the same scan that checks it is one, so
        # the index it yields is what gets handed on.
        i, j = self._rep_index(phiD, phi), self._rep_index(psiD, psi)

        if i is None:
            raise ValueError('No cocycle phi in given dimension.')
        if j is None:
            raise ValueError('No cocycle psi in given dimension.')

        # the coefficients of the product in the basis of H^{phiD + psiD}, in
        # the same order as sorted(self.cocycle_reps[phiD + psiD]).
        return phiD + psiD, self._cup_indices(phiD, i, psiD, j)

    # (int, tuple) : int represents dimension of the class.
    #              : tuple represents a Z_p-linear combination of the cocycle reps,
    #                sorted according to ordering on their pivots - guaranteed to be
    #                unambiguous as pivots are unique per design.
    # cup will produce an error if the inputs are not less than or equal to the 
    # dimension of the complex!
    def cup(self, PHI: tuple[int, tuple], PSI: tuple[int, tuple]) -> tuple[int, tuple]:
        phiD, phi = PHI
        psiD, psi = PSI
        
        if not {phiD, psiD}.issubset( set([i for i in range(0, self.K.dim+1)]) ):
            raise ValueError('Dimension error on cocycle representatives.')
        
        if not len(phi) == len(self.cocycle_reps[phiD]):
            raise ValueError('Length of phi out of bounds.')
        if not len(psi) == len(self.cocycle_reps[psiD]):
            raise ValueError('Length of psi out of bounds.')

        # phiD + psiD exceeds the dimension of our complex, and will output
        # a zero class.
        if phiD + psiD > self.K.dim:
            return phiD + psiD, (0, )
                
        # sends any cycle that is all zeros straight to zero.
        if not any(phi) or not any(psi):
            return phiD + psiD, (0, ) * len( self.cocycle_reps[phiD+psiD] )

        # the products of the generators are fixed, so the expansion is arithmetic
        # on the cached block: sum_ij phi_i psi_j (x_i cup y_j), one coefficient per
        # generator of H^{phiD + psiD}. no simplex is touched here.
        block = self._product_block(phiD, psiD)
        final_product = np.einsum('i,j,ijk->k', np.array(phi).astype(int),
                                  np.array(psi).astype(int), block)

        return phiD+psiD, tuple( map(int, final_product % self.p) )

    def _product_block(self, d: int, e: int) -> np.array:
        """The products of the generators of H^d with those of H^e, as one array.

        Entry [i][j] holds the coefficients of the i-th generator of H^d cup the
        j-th of H^e, in the basis of H^{d+e}. Only d <= e is ever multiplied out:
        graded commutativity, psi cup phi = (-1)^{de} phi cup psi, makes the
        opposite block a signed transpose, and mod 2 the sign is invisible.
        """
        if (d, e) not in self._products:
            heights = (len(self.cocycle_reps[d]), len(self.cocycle_reps[e]),
                       len(self.cocycle_reps[d + e]))

            if d <= e:
                block = np.array([[self._cup_indices(d, i, e, j) for j in range(0, heights[1])]
                                  for i in range(0, heights[0])]).astype(int).reshape(heights)
            else:
                sign = -1 if (d * e) % 2 else 1
                block = (sign * np.swapaxes(self._product_block(e, d), 0, 1)) % self.p

            self._products[(d, e)] = block

        return self._products[(d, e)]

    def _cache_products(self) -> dict:
        """Every bidegree's product block, keyed by (d, e). Filled once, then reused."""
        for d in range(0, self.K.dim + 1):
            for e in range(0, self.K.dim + 1 - d):
                self._product_block(d, e)

        return self._products

    def cohomology_ring(self):
        reached_cocycle_reps = self.cocycle_reps.copy()

        for d in range(0, self.K.dim + 1):
            if d not in self.non_trivial_dim:
                continue

        #implmeent from here.