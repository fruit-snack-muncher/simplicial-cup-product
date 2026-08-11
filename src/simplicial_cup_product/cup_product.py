from simplicial_cup_product.simplicial_complex import SimplicialComplex
from simplicial_cup_product.chain_complex import ChainComplex
from simplicial_cup_product.simp_homology import SimpHomology
from sympy import Matrix, GF, isprime
import numpy as np


class CupProduct():
    # implicit to be a connected simplicial complex. will bug if not!
    def __init__(self, K: SimplicialComplex, p: int = 2):
        self.p = abs(p)
        self._p_prime()

        self.K = K
        self.C = ChainComplex(K)
        self.H = SimpHomology(K, self.p)

        self._coboundary_sieves = {d: self.H._sieve_basis(d, cocycles=True) for d in range(0, self.K.dim)}
        self._face_maps = {d: {face: i for i, face in enumerate(self.K.sorted_simplices[d])} for d in range(0, self.K.dim+1)}
        self.cocycle_reps = {d: self.H.cocycle_reps(d) for d in range(0, self.K.dim + 1)}
        self.non_trivial_dim = frozenset(self.cocycle_reps.keys())

    def _p_prime(self):
        if not isprime(self.p):
            raise ValueError('p must be a prime number.')

    def cup(self, PHI: tuple[int, np.array], PSI: tuple[int, np.array]) -> tuple:
        """The cup product of two classes, each given as (degree, cocycle representative).

        Returns the degree of the product and its coefficients in the basis of that
        degree, ordered by pivot. Each representative must be one of self.cocycle_reps.
        """
        phiD, phi = PHI
        psiD, psi = PSI

        if phiD not in range(0, self.K.dim+1) or psiD not in range(0, self.K.dim+1):
            raise ValueError('Dimension error on cocycle representatives.')

        if not any(np.array_equal(phi, rep) for _, rep in self.cocycle_reps[phiD].items()):
            raise ValueError('No cocycle phi in respective dimension.')
        if not any(np.array_equal(psi, rep) for _, rep in self.cocycle_reps[psiD].items()):
            raise ValueError('No cocycle psi in respective dimension.')

        sorted_simplices = self.K.sorted_simplices[phiD + psiD] if phiD+psiD <= self.K.dim else tuple()

        if not sorted_simplices:
            return phiD + psiD, np.array((0, ))

        product = []
        for simplex in sorted_simplices:
            front_face, back_face = simplex[:phiD + 1], simplex[phiD:] # must index to find values.
            front_idx, back_idx = self._face_maps[phiD][front_face], self._face_maps[psiD][back_face]
            product.append( phi[front_idx] * psi[back_idx] % self.p )
        product = np.array(product).astype(int)

        if not any(product): # only happens when product is all zeros.
            return phiD + psiD, (0, ) * len(self.cocycle_reps[phiD+psiD])
        
        p, d = self.p, phiD + psiD
        residue = product % p
        sieve = self._coboundary_sieves[d - 1] if d >= 1 else {}
        reps = self.cocycle_reps[d]
        pivots, coeffs = {**sieve, **reps}, {}

        for pivot in sorted(pivots):
            c = int(residue[pivot]) % p
            if c:
                residue = (residue - c * pivots[pivot]) % p
            if pivot in reps:
                coeffs[pivot] = c

        assert not residue.any()          # the reduction has to clear completely

        # the coefficients of the product in the basis of H^{phiD + psiD}, in
        # the same order as sorted(self.cocycle_reps[phiD + psiD]).
        return phiD + psiD, tuple(coeffs[k] for k in sorted(reps))

    def cohomology_ring(self):
        reached_cocycle_reps = self.cocycle_reps.copy()

        for d in range(0, self.K.dim + 1):
            if d not in self.non_trivial_dim:
                continue

        #implmeent from here.



            