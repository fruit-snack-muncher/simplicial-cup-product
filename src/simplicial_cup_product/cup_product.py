from simplicial_cup_product.simplicial_complex import SimplicialComplex
from simplicial_cup_product.chain_complex import ChainComplex
from simplicial_cup_product.simp_homology import SimpHomology
from sympy import isprime
import numpy as np


class CupProduct():
    """The cup product on H^*(K; Z_p), as a multiplication table.

    Everything here works towards cohomology_products(): the square matrix whose
    (a, b) entry is the a-th generator of H^*(K; Z_p) cup the b-th, over the flat
    basis self.generators. cup() and cup_from_rep() are that table read one entry
    or one linear combination at a time.

    K is taken to be a connected simplicial complex, and must not gain simplices
    once this is initialized: build a new CupProduct for a new complex.
    """

    def __init__(self, K: SimplicialComplex, p: int = 2):
        self.p = abs(p)
        if not isprime(self.p):
            raise ValueError('p must be a prime number.')

        self.K = K
        self.C = ChainComplex(K)
        self.H = SimpHomology(K, self.p)
        self.degrees = range(0, K.dim + 1)

        self._face_maps = {d: {face: i for i, face in enumerate(K.sorted_simplices[d])}
                           for d in self.degrees}
        self.cocycle_reps = {d: self.H.cocycle_reps(d) for d in self.degrees}

        # the generators of H^d as integer cochains, ordered by pivot. this order
        # is the index i of the i-th generator everywhere below.
        self.basis = {d: tuple(np.asarray(reps[pivot]).astype(int) for pivot in sorted(reps))
                      for d, reps in self.cocycle_reps.items()}

        # a flat basis of H^*: (degree, index into that degree's generators), by
        # degree and then by pivot. this is the row and column order of
        # cohomology_products, and the only place the grading is flattened.
        self.generators = tuple((d, i) for d in self.degrees for i in range(0, len(self.basis[d])))

        # filled on demand by _gather, _reducer and _product_block: the per-bidegree
        # face lookups, the per-degree echelon basis, and the products of the
        # generators, all of which are reused by every product of that shape.
        self._gathers, self._reducers, self._products = {}, {}, {}

    def _check_degrees(self, *degrees: int):
        if not set(degrees).issubset(self.degrees):
            raise ValueError('Dimension error on cocycle representatives.')

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

    def _reducer(self, d: int) -> tuple[dict, dict]:
        """The echelon basis of the d-cocycles, keyed by pivot, with the rep columns.

        The coboundaries and the cocycle representatives together span Z^d, each with
        a leading term of its own, so any cocycle reduces against them uniquely. The
        second entry says which pivots are representatives, and where their
        coefficients belong in the answer.
        """
        if d not in self._reducers:
            sieve = self.H._sieve_basis(d - 1, cocycles=True) if d >= 1 else {}
            reps = self.cocycle_reps[d]

            self._reducers[d] = ({pivot: np.asarray(vector).astype(int)
                                  for pivot, vector in {**sieve, **reps}.items()},
                                 {pivot: column for column, pivot in enumerate(sorted(reps))})

        return self._reducers[d]

    def _cup_indices(self, d: int, i: int, e: int, j: int) -> tuple:
        """The coefficients of the i-th generator of H^d cup the j-th of H^e.

        Unchecked: d + e must not exceed the dimension and both indices must be in
        range. cup and cup_from_rep are the ways in from outside.
        """
        front, back = self._gather(d, e)
        residue = (self.basis[d][i][front] * self.basis[e][j][back]) % self.p

        pivots, columns = self._reducer(d + e)
        coeffs = [0] * len(columns)

        # in an echelon basis the leading term of anything in the span is itself a
        # pivot, so this visits only the pivots that actually turn up - a handful,
        # where a sweep over all of them would run the length of the cochain. a
        # product of cocycles is a cocycle, hence always in the span.
        nonzero = np.flatnonzero(residue)
        while nonzero.size:
            pivot = int(nonzero[0])
            c = int(residue[pivot])

            residue = (residue - c * pivots[pivot]) % self.p
            if pivot in columns:
                coeffs[columns[pivot]] = c

            nonzero = np.flatnonzero(residue)

        return tuple(coeffs)

    def _product_block(self, d: int, e: int) -> np.array:
        """The products of the generators of H^d with those of H^e, as one array.

        Entry [i][j] holds the coefficients of the i-th generator of H^d cup the
        j-th of H^e, in the basis of H^{d+e}. Only d <= e is ever multiplied out:
        graded commutativity, psi cup phi = (-1)^{de} phi cup psi, makes the
        opposite block a signed transpose, and mod 2 the sign is invisible.
        """
        if (d, e) not in self._products:
            shape = (len(self.basis[d]), len(self.basis[e]), len(self.basis[d + e]))

            if d <= e:
                block = np.array([[self._cup_indices(d, i, e, j) for j in range(0, shape[1])]
                                  for i in range(0, shape[0])]).astype(int).reshape(shape)
            else:
                sign = -1 if (d * e) % 2 else 1
                block = (sign * np.swapaxes(self._product_block(e, d), 0, 1)) % self.p

            self._products[(d, e)] = block

        return self._products[(d, e)]

    def _generator_product(self, d: int, i: int, e: int, j: int) -> tuple[int, tuple]:
        """One entry of the table: the i-th generator of H^d cup the j-th of H^e.

        Reported as the degree of the product and its coefficients in the basis of
        that degree. Above the top dimension there is nothing to evaluate on and no
        basis to expand in, so the zero class is reported as a bare zero.
        """
        if d + e > self.K.dim:
            return d + e, (0,)

        return d + e, tuple(map(int, self._product_block(d, e)[i][j]))

    def _rep_index(self, d: int, cochain: np.array) -> int:
        """Which generator of H^d this cochain is, or None if it is not one of them."""
        for i, rep in enumerate(self.basis[d]):
            if np.array_equal(cochain, rep):
                return i

        return None

    def cup_from_rep(self, PHI: tuple[int, np.array], PSI: tuple[int, np.array]) -> tuple[int, tuple]:
        """The cup product of two classes, each given as (degree, cocycle representative).

        Each representative must be one of self.basis[degree], or a zero cochain,
        which represents the zero class. The answer is a table entry: the degree of
        the product and its coefficients in the basis of that degree, ordered by
        pivot.
        """
        (phiD, phi), (psiD, psi) = PHI, PSI
        self._check_degrees(phiD, psiD)

        if phiD + psiD > self.K.dim:
            return phiD + psiD, (0,)

        # a zero cochain is a coboundary, not a generator, so multiplication by it
        # is answered here rather than by the search for one below.
        if not any(phi) or not any(psi):
            return phiD + psiD, (0,) * len(self.basis[phiD + psiD])

        # locating each representative is the same scan that checks it is one, so
        # the index it yields is what gets handed on.
        i, j = self._rep_index(phiD, phi), self._rep_index(psiD, psi)

        if i is None:
            raise ValueError('No cocycle phi in given dimension.')
        if j is None:
            raise ValueError('No cocycle psi in given dimension.')

        return self._generator_product(phiD, i, psiD, j)

    def cup(self, PHI: tuple[int, tuple], PSI: tuple[int, tuple]) -> tuple[int, tuple]:
        """The cup product of two classes, each given as (degree, coefficients).

        The coefficients are a Z_p-linear combination of the generators of that
        degree, one entry per generator and in the order of self.basis. The answer
        is in the same form, in the basis of the product's degree.
        """
        (phiD, phi), (psiD, psi) = PHI, PSI
        self._check_degrees(phiD, psiD)

        if len(phi) != len(self.basis[phiD]) or len(psi) != len(self.basis[psiD]):
            raise ValueError('Coefficients do not match the generators of their degree.')

        if phiD + psiD > self.K.dim:
            return phiD + psiD, (0,)

        # the products of the generators are fixed, so the expansion is arithmetic
        # on the cached block: sum_ij phi_i psi_j (x_i cup y_j), one coefficient per
        # generator of H^{phiD + psiD}. no simplex is touched here.
        product = np.einsum('i,j,ijk->k', np.array(phi).astype(int), np.array(psi).astype(int),
                            self._product_block(phiD, psiD)) % self.p

        return phiD + psiD, tuple(map(int, product))

    def cohomology_products(self) -> np.ndarray:
        """The multiplication table of H^*(K; Z_p) under the cup product.

        A square object array over the generators listed in self.generators: entry
        [a][b] is the a-th generator cup the b-th, as (degree, coefficients in the
        basis of that degree) - the same answer cup gives for the corresponding
        basis vectors. Every bidegree is multiplied out once and cached on the way,
        so the table costs one pass over the blocks however often it is asked for.
        """
        n = len(self.generators)
        table = np.empty((n, n), dtype=object)

        for a, (d, i) in enumerate(self.generators):
            for b, (e, j) in enumerate(self.generators):
                table[a, b] = self._generator_product(d, i, e, j)

        return table
