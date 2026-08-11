"""
A representation of (abstract) simplicial complexes as a set
of tuples - where simplices are implemented as tuples of points,
ordered with a canonical global ordering. Simplicial triangula-
tions are always built from an empty set, and simplices are 
added to the abstract simplicial complex in a way that is down-
ward closed. The empty *simplex* is not included.

Builds a ranking of vertices (rank), assigning an order to all
0-simplices in the complex.
"""

from itertools import chain, combinations

class SimplicialComplex:
    def __init__(self, *K):
        self.Vrank = []
        self.simplices = {}
        self.sorted_simplices = {}
        self.dim = 0

        for simplex in K:
            self.add_simplex(simplex)


    def __str__(self):
        return str(self.sorted_simplices)

    # return the dimension of a simplex; the length of the tuple - 1
    def _simplex_dimension(self, s: tuple) -> int:
        return len(s) - 1

    def _update_Vrank(self, s: tuple) -> None:
        """Append the vertices of s to the global ordering, keeping first-seen order."""
        new_Vrank = self.Vrank + list(s)
        updated_Vrank = dict.fromkeys(new_Vrank)
        self.Vrank = list(updated_Vrank)

    def _update_order_lookup(self) -> None:
        """Rebuild the vertex -> position map from the global ordering."""
        self._order_lookup = {point: order for order, point in enumerate(self.Vrank)}

    def _update_dim(self) -> None:
        if self.simplices: # throws an error upon init...
            self.dim = max(self.simplices.keys())

    def _sort_all_simplices(self) -> None:
        """Order each dimension's simplices lexicographically by vertex rank.

        The tuples themselves are left untouched: the boundary operator in
        chain_complex.py relies on a face carrying the vertex order of the simplex it
        came from.
        """
        for d in range(len(self.simplices.keys())):
            # all dimension d simplices
            d_simplices = list(self.simplices[d])
            self.sorted_simplices[d] = sorted(d_simplices, key = lambda l: tuple(self._order_lookup[p] for p in l))

    def d_simplices(self, d: int) -> set:
        if d >= 0 and d <= self.dim:
            return self.simplices[d]
        raise ValueError('Dimension d must be nonnegative and at most the dimension of the complex.')

    def d_sorted_simplices(self, d: int) -> set:
            if d >= 0 and d <= self.dim:
                return self.sorted_simplices[d]
            raise ValueError('Dimension d must be nonnegative and at most the dimension of the complex.')
    
    def add_simplex(self, *simplices) -> None:
        """Add simplices with all their faces, keeping the complex downward closed."""
        for s in simplices:
            if not type(s) is tuple:
                raise TypeError(f"Expected tuple input, found input type {type(s)}")
            if len(set(s)) != len(s):
                raise ValueError(f"Vertices in any simplex must be unique. Found a simplex: {s}")

            # add the 0-simplices to the global vertex rank
            self._update_Vrank(s) 

            # add empty sets for dimension of all faces of K, if necessary.
            for i in range(0, self._simplex_dimension(s) + 1):
                if i not in self.simplices:
                    self.simplices[i] = set()

            # add face simplices to corresponding dimension
            faces = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
            for face in faces:
                self.simplices[self._simplex_dimension(face)].add(face)

        self._update_order_lookup()
        self._sort_all_simplices()
        self._update_dim()
    