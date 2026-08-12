## Simplicial Cup Product ##

Computing simplicial cup products over `Z_p` of (finite) abstract simplicial
complexes. Intended to be used for smaller simplicial complexes; large complexes
with thousands of vertices have not been tested. Also computing homology and
cohomology over `Z` and `Z_p.` Implements simplicial complexes and chain 
complexes from scratch. Uses matrices for much of actual computations.

Simplicial cup products are computed via cocycle representatives, which were
computed by creating a sieve of coboundaries at each relevant dimension. Each
coboundary was represented as a vector, where the entries corresponded to the
image of the simplex associated with that position.

A cocycle basis was ran through the sieve. The sieve was constructed so that
coboundaries all have unique pivots. Each surviving cocycle had a unique leading
term (pivot) position not found in any of the coboundaries. These surviving
cocycles formed the corresponding dimension homology class representatives, where 
the number of them was verified to be equal to the Betti number.

Simplicial cup products were computed via the usual cochain-level formula on
the cocycle representatives, where the resulting cochain was ran through the 
corresponding dimension coboundary sieve, and then reduced against the cycle 
representatives, to produce a sum of cycle representatives in the target 
dimension. A matrix encoding the cup product computations was also implemented, 
taking advantage of graded-commutativity to make computations quicker.

I may soon implement a method to compute the actual cohomology ring by somehow 
finding a minimal set of generators and relations. 

Work in progress.


Contains a number of classes to :
1. Construct simplicial complexes.
2. Compute the Euler characteristic and boundary matrices of the simplicial 
   complex, with simplicial chains. Coefficients default to `Z`.
3. Compute the simplicial homology over `Z` and `Z_p`; cohomology over `Z_p`
   by elementary application of the UCT.
4. Computing simplicial cup products and a cup product matrix for a given
   abstract simplicial complex.

AI USAGE (Claude, via Claude Code):
1. Docstrings, most comments, citations and credits.
2. The test suite in `tests/`, and diagnosis against it. Several defects were
   found this way and fixed. The largest three:
    - Acycle basis read off `Matrix.nullspace`, which stays over `Q` and so 
      missed the mod 2 fundamental classes of `RP^2` and the Klein bottle
    - A reduction that renormalized the running vector and so lost the coefficient 
      in odd characteristic, breaking graded-commutativity
    - A `dict.keys()` view that discarded every cocycle representative.
3. Sourcing and verifying the 9-vertex `CP^2` used in the tests.
4. Profiling the product path and reworking it around cached per-bidegree
   lookups and product blocks, for a large speedup on repeated products.


## References ##

The method used in `simp_homology.py` for reading homology off the Smith
normal forms of the boundary matrices — the free rank as `m - r - s` and the
torsion coefficients as the non-unit elementary divisors — follows:

- Omar Antolín Camarena, *Using the Smith normal form to compute homology*,
  <https://www.matem.unam.mx/~omar/mathX27/smith-form.html>

The vertex-minimal `CP^2` triangulated in `tests/test_simp_homology.py` is due to:

- Wolfgang Kühnel and Thomas F. Banchoff, *The 9-vertex complex projective
  plane*, The Mathematical Intelligencer **5** (1983), 11–22.

## Credits ##

This package is built on the following third-party libraries:

- [SymPy](https://www.sympy.org/) (BSD-3-Clause) — exact linear algebra over
  `Z` and `GF(p)`: `Matrix`, `smith_normal_form`, `DomainMatrix`, and `isprime`.
  Smith normal form is not available in NumPy, which is why the homology
  computation runs through SymPy rather than staying in NumPy.
- [NumPy](https://numpy.org/) (BSD-3-Clause) — the boundary and coboundary
  matrices in `chain_complex.py`, and the integer matrix arithmetic used to
  check that the boundary map squares to zero.
- [pytest](https://pytest.org/) (MIT) — the test suite in `tests/`.
- [Hatchling](https://hatch.pypa.io/) (MIT) — the build backend declared in
  `pyproject.toml`.

`itertools` (`chain`, `combinations`), used in `simplicial_complex.py` to
generate the faces of a simplex, is part of the Python standard library.