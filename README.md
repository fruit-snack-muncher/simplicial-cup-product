## Simplicial Cup Product ##

A work-in-progress repo, aiming tocomputing the simplicial cup product 
over `Z_p` of simplicial complexes. Intended to be used for smaller simplicial
complexes.

Contains a number of classes to :
1. Construct simplicial complexes (simplicial_complex.py)
2. Compute the Euler characteristic and boundary matrices
   of the simplicial complex, with simplicial chains.
   Coefficients default to `Z`. (chain_complex.py)
3. Compute the simplicial homology over `Z` and `Z_p`, and
   cohomology over `Z_p.` 

AI USAGE (Claude, via Claude Code):
1. Docstrings, and a few comments.
2. Citations + credits
3. Creating + using test suite.
4. Three defects in `_cycle_reps` (`simp_homology.py`), found by writing its
   test suite and then fixed: an unbound `sieve` in the top dimension, a live
   `dict.keys()` view that discarded every representative, and a cycle basis
   read off `Matrix.nullspace`, which stays over `Q` and so missed the mod 2
   fundamental classes of `RP^2` and the Klein bottle.
5. Sourcing the 9-vertex `CP^2` used in the tests, and checking its f-vector,
   Euler characteristic, ridge degrees and integral homology before use.
6. `tests/test_cup_product.py`, and repeated diagnosis of `cup` against it —
   most substantially that reducing a product with `_cycle_sieve` renormalizes
   the running vector and loses its scale, so the coefficient was wrong in odd
   characteristic; plain reduction against the coboundary and representative
   pivots, with no renormalization, fixed it and made `cup` graded-commutative.


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