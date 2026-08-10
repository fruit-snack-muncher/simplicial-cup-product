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

AI USAGE:
1. Non-structural bugs, docstrings
2. Citations
3. Creating + using test suite
4. Credits

## References ##

The method used in `simp_homology.py` for reading homology off the Smith
normal forms of the boundary matrices — the free rank as `m - r - s` and the
torsion coefficients as the non-unit elementary divisors — follows:

- Omar Antolín Camarena, *Using the Smith normal form to compute homology*,
  <https://www.matem.unam.mx/~omar/mathX27/smith-form.html>

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