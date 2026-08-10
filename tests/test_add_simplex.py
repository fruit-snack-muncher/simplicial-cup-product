from simplicial_cup_product.simplicial_complex import SimplicialComplex
import pytest

def test_add_simplex_1():
    comp = SimplicialComplex()

    simp1 = ('a', 'b', 'c')
    simp2 = ('b', 'c', 'd')
    simp3 = ('b', 'd', 'e')

    comp.add_simplex(simp1, simp2, simp3)

    print(comp)

def test_add_simplex_2():
    comp = SimplicialComplex()

    simp1 = 'a'
    simp2 = ('b', 'c', 'd')
    simp3 = ('b', 'd', 'e')

    with pytest.raises(TypeError):
        comp.add_simplex(simp1, simp2, simp3)

def test_add_simplex_3():
    comp = SimplicialComplex()

    simp1 = ('a', 'b', 'a')
    simp2 = ('b', 'c', 'd')
    simp3 = ('b', 'd', 'e')

    with pytest.raises(ValueError):
        comp.add_simplex(simp1, simp2, simp3)

if __name__ == "__main__":
    test_add_simplex_2