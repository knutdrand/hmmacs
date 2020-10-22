import pytest
import numpy as np
from hmmacs.sparse.estimation import diagonal_sum, sum_range
from hmmacs.sparse.sparsebase import diagonalize

@pytest.fixture
def d():
    return np.array([0.9, 1.1])

@pytest.fixture
def A():
    return np.array([[0.6, 0.8], [1.0, 1.2]])

@pytest.fixture
def f():
    return np.array([[0.5, 1.5]])

@pytest.fixture
def b():
    return np.array([[0.7],
                     [0.9]])
l = 6
mpow = np.linalg.matrix_power

def test_diagonal_sum(d, A):
    true = sum(np.diag(d**(l-i-1))@A@np.diag(d**i) for i in range(l))
    assert np.allclose(diagonal_sum(d, A, l), true)

def test_sum_range(d, A, f, b):
    true = sum(mpow(A, l-i-1) @ b @ f @ mpow(A, i)
               for i in range(l))
    pdp = diagonalize(A)
    assert np.allclose(sum_range(pdp, b, f, l), true)
    

