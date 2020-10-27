import pytest
from .fixtures import f, b
import numpy as np

from hmmacs.sparse.singulars import singular_sum_range, log_singular_sum_range

mpow = np.linalg.matrix_power

@pytest.fixture
def A():
    return np.array([[2, 3], [4, 6]])


@pytest.mark.parametrize("l", [1, 2, 3, 10, 100])
def test_sum_range(A, f, b, l):
    true = sum(mpow(A, l-i-1) @ b @ f @ mpow(A, i)
               for i in range(l))
    assert np.allclose(singular_sum_range(A, b, f, l), true)

@pytest.mark.parametrize("l", [1, 2, 3, 10])
def test_log_sum_range(A, f, b, l):
    true = sum(mpow(A, l-i-1) @ b @ f @ mpow(A, i)
               for i in range(l))
    res = log_singular_sum_range(np.log(A), np.log(b), np.log(f), l)
    print(np.exp(res))
    print(true)
    assert np.allclose(np.exp(res),
                       true)
