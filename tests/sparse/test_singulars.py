import pytest
from .fixtures import f, b
import numpy as np

from hmmacs.sparse.singulars import singular_sum_range

mpow = np.linalg.matrix_power

@pytest.fixture
def A():
    return np.array([[2, 3], [4, 6]])


@pytest.mark.parametrize("l", [1, 10, 100])
def test_sum_range(A, f, b, l):
    true = sum(mpow(A, l-i-1) @ b @ f @ mpow(A, i)
               for i in range(l))
    assert np.allclose(singular_sum_range(A, b, f, l), true)
