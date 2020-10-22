import pytest
import numpy as np
from hmmacs.dense.xisum import *

N = 6
@pytest.fixture
def fs():
    return (np.arange(1,N*2+1)/10).reshape((-1, 2))

@pytest.fixture
def bs():
    return ((np.arange(1,N*2+1)/10)**2).reshape((-1, 2))

@pytest.fixture
def T():
    return np.array([[0.8, 0.2], [0.4, 0.6]])

@pytest.fixture
def os():
    return np.array([[0.7, 0.6], [0.3, 0.4]]*3)

def test_xi_sum(fs, bs, T, os):
    true = np.exp(compute_log_xi_sum(np.log(fs), np.log(T), np.log(bs), np.log(os)))
    calc = xi_sum(fs, T, bs, os)
    print(calc, true, calc/true)
    assert np.allclose(calc, true)
