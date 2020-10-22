import pytest
import numpy as np
from hmmacs.sparse.poissonhmm import PoissonHMM
from hmmacs.dense.poissonhmm import PoissonHMM as DensePoissonHMM

@pytest.fixture
def model():
    m = PoissonHMM()
    m.startprob_ = np.array([0.2, 0.8])
    m.transmat_ = np.array([[0.3, 0.7],
                            [0.1, 0.9]])
    m.rate_ = np.array([2., 1.])
    return m

@pytest.fixture
def dense_model():
    m = DensePoissonHMM()
    m.startprob_ = np.array([0.2, 0.8])
    m.transmat_ = np.array([[0.3, 0.7],
                            [0.1, 0.9]])
    m.rate_ = np.array([2., 1.])
    return m

@pytest.fixture
def dense_X():
    return np.asanyarray(np.concatenate([np.ones(4)*3, np.ones(6)*10]*3).reshape((-1, 1)), dtype="int")

@pytest.fixture
def X():
    return np.array([3, 10]*3, dtype="int").reshape((-1, 1))

@pytest.fixture
def lengths():
    return np.array([4, 6]*3, dtype="int").reshape((-1, 1))

def test_score(X, lengths, model, dense_X, dense_model):
    true = dense_model.score(dense_X)
    sparse = model.score(X, lengths)
    assert np.allclose(sparse, true)

def _test_forward_pass(X, lengths, model, dense_X, dense_model):
    _, true = dense_model._do_forward_pass(dense_X)
    ts = np.cumsum(lengths)-1
    _, sparse = model._do_forward_pass(X, lengths)
    assert np.allclose(sparse, true[ts])
