import numpy as np
import pytest
from hmmacs.sparse.poissonhmm import PoissonHMM
from hmmacs.dense.poissonhmm import PoissonHMM as DensePoissonHMM

@pytest.fixture
def model():
    m = PoissonHMM(init_params="")
    m.startprob_ = np.array([0.2, 0.8])
    m.transmat_ = np.array([[0.3, 0.7],
                            [0.1, 0.9]])
    m.rate_ = np.array([2., 9.])
    return m

@pytest.fixture
def dense_model():
    m = DensePoissonHMM(init_params="")
    m.startprob_ = np.array([0.2, 0.8])
    m.transmat_ = np.array([[0.3, 0.7],
                            [0.1, 0.9]])
    m.rate_ = np.array([2., 9.])
    return m


@pytest.fixture
def X():
    return np.array([0, 1,2,3] + [3, 10]*3, dtype="int").reshape((-1, 1))

@pytest.fixture
def lengths():
    return np.array([1, 1,1,1] + [4, 6]*3, dtype="int").reshape((-1, 1))

@pytest.fixture
def f():
    return np.array([[0.5, 1.5]])

@pytest.fixture
def b():
    return np.array([[0.7],
                     [0.9]])

def get_dense_X(X, lengths):
    return np.concatenate([[x]*l for x, l in zip(X.flatten(), lengths.flatten())]).reshape((-1, 1))
