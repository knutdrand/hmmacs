import pytest
import numpy as np
from scipy.special import logsumexp
from hmmacs.sparse.poissonhmm import PoissonHMM
from hmmacs.dense.poissonhmm import PoissonHMM as DensePoissonHMM

@pytest.fixture
def model():
    m = PoissonHMM()
    m.startprob_ = np.array([0.2, 0.8])
    m.transmat_ = np.array([[0.3, 0.7],
                            [0.1, 0.9]])
    m.rate_ = np.array([2., 9.])
    return m

@pytest.fixture
def dense_model():
    m = DensePoissonHMM()
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

def get_dense_X(X, lengths):
    return np.concatenate([[x]*l for x, l in zip(X.flatten(), lengths.flatten())]).reshape((-1, 1))

def test_score(X, lengths, model, dense_model):
    dense_X = get_dense_X(X, lengths)
    true = dense_model.score(dense_X)
    sparse = model.score(X, lengths)
    assert np.allclose(sparse, true)

def test_forward_pass(X, lengths, model, dense_model):
    dense_X = get_dense_X(X, lengths)
    _, true = dense_model._do_forward_pass(dense_model._compute_log_likelihood(dense_X))
    ts = np.cumsum(lengths)-1
    _, sparse = model._do_forward_pass(model._compute_log_likelihood(X), lengths)
    for t in zip(true[ts], sparse, X, lengths):
        print(t)

    assert np.allclose(sparse, true[ts])


def test_backward_pass(X, lengths, model, dense_model):
    dense_X = get_dense_X(X, lengths)
    true = dense_model._do_backward_pass(dense_model._compute_log_likelihood(dense_X))
    ts = np.cumsum(lengths)-1
    sparse = model._do_backward_pass(model._compute_log_likelihood(X), lengths)
    for t in zip(true[ts], sparse, X, lengths):
        print(t)

    assert np.allclose(sparse, true[ts])
