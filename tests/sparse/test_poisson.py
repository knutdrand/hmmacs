import pytest
import numpy as np
from scipy.special import logsumexp
from hmmacs.sparse.poissonhmm import PoissonHMM
from hmmacs.dense.poissonhmm import PoissonHMM as DensePoissonHMM
from .fixtures import *

def test_score(X, lengths, model, dense_model):
    dense_X = get_dense_X(X, lengths)
    true = dense_model.score(dense_X)
    sparse = model.score(X, lengths)
    assert np.allclose(sparse, true)

@pytest.mark.parametrize("lengths", [[1]*4 + [5, 5]*3, [6, 4]*3 + [1]*4])
def test_forward_pass(X, lengths, model, dense_model):
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    _, true = dense_model._do_forward_pass(dense_model._compute_log_likelihood(dense_X))
    ts = np.cumsum(lengths)-1
    _, sparse = model._do_forward_pass(model._compute_log_likelihood(X), lengths)
    for t in zip(true[ts], sparse, X, lengths):
        print(t)

    assert np.allclose(sparse, true[ts])

@pytest.mark.parametrize("lengths", [[1]*4 + [5, 5]*3, [6, 4]*3 + [1]*4])
def test_backward_pass(X, lengths, model, dense_model):
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    true = dense_model._do_backward_pass(dense_model._compute_log_likelihood(dense_X))
    ts = np.cumsum(lengths)-1
    sparse = model._do_backward_pass(model._compute_log_likelihood(X), lengths)
    for t in zip(true[ts], sparse, X, lengths):
        print(t)

    assert np.allclose(sparse, true[ts])
