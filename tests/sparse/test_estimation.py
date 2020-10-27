import pytest
import numpy as np
from hmmacs.sparse.estimation import diagonal_sum, sum_range, xi_sum, log_diagonal_sum, log_sum_range, compute_log_xi_sum, posterior_sum, log_posterior_sum, log_mat_mul
from hmmacs.sparse.utils import log_inv
from hmmacs.sparse.sparsebase import diagonalize
from hmmacs.dense.xisum import xi_sum as dense_xi_sum
from hmmacs.dense.xisum import compute_log_xi_sum as dense_log_xi_sum
from scipy.special import logsumexp
from hmmlearn.utils import normalize
from .fixtures import *

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
    
@pytest.mark.parametrize("X", [[5, 6, 7, 8]])
@pytest.mark.parametrize("lengths", [[4, 3, 2, 1], [1, 2, 3, 4]])
def test_xi_sum(X, lengths, model, dense_model):
    X = np.array(X)[:, None]
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    dense_os = dense_model._compute_log_likelihood(dense_X)
    sparse_os = model._compute_log_likelihood(X)
    dense_fs = np.exp(dense_model._do_forward_pass(dense_os)[1])
    dense_bs = np.exp(dense_model._do_backward_pass(dense_os))
    dense_xi = dense_xi_sum(dense_fs, dense_model.transmat_, dense_bs, np.exp(dense_os))

    sparse_fs = np.exp(model._do_forward_pass(sparse_os, lengths)[1])
    sparse_bs = np.exp(model._do_backward_pass(sparse_os, lengths))
    sparse_xi = xi_sum(sparse_fs, model.transmat_, sparse_bs, np.exp(sparse_os), lengths)
    print(sparse_xi)
    print(dense_xi)
    assert np.allclose(sparse_xi, dense_xi)

def test_log_diagonal_sum(d, A):
    true = diagonal_sum(d, A, l)
    logged = log_diagonal_sum(np.log(np.abs(d)), np.log(A), l, np.sign(d))
    print(true)
    print(np.exp(logged))
    assert np.allclose(np.exp(logged), true)

def test_log_sum_range(d, A, f, b):
    pdp = diagonalize(A)
    true = sum_range(pdp, b, f, l)
    log_pdp = tuple(np.log(abs(m)) for m in pdp)
    sign_pdp = tuple(np.sign(m) for m in pdp)
    logged = log_sum_range(log_pdp, np.log(b), np.log(f), l, sign_pdp)
    print(true)
    print(np.exp(logged))
    assert np.allclose(np.exp(logged), true)

@pytest.mark.parametrize("X", [[5, 6, 7, 8]])
@pytest.mark.parametrize("lengths", [[4, 3, 2, 1], [1, 2, 3, 4]])
def test_log_xi_sum(X, lengths, model):
    X = np.array(X)[:, None]
    lengths = np.array(lengths)[:, None]
    sparse_os = model._compute_log_likelihood(X)
    sparse_fs = model._do_forward_pass(sparse_os, lengths)[1]
    sparse_bs = model._do_backward_pass(sparse_os, lengths)
    true = xi_sum(np.exp(sparse_fs), model.transmat_, np.exp(sparse_bs), np.exp(sparse_os), lengths)
    sparse_fs = np.vstack((np.log(model.startprob_)+sparse_os[0], sparse_fs))
    logged = compute_log_xi_sum(sparse_fs, np.log(model.transmat_), sparse_bs, sparse_os, lengths)
    print(true)
    print(logged)
    assert np.allclose(true, np.exp(logged))

@pytest.mark.parametrize("lengths", [[1]*4 + [5, 5]*3, [6, 4]*3 + [1]*4, [100]*10])
def test_xi_sum_full(model, dense_model, X, lengths):
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    dense_os = dense_model._compute_log_likelihood(dense_X)
    sparse_os = model._compute_log_likelihood(X)
    dense_fs = dense_model._do_forward_pass(dense_os)[1]
    dense_bs = dense_model._do_backward_pass(dense_os)
    dense_xi = dense_log_xi_sum(dense_fs, np.log(dense_model.transmat_), dense_bs, dense_os)
    sparse_fs = model._do_forward_pass(sparse_os, lengths)[1]
    sparse_bs = model._do_backward_pass(sparse_os, lengths)
    sparse_fs = np.vstack((np.log(model.startprob_)+sparse_os[0], sparse_fs))
    sparse_xi = compute_log_xi_sum(sparse_fs, np.log(model.transmat_), sparse_bs, sparse_os, lengths)
    sparse_xi = np.exp(sparse_xi)
    dense_xi = np.exp(dense_xi)
    normalize(sparse_xi, axis=1)
    normalize(dense_xi, axis=1)
    print(sparse_xi)
    print(dense_xi)
    assert np.allclose(sparse_xi, dense_xi)

@pytest.mark.parametrize("lengths", [[1]*4 + [4, 6]*3, [6, 4]*3 + [1]*4, [10]*10])
def test_fit(model, dense_model, X, lengths):
    lengths = np.array(lengths, dtype="int").reshape((-1, 1))
    model.n_iter=1
    dense_model.n_iter=1
    dense_X = get_dense_X(X, lengths)
    dense_model.fit(dense_X)
    model.fit(X, lengths)
    print(dense_model.transmat_)
    print(model.transmat_)
    assert np.allclose(dense_model.transmat_, model.transmat_)
    print(dense_model.startprob_)
    print(model.startprob_)
    assert np.allclose(dense_model.startprob_, model.startprob_)
    print(dense_model.rate_)
    print(model.rate_)
    assert np.allclose(dense_model.rate_, model.rate_)


# @pytest.mark.parametrize("X", [[5, 6, 7, 8]])
# @pytest.mark.parametrize("lengths", [[4, 3, 2, 1], [1, 2, 3, 4]])
@pytest.mark.parametrize("lengths", [[1]*4 + [4, 6]*3])#, [6, 4]*3 + [1]*4])
@pytest.mark.skip
def test_posterior_sum(X, lengths, model, dense_model):
    # X = np.array(X)[:, None]
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    dense_posteriors = dense_model.predict_proba(dense_X)
    dense_sum = np.sum(dense_posteriors, axis=0)
    sparse_os = model._compute_log_likelihood(X)
    sparse_fs = np.exp(model._do_forward_pass(sparse_os, lengths)[1])
    sparse_bs = np.exp(model._do_backward_pass(sparse_os, lengths))
    prob = np.sum(sparse_fs[-1].flatten())
    ts = np.cumsum(lengths)-1
    sparse_posteriors =  [posterior_sum(f, model.transmat_, b, np.exp(o), int(l), prob)
                          for f, b, o, l in zip(sparse_fs, sparse_bs, sparse_os, lengths)]
    sparse_sum = np.sum(sparse_posteriors, axis=0)
    assert np.allclose(sparse_sum, dense_sum)

# @pytest.mark.parametrize("X", [[5, 6, 7, 8]])
# @pytest.mark.parametrize("lengths", [[4, 3, 2, 1], [1, 2, 3, 4]])
@pytest.mark.parametrize("lengths", [[1]*4 + [4, 6]*3, [6, 4]*3 + [1]*4])
@pytest.mark.skip
def test_log_posterior_sum(X, lengths, model):
    # X = np.array(X)[:, None]
    lengths = np.array(lengths)[:, None]
    sparse_os = model._compute_log_likelihood(X)
    sparse_fs = model._do_forward_pass(sparse_os, lengths)[1]
    sparse_bs = model._do_backward_pass(sparse_os, lengths)
    prob = np.sum(np.exp(sparse_fs[-1]).flatten())
    log_prob = np.log(prob)
    posteriors =  [posterior_sum(np.exp(f), model.transmat_, np.exp(b), np.exp(o), int(l))/prob
                          for f, b, o, l in zip(sparse_fs, sparse_bs, sparse_os, lengths)]
    p_sum = np.sum(posteriors, axis=0)
    log_posteriors =  [log_posterior_sum(f, np.log(model.transmat_), b, o, int(l))[0]-log_prob
                       for f, b, o, l in zip(sparse_fs, sparse_bs, sparse_os, lengths)]
    log_p_sum = logsumexp(log_posteriors, axis=0)
    p_sum = logsumexp(np.log(posteriors), axis=0)
    assert np.allclose(p_sum, log_p_sum)


@pytest.mark.parametrize("lengths", [[1]*4 + [5, 5]*3, [6, 4]*3 + [1]*4, [1]*10])
def test_full_log_posterior_sum(X, lengths, model, dense_model):
    lengths = np.array(lengths)[:, None]
    dense_X = get_dense_X(X, lengths)
    dense_posteriors = dense_model.predict_proba(dense_X)
    dense_sum = np.sum(dense_posteriors, axis=0)
    sparse_os = model._compute_log_likelihood(X)
    sparse_fs = model._do_forward_pass(sparse_os, lengths)[1]
    sparse_bs = model._do_backward_pass(sparse_os, lengths)
    logprob = logsumexp(sparse_fs[-1].flatten())
    ts = np.cumsum(lengths)-1
    logT = np.log(model.transmat_)
    inv_mat, s_inv_mat = log_inv(logT+ sparse_os[0][None, :], np.sign(model.transmat_))
    first_f, sf = log_mat_mul((np.log(model.startprob_) + sparse_os[0]).reshape((1, -1)), inv_mat, np.ones((1, 2)), s_inv_mat)
    assert np.all(sf==1)
    sparse_fs = np.vstack((first_f, sparse_fs))
    log_sparse_posteriors = [log_posterior_sum(f, np.log(model.transmat_), b, o, int(l), logprob)
                             for f, b, o, l in zip(sparse_fs, sparse_bs, sparse_os, lengths)]
    sparse_posteriors = np.exp(log_sparse_posteriors)
    sparse_sum = np.sum(sparse_posteriors, axis=0)
    assert np.allclose(sparse_sum, dense_sum)
