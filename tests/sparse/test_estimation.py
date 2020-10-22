import pytest
import numpy as np
from hmmacs.sparse.estimation import diagonal_sum, sum_range, xi_sum, log_diagonal_sum, log_sum_range
from hmmacs.sparse.sparsebase import diagonalize
from hmmacs.dense.xisum import xi_sum as dense_xi_sum
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
    
def test_xi_sum(model, dense_model):
    X = np.array([5, 6])[:, None]
    lengths = np.array([1, 2])[:, None]
    dense_X = get_dense_X(X, lengths)
    dense_os = dense_model._compute_log_likelihood(dense_X)
    sparse_os = dense_model._compute_log_likelihood(X)
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
