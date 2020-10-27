import numpy as np
from hmmacs.sparse.utils import log_eigvals, log_eig, log_inv, diagonalize, log_diagonalize

M = np.array([[1.2, 2.2], [0.2, 0.8]])

def test_log_eigvals():
    log_M = np.log(M)
    true_eig = np.linalg.eigvals(M)
    leig, seig = log_eigvals(log_M, np.ones_like(log_M))

    assert np.allclose(seig*np.exp(leig), true_eig)

def test_log_eig():
    log_M = np.log(M)
    true_eig, true_p = np.linalg.eig(M)
    (leig, seig), (lp, sp) = log_eig(log_M, np.ones_like(log_M))
    assert np.allclose(seig*np.exp(leig), true_eig)
    ratios = sp*np.exp(lp)/true_p
    print(ratios)
    assert np.allclose(ratios[0], ratios[1])

def test_log_inv():
    log_M = np.log(M)
    true_inv = np.linalg.inv(M)
    print(np.linalg.det(M))
    l, s = log_inv(np.log(np.abs(M)), np.sign(M))
    print(s*np.exp(l)/true_inv)
    print(true_inv)
    assert np.allclose(s*np.exp(l), true_inv)
    
def test_log_diagonalize():
    p, d, r = diagonalize(M)
    (lp, sp), (ld, sd), (lr, sr) = log_diagonalize(np.log(np.abs(M)), np.sign(M))
    rM = (sp*np.exp(lp)) @ (np.diag(sd*np.exp(ld))) @ (sr*np.exp(lr))
    assert np.allclose((sp*np.exp(lp)) @ (sr*np.exp(lr)), np.array([[1, 0], [0, 1]]))
    print(sp*np.exp(lp), sd*np.exp(ld), sr*np.exp(lr))


