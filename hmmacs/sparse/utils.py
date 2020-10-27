from scipy.special import logsumexp
from itertools import product
import numpy as np

def diagonalize(matrices):
    l, p = np.linalg.eig(matrices)
    return p, l, np.linalg.inv(p)

def log_eigvals(log_matrix, signs):
    a, b, c, d = log_matrix.flatten()
    sa, sb, sc, sd = signs.flatten()
    sqrt_exp, tmp_s = logsumexp([2*a, 2*d, np.log(2)+a+d, np.log(4)+b+c],
                                b=[1, 1, -1*sa*sd, 1*sb*sc], return_sign=True)
    assert np.all(tmp_s==1), (sqrt_exp, tmp_s)
    lp, sp = logsumexp([a, d, 0.5*sqrt_exp], b=[sa, sd, 1], return_sign=True)
    ln, sn = logsumexp([a, d, 0.5*sqrt_exp], b=[sa, sd, -1], return_sign=True)
    return np.array([lp, ln])-np.log(2), np.array([sp, sn])

def log_eig(log_matrix, signs):
    eigvals, eigsigns = log_eigvals(log_matrix, signs)
    assert np.all(np.abs(eigsigns)==1), (eigsigns, np.exp(log_matrix)*signs, log_matrix, signs, eigvals)
    p = np.zeros_like(log_matrix)
    p_signs = np.ones_like(log_matrix)
    ys, y_signs = logsumexp([log_matrix[0, 0]*np.ones_like(eigvals), eigvals], 
                            b=[-signs[0, 0]*np.ones_like(eigvals), eigsigns],
                            return_sign=True, axis=0)
    p[1, :] = ys-log_matrix[0, 1]
    p_signs[1, :] = signs[0, 1]*y_signs
    det, s_det = log_det(p, p_signs)
    return (eigvals, eigsigns), (p-0.5*det, s_det*p_signs)

def log_det(p, p_signs):
    return logsumexp([p[0, 0]+p[1, 1], p[1, 0]+p[0, 1]],
                     b=[p_signs[0, 0]*p_signs[1, 1], -p_signs[1, 0]*p_signs[0, 1]], return_sign=True)


def log_inv(log_matrix, signs):
    det, s_det = log_det(log_matrix, signs)
    a, b, c, d = log_matrix.flatten()
    sa, sb, sc, sd = signs.flatten()
    i_matrix = np.array([[d, b], [c, a]])
    i_signs = np.array([[sd, -sb], [-sc, sa]])
    return i_matrix-det, i_signs*s_det
    
def log_diagonalize(log_matrix, signs=np.ones((2, 2))):
    (leig, seig), (lp, sp) =  log_eig(log_matrix, signs)
    assert np.allclose(log_det(lp, sp)[0], 0), (lp, sp, log_matrix, log_det(lp, sp)[0])
    lr, sr = log_inv(lp, sp)

    return (lp, sp), (leig, seig), (lr, sr)

def log_matprod(matrices, signs):
    for m, s in zip(matrices, signs):
        assert m.shape == s.shape, (m, s)
    shape = (matrices[0].shape[0], matrices[-1].shape[1])
    res = np.zeros(shape)
    out_signs = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            tuples = [[i] + list(tup) + [j] for tup in  product(*(range(m.shape[1]) for m in matrices[:-1]))]
            res[i, j], out_signs[i, j] = logsumexp([np.sum([matrices[t][i_tup[t], i_tup[t+1]] for t in range(len(matrices))]) for i_tup in tuples],
                                  b = [np.prod([signs[t][i_tup[t], i_tup[t+1]] for t in range(len(matrices))]) for i_tup in tuples], return_sign=True)
    return res, out_signs


def log_mat_mul(A, B, sA, sB):
    res = np.empty((A.shape[0], B.shape[1]))
    sres = np.empty((A.shape[0], B.shape[1]))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j], sres[i, j] = logsumexp([A[i, k]+B[k, j] for k in range(A.shape[1])],
                                              b=[sA[i, k]*sB[k, j] for k in range(A.shape[1])], return_sign=True)
    return res, sres
