import numpy as np
from numpy.linalg import matrix_power as mpow
from scipy.special import logsumexp
def diagonalize(matrices):
    l, p = np.linalg.eig(matrices)
    return p, l, np.linalg.inv(p)

def diagonal_sum(d, A, l):
    """
    sum_{0<=t<l}D^{l-t-1}AD^t
    """
    l = int(l)
    d1, d2 = d
    dij = (d1**(l)-d2**(l))/(d1-d2)
    return np.array([[l*d1**(l-1), dij],
                     [dij, l*d2**(l-1)]])*A

def log_diagonal_sum(d, A, l, sd):
    """
    logsumexp_{0<=t<l}D^{l-t-1}AD^t
    """
    l = int(l)
    d1, d2 = d
    numerator, s = logsumexp([d1*l, d2*l], b=[sd[0], -sd[1]], return_sign=True)
    denomenator, s2 = logsumexp((d1, d2), b=[sd[0], -sd[1]], return_sign=True)
    assert s*s2==1, (s, s2)
    dij = numerator-denomenator
    return np.array([[np.log(l)+(l-1)*d1, dij],
                     [dij, np.log(l)+d2*(l-1)]])+A


def sum_range(pdp, b, f, l):
    """
    TO = PDP^{-1}
    sum_{0<=t<=l} (TO)^{l-t} b  f(TO)^t
    PD^tP^{-1}bfPD^{l-t}P^{-1}
    """
    b = b.reshape((2, 1))
    f = f.reshape((1, 2))
    p, d, r = pdp
    A = r @ b @ f @ p
    S = diagonal_sum(d, A, l)
    return (p @ S @ r)

def log_mat_mul(A, B, sA, sB):
    print(A, B, sA, sB)
    res = np.empty((A.shape[0], B.shape[1]))
    sres = np.empty((A.shape[0], B.shape[1]))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j], sres[i, j] = logsumexp([A[i, k]+B[k, j] for k in range(A.shape[1])],
                                              b=[sA[i, k]*sB[k, j] for k in range(A.shape[1])], return_sign=True)
    return res, sres

def log_sum_range(pdp, b, f, l, sign_pdp):
    b = b.reshape((2, 1))
    f = f.reshape((1, 2))
    p, d, r = pdp
    sp, sd, sr = sign_pdp
    tmp_a, s_tmp_a = log_mat_mul(r, b, sr, np.ones_like(b))
    tmp_b, s_tmp_b = log_mat_mul(f, p, np.ones_like(f), sp)
    A, sA = log_mat_mul(tmp_a, tmp_b, s_tmp_a, s_tmp_b)
    S = log_diagonal_sum(d, A, l, sd)
    tmp_c, s_tmp_c = log_mat_mul(p, S, sp, sA)
    return log_mat_mul(tmp_c, r, s_tmp_c, sr)[0]

def compute_log_xi_sum():
    
    pass

def xi_sum(fs, T, bs, os, ls):
    ls = ls.copy()
    ls[0]-=1

    matrices = T[None, ...] * os[:, None, : ]
    pdps = diagonalize(matrices)
    ps, ds, rs = pdps
    prob = sum(fs[-1])
    if ls[0]>0:
        M = ps[0] @ np.diag(ds[0]*ls[0]) @ rs[0]
        M_inv = np.linalg.inv(M)
        first_f = fs[0][None, :] @ M_inv
    else:
        first_f = fs[0][None, :] # DOESNT-MATTER WHAT 
    fs = np.vstack((first_f, fs))
    local_sums = [T*o[None, :]*sum_range((p, d, r), b, f, l).T/prob
                  for p, d, r, b, f, o, l in zip(ps, ds, rs, bs, fs, os, ls)]
    return np.sum(local_sums, axis=0)


