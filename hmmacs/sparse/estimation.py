import numpy as np
from numpy.linalg import matrix_power as mpow
from scipy.special import logsumexp
from itertools import product
from .utils import log_diagonalize

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
    assert np.all(~np.isnan(A)), A
    assert np.all(~np.isnan(d)), d
    l = int(l)
    d1, d2 = d
    numerator, s = logsumexp([d1*l, d2*l], b=[sd[0], -sd[1]], return_sign=True)
    denomenator, s2 = logsumexp((d1, d2), b=[sd[0], -sd[1]], return_sign=True)
    assert s*s2==1, (s, s2, d, sd, l)
    dij = numerator-denomenator
    assert np.all(~np.isnan(dij)), dij
    assert l >= 1
    res = np.array([[np.log(l)+(l-1)*d1, dij],
                    [dij, np.log(l)+d2*(l-1)]])+A
    assert np.all(~np.isnan(res)), res
    # assert np.all(~np.isinf(res)), (res, A, d, l, sd)
    return res
                     


def sum_range(pdp, b, f, l):
    """
    TO = PDP^{-1}
    sum_{0<=t<=l} (TO)^{l-t} b  f(TO)^t
    PD^tP^{-1}bfPD^{l-t}P^{-1}
    """
    b = b.reshape((2, 1))
    f = f.reshape((1, 2))
    #if l==0:
    #    return b @ f
    p, d, r = pdp
    A = r @ b @ f @ p
    S = diagonal_sum(d, A, l)
    return (p @ S @ r)


def matprod(matrices):
    res = np.zeros((matrices[0].shape[0], matrices[-1].shape[1]))
    for i in range(matrices[0].shape[0]):
        for j in range(matrices[-1].shape[1]):
            tuples = [[i] + list(tup) + [j] for tup in  product(*(range(m.shape[1]) for m in matrices[:-1]))]
            res[i, j] = sum(np.prod([matrices[t][i_tup[t], i_tup[t+1]] for t in range(len(matrices))]) for i_tup in tuples)
    return res

def log_matprod(matrices, signs):
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

def log_sum_range(pdp, b, f, l, sign_pdp):
    assert l>0
    b = b.reshape((2, 1))
    f = f.reshape((1, 2))
    p, d, r = pdp
    sp, sd, sr = sign_pdp
    A, sA = log_matprod([r, b, f, p], [sr, np.ones_like(b), np.ones_like(f), sp])
    # assert np.all(~np.isinf(A)), (A, pdp, sign_pdp, b, f)
    S = log_diagonal_sum(d, A, l, sd)
    return log_matprod([p, S, r], [sp, sA, sr])[0]
    tmp_c, s_tmp_c = log_mat_mul(p, S, sp, sA)
    return log_mat_mul(tmp_c, r, s_tmp_c, sr)[0]

def compute_log_xi_sum(fs, T, bs, os, ls):
    ls = ls.copy()
    ls[0]-=1
    matrices = T[None, ...] + os[:, None, :]
    diagonalized = [log_diagonalize(m) for m in matrices]
    logprob = logsumexp(fs[-1])
    local_sums = [T+o[None, :] + log_sum_range((p, d, r), b, f, l, (sp, sd, sr)).T-logprob
                  for ((p, sp), (d, sd), (r, sr)), b, f, o, l in zip(diagonalized, bs, fs, os, ls) if l>0]

    return logsumexp(local_sums, axis=0)

def get_log_init_posterior(f, b, l, T, o):
    if l == 1:
        return f + b
    M = T + o[None, :]
    (p, sp), (d, sd), (r, sr) = log_diagonalize(M)
    M, s_M = log_mat_mul(p, (d*(l-1))[:, None] + r, sp, sd[:, None]*sr)
    first_b, _ = log_mat_mul(M, b[:, None], s_M,  np.ones_like(b[:, None]))
    return f.flatten()+first_b.flatten()

def xi_sum(fs, T, bs, os, ls):
    ls = ls.copy()
    ls[0]-=1

    matrices = T[None, ...] * os[:, None, : ]
    pdps = diagonalize(matrices)
    ps, ds, rs = pdps
    prob = sum(fs[-1])
    if ls[0]>0:
        M = ps[0] @ np.diag(ds[0]**ls[0]) @ rs[0]### TODO: Should be power
        M_inv = np.linalg.inv(M)
        assert np.allclose(M_inv, ps[0] @ np.diag(1/ds[0]**ls[0]) @ rs[0]), (M, ps[0] @ np.diag(1/ds[0]**ls[0]) @ rs[0])
        first_f = fs[0][None, :] @ M_inv
    else:
        first_f = fs[0][None, :] # DOESNT-MATTER WHAT 
    fs = np.vstack((first_f, fs))
    local_sums = [T*o[None, :]*sum_range((p, d, r), b, f, l).T/prob
                  for p, d, r, b, f, o, l in zip(ps, ds, rs, bs, fs, os, ls)]
    return np.sum(local_sums, axis=0)

def posterior_sum(f, T, b, o, l, prob=1):
    f = f.reshape((1, -1))
    b = b.reshape((-1, 1))
    M = T*o[None, :]
    p, d, r = diagonalize(M)
    A = r @ (b/prob) @ f @ p
    ratios = d/d[::-1]
    d_ij = (1-ratios**l)/(1-ratios)
    D = np.array([[l, d_ij[0]], 
                  [d_ij[1], l]])
    res = (p @ (D*A) @ r).diagonal()
    assert np.all(res>0), (res, A, D, f, b, o, l)
    return res

def log_posterior_sum(f, T, b, o, l, logprob=0):
    M = T+o[None, :]
    f = log_mat_mul(f.reshape((1, -1)), M, np.ones((1,2)), np.ones_like(M))[0].flatten() # Get first f of range
    (p, sp), (d, sd), (r, sr) = log_diagonalize(M)
    return log_sum_range((p, d, r), b, f, l, (sp, sd, sr)).diagonal()-logprob
