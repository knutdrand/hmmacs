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
    assert s*s2==1, (s, s2, d, sd, l)
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

def compute_log_xi_sum(fs, T, bs, os, ls):
    ls = ls.copy()
    ls[0]-=1

    matrices = np.exp(T)[None, ...] * np.exp(os)[:, None, : ]
    pdps = diagonalize(matrices)
    ps, ds, rs = (np.log(np.abs(m)) for m in pdps)
    sps, sds, srs = (np.sign(m) for m in pdps)
    logprob = logsumexp(fs[-1])
    if ls[0]>0:
        M_inv, s_M = log_mat_mul(ps[0], -(ds[0]*ls[0])[:, None] + rs[0], sps[0], sds[0][:, None]*srs[0])
        first_f, _ = log_mat_mul(fs[0][None, :],  M_inv, np.ones_like(fs[0][None, :]), s_M)
    else:
        first_f = fs[0][None, :] # DOESNT-MATTER WHAT 
    fs = np.vstack((first_f, fs))
    local_sums = [T+o[None, :]+ log_sum_range((p, d, r), b, f, l, (sp, sd, sr)).T-logprob
                  for p, d, r, b, f, o, l, sp, sd, sr in zip(ps, ds, rs, bs, fs, os, ls, sps, sds, srs) if l>0]
    return logsumexp(local_sums, axis=0)

def get_log_init_posterior(f, b, l, T, o):
    if l == 1:
        return f + b
    M = np.exp(T) * np.exp(o)[None, :]
    pdp = diagonalize(M)
    p, d, r = (np.log(np.abs(m)) for m in pdp)
    sp, sd, sr = (np.sign(m) for m in pdp)
    M, s_M = log_mat_mul(p, (d*(l-1))[:, None] + r, sp, sd[:, None]*sr)
    M_inv, s_M_inv = log_mat_mul(p, -(d*(l-1))[:, None] + r, sp, sd[:, None]*sr)
    first_f, _ = log_mat_mul(f[None, :],  M_inv, np.ones_like(f[None, :]), s_M_inv)
    first_b, _ = log_mat_mul(M, b[:, None], s_M,  np.ones_like(b[:, None]))
    return first_f.flatten()+first_b.flatten()

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

def posterior_sum(f, T, b, o, l):
    f = f.reshape((1, -1))
    b = b.reshape((-1, 1))
    M = T*o[None, :]
    p, d, r = diagonalize(M)
    A = r @ b @ f @ p
    ratios = d/d[::-1]
    d_ij = (1-ratios**l)/(1-ratios)
    D = np.array([[l, d_ij[0]], 
                  [d_ij[1], l]])
    print(D, A)
    return (p @ (D*A) @ r).diagonal()
