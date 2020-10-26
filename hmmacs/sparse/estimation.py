import numpy as np
from numpy.linalg import matrix_power as mpow
from scipy.special import logsumexp
from itertools import product

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
    b = b.reshape((2, 1))
    f = f.reshape((1, 2))
    p, d, r = pdp
    sp, sd, sr = sign_pdp
    # tmp_a, s_tmp_a = log_mat_mul(r, b, sr, np.ones_like(b))
    # tmp_b, s_tmp_b = log_mat_mul(f, p, np.ones_like(f), sp)
    A, sA = log_matprod([r, b, f, p], [sr, np.ones_like(b), np.ones_like(f), sp])
    # A, sA = log_mat_mul(tmp_a, tmp_b, s_tmp_a, s_tmp_b)
    # assert np.allclose(A, Atmp), (A, Atmp, pdp, b,f,l)
    S = log_diagonal_sum(d, A, l, sd)
    return log_matprod([p, S, r], [sp, sA, sr])[0]
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
    print(local_sums)
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
    M = np.exp(T+o[None, :])
    f = log_mat_mul(f.reshape((1, -1)), np.log(M), np.ones((1,2)), np.ones_like(M))[0].flatten() # Get first f of range
    pdp = diagonalize(M)
    lpdp = (np.log(np.abs(m)) for m in pdp)
    spdp = (np.sign(m) for m in pdp)
    return log_sum_range(lpdp, b, f, l, spdp).diagonal()-logprob

def verbose_log_posterior_sum(f, T, b, o, l, logprob=1):
    f = f.flatten()
    b = b.flatten()
    M = np.exp(T+o[None, :])
    pdp = diagonalize(M)
    lp, ld, lr = (np.log(np.abs(m)) for m in pdp)
    sp, sd, sr = (np.sign(m) for m in pdp)
    fs = []
    bs = []
    for t in range(0, l):
        cur_f = np.zeros(2)
        cur_b = np.zeros(2)
        for i in range(2):
            cur_b[i] = logsumexp([lp[i, k]+t*ld[k]+lr[k, c]+b[c] for k in range(2) for c in range(2)], 
                                 b=[sp[i, k]*sd[k]*sr[k, c] for k in range(2) for c in range(2)])
            cur_f[i] = logsumexp([f[c] + lp[c, k]+(l-t)*ld[k]+lr[k, i] for k in range(2) for c in range(2)], 
                                 b=[sp[c, k]*sd[k]*sr[k, i] for k in range(2) for c in range(2)])
        print(t, l, np.exp(cur_f.flatten()+cur_b.flatten()-logprob))
        fs.append(cur_f)
        bs.append(cur_b)
    posteriors = [f.flatten() + b.flatten()-logprob for f, b in zip(fs, bs)]
    return logsumexp(posteriors, axis=0)

def verbose_posterior_sum(f, T, b, o, l, prob=1):
    f = f.reshape((1, -1))
    b = b.reshape((-1, 1))
    M = T*o[None, :]
    # M_inv = np.linalg.inv(M)
    bs = [(mpow(M, i) @ b).flatten() for i in range(l)]
    fs = [(f @ mpow(M, -i)).flatten() for i in range(l)]
    return np.sum([b*f/prob for b, f in zip(bs, fs)], axis=0)
         
    return np.sum([(mpow(M, -i) @ b).flatten()*(f @mpow(M ,i)).flatten()/prob
                   for i in range(l)], axis=0)


    p, d, r = diagonalize(M)
    v = (r @ b).flatten()
    w = (f @ p).flatten()
    res = np.zeros(2)
    for i in range(2):
        res[i] = l*p[i, 0]*r[0, i]*v[0]*w[0]
        res[i] += l*p[i, 1]*r[1, i]*v[1]*w[1]
        res[i] += (1-(d[0]/d[1])**l)/(1-d[0]/d[1])*p[i, 0]*r[1, i]*v[0]*w[1]
        res[i] += (1-(d[1]/d[0])**l)/(1-d[1]/d[0])*p[i, 1]*r[0, i]*v[1]*w[0]
    assert np.all(res>0), (res, v, w, f, b, o, l)
    return res/prob

def log_posterior_sum_old(f, T, b, o, l):
    """
    (1-r^n)/(1-r)
    (1-r^-n)/(1-r^-1) = (r^n-1)/r^l/(r-1)/r
    """
    f = f.reshape((1, -1))
    b = b.reshape((-1, 1))
    M = T+o[None, :]
    pdp= diagonalize(np.exp(M))
    p, d, r = (np.log(np.abs(m)) for m in pdp)
    sp, sd, sr = (np.sign(m) for m in pdp)
    ratio = d[0]-d[1]
    s_ratio = sd[0]*sd[1]
    print(s_ratio, sd, d)
    numerator, s = logsumexp([0, l*ratio], b=[1, -1*(s_ratio**l)], return_sign=True)
    denominator, s2 = logsumexp((0, ratio), b=[1, -1*s_ratio], return_sign=True)
    assert s*s2==1, (s, s2, d, sd, l)
    d_ij = numerator-denominator
    d_ji = d_ij-(l-1)*ratio
    D = np.array([[np.log(l), d_ij], 
                  [d_ji, np.log(l)]])
    sD = np.array([[1, s*s2], 
                   [s*s2*s_ratio**(l-1), 1]])

    A, sA = log_matprod([r, b, f, p], [sr, np.ones_like(b), np.ones_like(f), sp])
    # tmp_a, s_tmp_a = log_mat_mul(r, b, sr, np.ones_like(b))
    # tmp_b, s_tmp_b = log_mat_mul(f, p, np.ones_like(f), sp)
    # A, sA = log_mat_mul(tmp_a, tmp_b, s_tmp_a, s_tmp_b)
    S = D+A
    R, s= log_matprod([p, S, r], [sp, sA*sD, sr])
    # tmp_c, s_tmp_c = log_mat_mul(p, S, sp, sA)
    # R, s = log_mat_mul(tmp_c, r, s_tmp_c, sr)
    assert np.all(s.diagonal()==1), (s, R, f, b, o, T, l, sp,sA,sr)
    return R.diagonal(), s.diagonal()
