import numpy as np
from numpy.linalg import matrix_power as mpow

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
    dij = logsumexp([d1*l, d2*l], b=[sd[0], -sd[1]])-logsumexp((d1, d2), b=sd[0], -sd[1])
    return np.array([[np.log(l)+l*d1, dij],
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


