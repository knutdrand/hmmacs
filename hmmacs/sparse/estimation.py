import numpy as np

def diagonal_sum(d, A, l):
    """
    sum_{0<=t<l}D^{l-t-1}AD^t
    """
    d1, d2 = d
    dij = (d1**(l)-d2**(l))/(d1-d2)
    return np.array([[l*d1**(l-1), dij], [dij, l*d2**(l-1)]])*A

def sum_range(pdp, b, f, l):
    """
    TO = PDP^{-1}
    sum_{0<=t<=l} (TO)^{l-t} b  f(TO)^t
    PD^tP^{-1}bfPD^{l-t}P^{-1}
    """
    p, d, r = pdp
    A = r @ b @ f @ p
    S = diagonal_sum(d, A, l)
    return (p @ S @ r)

def xi_sum(T, pdps, bs, fs, os, ls):
    ps, ds, rs = pdps
    M = ps[0] @ np.diag(ds[0]) @ rs[0]
    M_inv = np.linalg.inv(M)
    first_f = fs[0] @ M_inv
    fs = np.insert(fs, 0, first_f)
    return T*sum(o[:, None]*sum_range(pdp, b, f, l)
                 for pdp, b, f, o, l in zip(pdps, bs, fs, os, ls))


