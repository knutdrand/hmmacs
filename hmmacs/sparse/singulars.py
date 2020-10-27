from scipy.special import logsumexp
import numpy as np
from .utils import log_mat_mul, log_matprod

def is_singular(M):
    return M[0, 0]*M[1, 1] == M[0, 1]*M[1, 0]

def log_is_singular(M):
    return M[0, 0]+M[1, 1] == M[0, 1]+M[1, 0]

def singular_power(M, k):
    if k==0:
        return np.eye(2)
    return M.trace()**(k-1)*M

def singular_log_power(M, sM, k):
    assert k > 0
    trace, s_trace = logsumexp(M.diagonal(), b=sM.diagonal(), return_sign=True)
    return k*trace + M, sM*s_trace

def singular_sum_range(M, b, f, l):
    assert l > 0
    if l == 1:
        return b @ f
    trace = M.trace()
    res = trace**(l-2)*((M @ b @ f)+ b @ f @ M )
    if l==2:
        return res
    return res + trace**(l-3)*(M @ b @ f @ M)*(l-2)

def log_singular_sum_range(M, b, f, l):
    assert l > 0
    sM, sb, sf = (np.ones_like(tmp) for tmp in (M, b, f))
    if l == 1:
        return log_mat_mul(b, f, sb, sf)[0]
    trace = logsumexp(M.diagonal())
    res = (l-2)*trace + logsumexp([log_matprod([M, b, f], [sM, sb, sf])[0],
                                   log_matprod([b, f, M], [sb, sf, sM])[0]], axis=0)
    if l == 2:
        return res
    print(np.exp(res))
    tmp = (l-3)*trace + log_matprod([M, b, f, M], [sM, sb, sf, sM])[0] + np.log((l-2))
    print(np.exp(tmp))
    return logsumexp([res, tmp] , axis=0)
