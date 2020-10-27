from scipy.special import logsumexp

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
    return trace**(l-2)*((M @ b @ f)+ b @ f @ M ) + trace**(l-3)*(M @ b @ f @ M)*(l-2)
