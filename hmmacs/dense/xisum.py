import numpy as np
from scipy.special import logsumexp

def compute_log_xi_sum(fwdlattice, log_transmat, bwdlattice, framelogprob):
    n_samples, n_components = framelogprob.shape
    logprob = logsumexp(fwdlattice[-1])
    log_xi_sum = np.full((n_components, n_components), -np.inf)
    work_buffer = np.full((n_components, n_components), -np.inf)
    for t in range(n_samples - 1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[i, j] = (fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j]
                                     - logprob)
        for i in range(n_components):
            for j in range(n_components):
                log_xi_sum[i, j] = np.logaddexp(log_xi_sum[i, j],
                                                work_buffer[i, j])
    return log_xi_sum

def xi_sum_simple(fs, T, bs, os):
    n_samples, n_components = fs.shape
    xi = np.zeros((n_components, n_components))
    tmp = np.zeros((n_components, n_components))
    prob = sum(fs[-1])
    for t in range(n_samples - 1):
        for i in range(n_components):
            for j in range(n_components):
                tmp[i, j] = fs[t, i]*T[i, j]*os[t+1, j]*bs[t+1, j]/prob
        xi += tmp
    return xi
                
def xi_sum(fs, T, bs, os):
    prob = sum(fs[-1])
    xis = [(b[:, None] @ f[None, :]).T*o[None, :] for f, b, o in zip(fs, bs[1:], os[1:])]
    return T*np.sum(xis, axis=0)/prob
