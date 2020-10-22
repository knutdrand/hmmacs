import numpy as np
from scipy.special import logsumexp
from hmmlearn.base import _BaseHMM
from hmmlearn._utils import check_is_fitted
from hmmlearn.utils import iter_from_X_lengths, log_mask_zero
from sklearn.utils import check_array
"""
transmat: t_{ij} from i to j
"""
def diagonalize(matrices):
    l, p = np.linalg.eig(matrices)
    return p, l, np.linalg.inv(p)

class _BaseSparseHMM(_BaseHMM):
    def score(X, lengths=None):
        pass

    def set_t_lattice(self, t):
        ###TODO Fix for multisequence
        self.__lengths = np.diff(t)
        self.__times = t

    def score(self, X, run_lengths, lengths=None):
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob, run_lengths[i:j])
            logprob += logprobij
        return logprob

    def _do_forward_pass(self, framelogprob, lengths):
        n_samples, n_components = framelogprob.shape
        matrices = self.transmat_[None, ...] * np.exp(framelogprob[..., None]) #add f_i to row i of t
        ms = diagonalize(matrices)
        lps, lds, lp_invs = (np.log(np.abs(m)) for m in ms)
        sps, sds, sp_invs = (np.sign(m) for m in ms)
        print(lds)
        print(lengths)
        lds *= lengths
        fwdlattice = np.zeros((n_samples, n_components))
        lv = log_mask_zero(self.startprob_)
        for t in range(n_samples):
            lp, ld, lr = (lps[t], lds[t], lp_invs[t])
            sp, sd, sr = (sps[t], sds[t], sp_invs[t])
            for i in range(2):
                fwdlattice[t, i] = logsumexp([lv[j] + lp[j, k]+ld[k]+lr[k, i] for k in range(2) for j in range(2)], 
                                             b=[sp[j, k]*sd[k]*sr[k, i] for k in range(2) for j in range(2)])
            lv = fwdlattice[t]
        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice
