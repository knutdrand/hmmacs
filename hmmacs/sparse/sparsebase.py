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

    def __get_lpdps(self, framelogprob):
        matrices = self.transmat_[None, ...] * np.exp(framelogprob[:, None, : ]) #add f_i to row i of t
        tmp = (self.startprob_ * np.exp(framelogprob[0]))[None, :]
        ms = diagonalize(matrices)
        lps, ilds, lp_invs = (np.log(np.abs(m)) for m in ms)
        sps, sds, sp_invs = (np.sign(m) for m in ms)
        return (lps, ilds, lp_invs), (sps, sds, sp_invs)

    def _do_backward_pass(self, framelogprob, lengths):
        n_samples, n_components = framelogprob.shape
        (lps, ilds, lp_invs), (sps, sds, sp_invs) = self.__get_lpdps(framelogprob)
        lds = ilds*lengths
        bwdlattice = np.zeros((n_samples, n_components))
        lv = bwdlattice[-1]
        for t in range(n_samples-1, 0, -1):
            lp, ld, lr = (lps[t], lds[t], lp_invs[t])
            sp, sd, sr = (sps[t], sds[t], sp_invs[t])
            for i in range(2):
                bwdlattice[t-1, i] = logsumexp([lp[i, k]+ld[k]+lr[k, c]+lv[c] for k in range(2) for c in range(2)], 
                                             b=[sp[i, k]*sd[k]*sr[k, c] for k in range(2) for c in range(2)])
            lv = bwdlattice[t-1]
        return bwdlattice
                
    def _do_forward_pass(self, framelogprob, lengths):
        n_samples, n_components = framelogprob.shape
        (lps, ilds, lp_invs), (sps, sds, sp_invs) = self.__get_lpdps(framelogprob)
        lds = ilds*lengths
        lds[0] -= ilds[0]
        fwdlattice = np.zeros((n_samples, n_components))
        lv = log_mask_zero(self.startprob_)+framelogprob[0]
        for t in range(n_samples):
            lp, ld, lr = (lps[t], lds[t], lp_invs[t])
            sp, sd, sr = (sps[t], sds[t], sp_invs[t])
            for j in range(2):
                fwdlattice[t, j] = logsumexp([lv[c] + lp[c, k]+ld[k]+lr[k, j] for k in range(2) for c in range(2)], 
                                             b=[sp[c, k]*sd[k]*sr[k, j] for k in range(2) for c in range(2)])
            lv = fwdlattice[t]
        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice
