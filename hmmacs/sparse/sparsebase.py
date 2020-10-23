import numpy as np
from scipy.special import logsumexp
from hmmlearn.base import _BaseHMM
from hmmlearn._utils import check_is_fitted
from hmmlearn.utils import iter_from_X_lengths, log_mask_zero, log_normalize
from sklearn.utils import check_array

from .estimation import compute_log_xi_sum, get_log_init_posterior

"""
transmat: t_{ij} from i to j
"""
def diagonalize(matrices):
    l, p = np.linalg.eig(matrices)
    return p, l, np.linalg.inv(p)

class _BaseSparseHMM(_BaseHMM):
    def score(X, lengths=None):
        pass

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

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                           posteriors, fwdlattice, bwdlattice, run_lengths):
        stats['nobs'] += 1
        if 's' in self.params:
            first_posterior = get_log_init_posterior(fwdlattice[0],
                                                     bwdlattice[0],
                                                     run_lengths[0],
                                                     log_mask_zero(self.transmat_),
                                                     framelogprob[0])
            log_normalize(first_posterior)
            stats['start'] += np.exp(first_posterior)
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            if n_samples <= 1:
                return
            log_xi_sum = compute_log_xi_sum(fwdlattice, log_mask_zero(self.transmat_), 
                                            bwdlattice, framelogprob, run_lengths)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def fit(self, X, run_lengths, lengths=None):
        X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_._reset()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                rls = run_lengths[i:j]
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob, rls)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob, rls)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                print(self._accumulate_sufficient_statistics)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice, rls)
            self._do_mstep(stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        if (self .transmat_.sum(axis=1) == 0).any():
            _log.warning("Some rows of transmat_ have zero sum because no "
                         "transition from the state was ever observed.")

        return self
