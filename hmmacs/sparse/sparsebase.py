import numpy as np
from scipy.special import logsumexp
from .utils import log_diagonalize
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

    def _decode_viterbi(self, X, run_lengths):
        framelogprob = self._compute_log_likelihood(X)
        state_sequence, logprob = viterbi(log_mask_zero(self.startprob_),
            log_mask_zero(self.transmat_), framelogprob, run_lengths)
        return logprob, state_sequence


    def __get_lpdps(self, framelogprob):
        matrices = np.log(self.transmat_[None, ...]) + framelogprob[:, None, : ]
        return [log_diagonalize(m) for m in matrices]

    def _do_backward_pass(self, framelogprob, run_lengths):
        n_samples, n_components = framelogprob.shape
        diagonalized = self.__get_lpdps(framelogprob)
        bwdlattice = np.zeros((n_samples, n_components))
        lv = bwdlattice[-1]
        for t in range(n_samples-1, 0, -1):
            (lp, sp), (ld, sd), (lr, sr) = diagonalized[t]
            ld *= run_lengths[t]
            for i in range(2):
                bwdlattice[t-1, i] = logsumexp([lp[i, k]+ld[k]+lr[k, c]+lv[c] for k in range(2) for c in range(2)], 
                                             b=[sp[i, k]*sd[k]*sr[k, c] for k in range(2) for c in range(2)])
            lv = bwdlattice[t-1]
        return bwdlattice

    def _do_forward_pass(self, framelogprob, run_lengths):
        n_samples, n_components = framelogprob.shape
        diagonalized = self.__get_lpdps(framelogprob)
        fwdlattice = np.zeros((n_samples, n_components))
        lv = log_mask_zero(self.startprob_)+framelogprob[0]
        for t in range(n_samples):
            (lp, sp), (ld, sd), (lr, sr) = diagonalized[t]
            ld *= run_lengths[t]-(t==0)
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
            first_posterior = get_log_init_posterior(np.log(self.startprob_)+framelogprob[0],
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
            full_fwdlattice = np.vstack((np.log(self.startprob_)+framelogprob[0], fwdlattice))
            log_xi_sum = compute_log_xi_sum(full_fwdlattice,
                                            log_mask_zero(self.transmat_), 
                                            bwdlattice, 
                                            framelogprob, 
                                            run_lengths)
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

    def decode(self, X, run_lengths, lengths=None, algorithm=None):
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j], run_lengths[i:j])
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, run_lengths, lengths=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, run_lengths, lengths)
        return state_sequence
