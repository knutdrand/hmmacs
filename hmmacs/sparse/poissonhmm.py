from scipy.stats import poisson 
import numpy as np
from .sparsebase import _BaseSparseHMM
from sklearn.utils import check_random_state
from sklearn import cluster

class PoissonHMM(_BaseSparseHMM):
    def __init__(self, n_components=2,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stk", init_params="stk"):
        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "k": nc
        }

    def _init(self, X, lengths=None):
        self._check_and_set_n_features(X)
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'r' in self.init_params:
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.rate_ = np.sort(kmeans.cluster_centers_.flatten())

    def _check(self):
        super()._check()
        assert self.rate_.shape == (self.n_components,), (self.rate_.shape, (self.n_components,))
        
    def _compute_log_likelihood(self, X):
        return poisson.logpmf(X.reshape((-1, 1)), self.rate_)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['counts'] = np.zeros(self.n_components)
        stats['posts'] = np.zeros(self.n_components)
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'r' in self.params:
            stats['counts'] += np.sum(X.reshape((-1, 1))*posteriors, axis=0)
            stats['posts'] += np.sum(posteriors, axis=0)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'r' in self.params:
            self.rate_ = (stats['counts']/stats['posts'])

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a Poisson distribution, i.e. an
        array of non-negative integers.
        """
        assert np.issubdtype(X.dtype, np.integer), X
        assert X.min() >= 0, X
        if hasattr(self, "n_features"):
            assert self.n_features == 1
        self.n_features = 1
