from scipy.stats import poisson 
import numpy as np
from hmmlearn.base import _BaseHMM
from sklearn.utils import check_random_state
from sklearn import cluster

class PoissonHMM(_BaseHMM):
    r"""Hidden Markov Model with multinomial (discrete) emissions

    Parameters
    ----------

    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob\_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_components=2)  #doctest: +ELLIPSIS
    MultinomialHMM(algorithm='viterbi',...
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=2,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="str", init_params="str"):
        _BaseHMM.__init__(self, n_components,
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
            "r": nc
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
        # assert np.issubdtype(X.dtype, np.integer), X
        assert X.min() >= 0, X
        if hasattr(self, "n_features"):
            assert self.n_features == 1
        self.n_features = 1
