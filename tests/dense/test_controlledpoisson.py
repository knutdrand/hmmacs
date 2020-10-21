import pytest
import numpy as np
from hmmcaller.dense.controlledpoissonhmm import ControlledPoissonHMM

@pytest.fixture
def model():
    return ControlledPoissonHMM(verbose=True)

@pytest.fixture
def X():
    l = np.arange(30)+1
    k = np.concatenate([np.ones(4)*3, np.ones(6)*10]*3)*l
    return np.hstack((k[:, None], l[:, None]))

def test_simple_converge(X, model):
    model.fit(X)
    assert np.all(model.predict(X)==np.concatenate([np.zeros(4), np.ones(6)]*3))

def test_noisy_converge(X, model):
    X[:, 0] +=((np.arange(X.shape[0]) % 4)-2)
    model.fit(X)
    assert np.all(model.predict(X)==np.concatenate([np.zeros(4), np.ones(6)]*3))
