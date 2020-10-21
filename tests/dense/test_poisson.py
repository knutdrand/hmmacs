import pytest
import numpy as np
from hmmcaller.dense.poissonhmm import PoissonHMM

@pytest.fixture
def model():
    return PoissonHMM()

@pytest.fixture
def X():
    return np.asanyarray(np.concatenate([np.ones(4)*3, np.ones(6)*10]*3).reshape((-1, 1)), dtype="int")

def test_simple_converge(X, model):
    model.fit(X)
    assert np.all(model.predict(X)==np.concatenate([np.zeros(4), np.ones(6)]*3))

def test_noisy_converge(X, model):
    X+=((np.arange(X.size) % 4)-2)[:, None]
    model.fit(X)
    assert np.all(model.predict(X)==np.concatenate([np.zeros(4), np.ones(6)]*3))
