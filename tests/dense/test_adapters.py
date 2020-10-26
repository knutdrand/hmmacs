import pytest
import numpy as np

from bdgtools import Regions, BedGraph
from hmmacs.dense.poissonhmm import PoissonHMM
from hmmacs.dense.adapters import run

@pytest.fixture
def bedgraph():
    bedgraph = BedGraph([0, 4, 10, 14, 20, 24, 30],
                        [3, 10, 3, 10, 3, 10, 3])
    return [("chr1", bedgraph)]

@pytest.fixture
def regions():
    regions = Regions([4, 14, 24], [10, 20, 30])
    return {"chr1": regions}

def test_run(bedgraph, regions):
    assert run(bedgraph, PoissonHMM())[0] == regions
    
