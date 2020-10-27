"""Console script for hmmacs."""
"""Console script for hmmcaller."""
import sys
import click

from bdgtools.io import read_bedgraph
import numpy as np
from .io import write_narrowpeak
from .dense.adapters import run_controlled
from .dense.controlledpoissonhmm import ControlledPoissonHMM
from . import dense
from . import sparse

@click.command()
@click.argument("treatment", type=click.Path())
@click.argument("control", type=click.Path())
@click.argument("outfile", type=click.File("w"))
def main(treatment, control, outfile):
    t = read_bedgraph(treatment)
    c = read_bedgraph(control)
    regions = run_controlled(t, c, ControlledPoissonHMM())
    write_narrowpeak(regions, outfile)


@click.group()
def hmmtools():
    return 0

@hmmtools.command()
@click.argument("treatment", type=click.Path())
@click.argument("outfile", type=click.File("w"))
def poisson(treatment, outfile):
    model = sparse.PoissonHMM(init_params="rs")
    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    regions = sparse.run(read_bedgraph(treatment), model)

    write_narrowpeak(regions, outfile)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
