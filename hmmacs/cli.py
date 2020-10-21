"""Console script for hmmacs."""
"""Console script for hmmcaller."""
import sys
import click

from bdgtools.io import read_bedgraph

from .io import write_narrowpeak
from .dense.adapters import run_controlled
from .dense.controlledpoissonhmm import ControlledPoissonHMM

@click.command()
@click.argument("treatment", type=click.Path())
@click.argument("control", type=click.Path())
@click.argument("outfile", type=click.File("w"))
def main(treatment, control, outfile):
    t = read_bedgraph(treatment)
    c = read_bedgraph(control)
    regions = run_controlled(t, c, ControlledPoissonHMM())
    write_narrowpeak(regions, outfile)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
