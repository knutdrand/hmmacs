from bdgtools import Regions
import numpy as np
def run(bedgraphs, model):
    X, lengths, chroms = from_bedgraphs(bedgraphs)
    model.fit(X, lengths)
    states = model.predict(X, lengths)
    print(states)
    return to_regions(states, lengths, chroms)

def run_controlled(t_bedgraphs, c_bedgraphs, model):
    X_t, lengths_t, chroms_t = from_bedgraphs(t_bedgraphs)
    X_c, lengths_c, chroms_c = from_bedgraphs(c_bedgraphs)
    assert lengths_t==lengths_c, (lengths_t,lengths_c)
    assert chroms_t==chroms_c, (chroms_t, chroms_c)
    X = np.hstack((X_t.reshape((-1, 1)), X_c.reshape((-1, 1))))
    model.fit(X, lengths_t)
    states = model.predict(X, lengths_t)
    # probs = model.predict_proba(X, lengths_t)
    probs = model._compute_log_likelihood(X)
    return to_regions(states, probs, lengths_t, chroms_t)

def from_bedgraphs(bedgraphs):
    chroms = []
    dense = []
    lengths = []
    for chrom, bedgraph in bedgraphs:
        N = bedgraph._size if bedgraph._size is not None else bedgraph._indices[-1]+1
        d = np.zeros(N)
        bedgraph.update_dense_diffs(d)
        dense.append(d.cumsum().reshape((-1, 1)))
        chroms.append(chrom)
        lengths.append(N)

    dense = np.concatenate(dense)
    return dense, lengths, chroms

def to_regions(states, log_probs, lengths, chroms):
    offsets = np.cumsum(lengths)
    start = 0
    regions = {}
    scores = {}
    peaks = {}
    log_probs = log_probs[:, 1]-log_probs[:, 0]
    # cumulative_log_probs = np.insert(np.cumsum(log_probs), 0, 0)
    for chrom, length in zip(chroms, lengths):
        dense = states[start:start+length]
        local_probs = log_probs[start:start+length]
        changes = np.flatnonzero(np.diff(dense))+1
        if dense[0] == 1:
            changes = np.insert(changes, 0, 0)
        if dense[-1] == 1:
            changes = np.append(changes, dense.size)
        changes = changes.reshape((-1, 2))
        regions[chrom] = Regions(changes[:, 0], changes[:, 1])
        scores[chrom] = [np.max(local_probs[start:end]) for start, end in zip(changes[:, 0], changes[:, 1])]
        peaks[chrom] = [np.mean(np.flatnonzero(local_probs[start:end]==m)).astype(int) for start, end, m in zip(changes[:, 0], changes[:, 1], scores[chrom])]
        # np.argmax(local_probs[start:end]) for start, end in zip(changes[:, 0], changes[:, 1])]
        # cumulative_log_probs[changes[:, 1]]-cumulative_log_probs[changes[:, 0]]
    return regions, scores, peaks
