def viterbi(startprob, log_transmat, framelogprob, run_lengths):
    n_samples, n_components = framelogprob.shape
    state_sequence = np.empty(run_lengths.size)
    viterbi_lattice = np.zeros((n_samples, n_components))
    work_buffer = np.empty(n_components)
    viterbi_lattice[0] = log_startprob + framelogprob[0]

    # Induction
    for t in range(1, n_samples):
        viterbi_lattice[t] = np.max(viterbi_lattice[t-1][:, None] + log_transmat, axis=0) + framelogprob[t] + log_transmat.diagonal()*(run_lengths[t]-1)
        
    # Observation traceback
    state_sequence[-1] = where_from = np.argmax(viterbi_lattice[-1])
        
    logprob = viterbi_lattice[-1, where_from]

    for t in range(n_samples - 2, -1, -1):
        for i in range(n_components):
        state_sequence[t, where_from] = np.argmax(viterbi_lattice[t][:, None] + log_transmat[:,where_from])
    return np.asarray(state_sequence), logprob
