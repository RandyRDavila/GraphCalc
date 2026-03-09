import numpy as np

from graphcalc.quantum.channel_generators import (
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    identity_channel,
)
from graphcalc.quantum.channel_invariants import (
    choi_eigenvalues,
    choi_rank,
    input_dimension,
    kraus_rank,
    output_dimension,
)
from graphcalc.quantum.channels import QuantumChannel


def test_identity_channel_invariants_qubit():
    chan = identity_channel(2)

    assert input_dimension(chan) == 2
    assert output_dimension(chan) == 2
    assert choi_rank(chan) == 1
    assert kraus_rank(chan) == 1

    evals = choi_eigenvalues(chan)
    assert evals.shape == (4,)
    assert np.count_nonzero(evals > chan.tol) == 1


def test_identity_channel_invariants_qutrit():
    chan = identity_channel(3)

    assert input_dimension(chan) == 3
    assert output_dimension(chan) == 3
    assert choi_rank(chan) == 1
    assert kraus_rank(chan) == 1

    evals = choi_eigenvalues(chan)
    assert evals.shape == (9,)
    assert np.count_nonzero(evals > chan.tol) == 1


def test_bit_flip_channel_rank_at_extremes():
    chan0 = bit_flip_channel(0.0)
    chan1 = bit_flip_channel(1.0)

    assert choi_rank(chan0) == 1
    assert kraus_rank(chan0) == 1
    assert choi_rank(chan1) == 1
    assert kraus_rank(chan1) == 1


def test_bit_flip_channel_rank_interior_parameter():
    chan = bit_flip_channel(0.3)

    assert choi_rank(chan) == 2
    assert kraus_rank(chan) == 2

    evals = choi_eigenvalues(chan)
    assert np.count_nonzero(evals > chan.tol) == 2


def test_amplitude_damping_channel_rank_behavior():
    chan0 = amplitude_damping_channel(0.0)
    chan1 = amplitude_damping_channel(1.0)
    chan_mid = amplitude_damping_channel(0.4)

    assert choi_rank(chan0) == 1
    assert kraus_rank(chan0) == 1

    assert choi_rank(chan1) == 2
    assert kraus_rank(chan1) == 2

    assert choi_rank(chan_mid) == 2
    assert kraus_rank(chan_mid) == 2


def test_depolarizing_channel_qubit_rank_behavior():
    chan0 = depolarizing_channel(0.0, dim=2)
    chan1 = depolarizing_channel(1.0, dim=2)
    chan_mid = depolarizing_channel(0.5, dim=2)

    assert choi_rank(chan0) == 1
    assert kraus_rank(chan0) == 1

    assert choi_rank(chan1) == 4
    assert kraus_rank(chan1) == 4

    assert choi_rank(chan_mid) == 4
    assert kraus_rank(chan_mid) == 4


def test_depolarizing_channel_qutrit_rank_behavior():
    chan0 = depolarizing_channel(0.0, dim=3)
    chan1 = depolarizing_channel(1.0, dim=3)

    assert choi_rank(chan0) == 1
    assert kraus_rank(chan0) == 1

    assert choi_rank(chan1) == 9
    assert kraus_rank(chan1) == 9


def test_rectangular_channel_dimensions_and_rank():
    k0 = np.array([[1, 0]], dtype=complex)
    k1 = np.array([[0, 1]], dtype=complex)
    chan = QuantumChannel.from_kraus([k0, k1])

    assert input_dimension(chan) == 2
    assert output_dimension(chan) == 1
    assert choi_rank(chan) == 2
    assert kraus_rank(chan) == 2

    evals = choi_eigenvalues(chan)
    assert evals.shape == (2,)
    assert np.count_nonzero(evals > chan.tol) == 2


def test_choi_eigenvalues_are_nonnegative_for_cptp_channels():
    channels = [
        identity_channel(2),
        bit_flip_channel(0.25),
        depolarizing_channel(0.6, dim=2),
        amplitude_damping_channel(0.5),
    ]

    for chan in channels:
        evals = choi_eigenvalues(chan)
        assert np.all(evals >= -chan.tol)
