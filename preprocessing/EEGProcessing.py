import os
import glob
import numpy as np
from scipy import signal, stats
from neurokit2.complexity import complexity_hurst
from fooof import FOOOF
from avalanche import detect_avalanches, branching_ratio
from antropy import lziv_complexity, detrended_fluctuation
from mne.filter import filter_data
from joblib import Parallel, delayed


def load_data(data_path, hypnogram_path, dtype=None):
    """
    Loads EEG data and hypnogram file and returns them as numpy arrays.

    Args:
        data_path: file path to the EEG data file
        hypnogram: file path to the hypnogram file
        dtype: the data type to return the arrays in, None will not change the type

    Returns:
        EEG data as numpy array (electrodes x epoch steps x epochs)
        sleep hypnogram as numpy vector
    """
    data = np.load(data_path)
    hypnogram = np.load(hypnogram_path)
    if dtype is None:
        return data, hypnogram
    else:
        return data.astype(dtype), hypnogram.astype(dtype)


def extract_sleep_stages(data, hyp):
    """
    Extracts sleep stages from EEG data using a hypnogram, epochs from the same sleep stage are grouped.

    Args:
        data: EEG data as (electrodes x epoch steps x epochs)

    Returns:
        dictionary with sleep stage names as keys and EEG epochs as values with shape (electrodes x epoch steps x epochs)
    """
    if data.shape[2] != hyp.shape[0]:
        # the epoch count does not match the hypnogram length, adjusting EEG data shape
        data = data[:, :, : hyp.shape[0]]

    # set hypnogram values of AWA stage after the first sleep to 6
    mask = np.ones(hyp.shape)
    mask[: np.where(hyp != 0)[0][0]] = 0
    hyp[(hyp == 0) & (mask == 1)] = 6

    # create a dictionary with entries for each sleep stage
    return {
        "AWA": data[:, :, (hyp == 0) | (hyp == 6)],
        "AWSL": data[:, :, hyp == 6],
        "N1": data[:, :, hyp == 1],
        "N2": data[:, :, hyp == 2],
        "N3": data[:, :, (hyp == 3) | (hyp == 4)],
        "NREM": data[:, :, (hyp == 1) | (hyp == 2) | (hyp == 3) | (hyp == 4),],
        "REM": data[:, :, hyp == 5],
    }


def load_pre_split_data(path, subject_id):
    """
    Loads data that was previously split into sleep stages.

    Args:
        path: path to the directory where data is stored
        subject_id: identifier of the subject for which data should be loaded

    Returns:
        dictionary with sleep stage names as keys and EEG epochs as values with shape (electrodes x epoch steps x epochs)
    """
    data = dict()
    # find paths to all files for the specific subject
    paths = glob.glob(os.path.join(path, f"{subject_id}_*.npy"))
    for p in paths:
        # extract the sleep stage from the file name
        stage = p.split(os.sep)[-1].split("_")[1]
        # load the data
        current = np.load(p)

        if stage in data:
            data[stage].append(current)
        else:
            data[stage] = [current]
    # concatenate individual files along the epoch dimension
    for stage in data.keys():
        data[stage] = np.concatenate(data[stage], axis=0).T
    return data


def _extract_frequency_power_bands(freqs, values, relative=False):
    """
    Extracts power of six frequency bands from a spectral distribution.

    Args:
        freqs: vector containing the discrete frequencies
        values: vector containing the spectral distribution
        relative: boolean indicating if the power bands should be a probability distribution or absolute sums

    Returns:
        list containing the power bands (delta, theta, alpha, sigma, beta, low gamma)
    """
    if relative:
        total = np.sum(values)
    else:
        total = 1

    return [
        np.sum(values[(freqs >= 0.3) & (freqs < 4)]) / total,
        np.sum(values[(freqs >= 4) & (freqs < 8)]) / total,
        np.sum(values[(freqs >= 8) & (freqs < 12)]) / total,
        np.sum(values[(freqs >= 12) & (freqs < 16)]) / total,
        np.sum(values[(freqs >= 16) & (freqs < 32)]) / total,
        np.sum(values[(freqs >= 32) & (freqs <= 50)]) / total,
    ]


def _power_spectral_density_single_epoch(epoch, num_segments=6, frequency=256):
    """
    Computes the power spectral density for a single epoch with Welch's method.

    Args:
        epoch: vector with EEG data from one epoch
        num_windows: number of segments used in Welch's method
        frequency: sampling frequency of the EEG in Hz

    Returns:
        frequency distribution
        power spectral density
    """
    return signal.welch(
        epoch,
        fs=frequency,
        nperseg=len(epoch) // num_segments,
        noverlap=0,
        window="hamming",
    )


def power_spectral_density(stage, bands=True, relative=False):
    """
    Computes the power spectral density for one sleep stage and (if bands is true) separates
    the spectrum into frequency bands.

    Args:
        stage: EEG data from one sleep stage (electrodes x epoch steps x epochs)
        bands: boolean indicating if PSD for frequency bands or complete PSD should be returned (changes output shape)
        relative: boolean idicating if the frequency power bands should be a probability distribution

    Returns:
        power spectral density (electrodes x epochs x amplitudes) or (electrodes x epochs x bands)
    """
    electrode_count = stage.shape[0]
    epoch_count = stage.shape[2]

    psd = []
    for electrode in range(electrode_count):
        # add a row for each electrode
        psd.append([])

        for epoch in range(epoch_count):
            # calculate PSD for the current epoch of the current electrode
            freq, amp = _power_spectral_density_single_epoch(stage[electrode, :, epoch])

            # add a column for each epoch
            psd[-1].append([])

            if bands:
                # add list with the power band values as the third dimension
                power_bands = _extract_frequency_power_bands(
                    freq, amp, relative=relative
                )
                psd[-1][-1] = power_bands
            else:
                # add amplitude array as the third dimension
                psd[-1][-1] = amp

    return np.array(psd)


def shannon_entropy(signal, normalize=True):
    """
    Computes the shannon entropy for a single epoch.

    Args:
        signal: vector containing the signal over which the entropy is computed
        normalize: boolean indicating if the entropy value should be normalized to the range between 0 and 1

    Returns:
        shannon entropy as float
    """
    # normalize the signal to a probability distribution
    signal /= np.sum(signal)
    # calculate shannon entropy
    entropy = -np.sum(signal * np.log(signal))

    if normalize:
        # normalize entropy to range between 0 and 1
        entropy /= np.log(len(signal))
    return entropy


def permutation_entropy(signal, order=3, delay=1, normalize=True):
    """
    Computes the permutation entropy for a single epoch.

    Args:
        signal: vector containing the signal over which the entropy is computed
        order: permutation window length
        delay: step between the permutation windows
        normalize: boolean indicating if the entropy value should be normalized to the range between 0 and 1

    returns:
        permutation entropy as float
    """
    window_count = len(signal) - (order - 1) * delay
    # partition the signal into windows of length order with step size delay
    windows = np.array(
        [signal[i * delay : i * delay + window_count] for i in range(order)]
    ).T

    # calculate element ranks for each window in ascending order
    # same values become distinct ranks corresponding to the order of appearance
    # e.g. window [4, 6, 1] becomes [1, 2, 0]
    ranks = np.array(
        [stats.rankdata(window, method="ordinal") - 1 for window in windows]
    )
    # get relative frequency of each unique rank in the data
    rel_permutation_counts = (
        np.unique(ranks, axis=0, return_counts=True)[1] / ranks.shape[0]
    )

    # calculate permutation entropy
    entropy = -np.sum([perm * np.log2(perm) for perm in rel_permutation_counts])

    if normalize:
        # normalize entropy to range between 0 and 1
        entropy /= np.log2(np.math.factorial(order))
    return entropy


def sample_entropy(signal, dimension=2, tolerance=0.2, only_last=True):
    # code of this function taken from https://github.com/nikdon/pyEntropy
    """Calculates the sample entropy of degree m of a signal.
    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.
    Args:
        signal: numpy array of time series
        dimension: length of longest template vector
        tolerance: tolerance (defaults to 0.1 * std(signal)))
    Returns:
        Array of sample entropies:
            SE[k] is ratio "#templates of length k+1" / "#templates of length k"
            where #templates of length 0" = n*(n - 1) / 2, by definition
    Note:
        The parameter 'dimension' is equal to m + 1 in Ref[1].
    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    # The code below follows the sample length convention of Ref [1] so:
    M = dimension - 1

    signal = np.array(signal)
    if tolerance is None:
        tolerance = 0.1 * np.std(signal)
    else:
        tolerance = tolerance * np.std(signal)

    n = len(signal)

    # Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    # Templates of length 0 matches by definition:
    Ntemp[0] = n * (n - 1) / 2

    for i in range(n - M - 1):
        template = signal[i : (i + M + 1)]  # We have 'M+1' elements in the template
        rem_signal = signal[i + 1 :]

        searchlist = np.nonzero(np.abs(rem_signal - template[0]) < tolerance)[0]

        go = len(searchlist) > 0

        length = 1

        Ntemp[length] += len(searchlist)

        while go:
            length += 1
            nextindxlist = searchlist + 1
            # Remove candidates too close to the end
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]
            nextcandidates = rem_signal[nextindxlist]
            hitlist = np.abs(nextcandidates - template[length - 1]) < tolerance
            searchlist = nextindxlist[hitlist]

            Ntemp[length] += np.sum(hitlist)

            go = any(hitlist) and length < M + 1

    sampen = -np.log(Ntemp[1:] / Ntemp[:-1])
    if only_last:
        return sampen[-1]
    else:
        return sampen


def _distance(x1, x2):
    return np.max(np.abs(x1 - x2))


def sample_entropy_slow(signal, dimension=2, tolerance=0.2):
    """
    Computes the sample entropy for a single epoch.

    Args:
        signal: vector containing the signal over which the entropy is computed
        dimension: length of the templates which are used during calculation
        tolerance: the maximum distance between two windows to allow a match

    Returns:
        sample entropy as float
    """
    tolerance = tolerance * np.std(signal)

    # generate windows with length dimension + 1 over the signal
    embeddings = np.array(
        [signal[i : i + dimension + 1] for i in range(len(signal) - dimension)]
    )
    # compute number of matches for windows with size dimension and size dimension + 1
    matches = np.sum(
        [
            np.sum(
                [
                    [
                        _distance(template[:-1], embedding[:-1]) <= tolerance,
                        _distance(template, embedding) <= tolerance,
                    ]
                    for j, embedding in enumerate(embeddings)
                    if i != j
                ],
                axis=0,
            )
            for i, template in enumerate(embeddings)
        ],
        axis=0,
    )

    # return negative ln of # matches for size dimension + 1 divided by # matches for size dimension
    return -np.log(matches[1] / matches[0])


def spectral_entropy(stage, method="shannon"):
    """
    Computes the spectral entropy for one sleep stage using the specified entropy method (permutation or shannon).

    Args:
        stage: EEG data over which the spectral entropy should be computed (electrodes x epoch steps x epochs)
        method: string indicating which entropy method to be used (shannon, permutation or sample)

    Returns:
        spectral entropy of the sleep stage (electrodes x epochs)
    """
    # compute power spectral density
    psd = power_spectral_density(stage, bands=False)

    electrode_count = psd.shape[0]
    epoch_count = psd.shape[1]

    spec_entropy = np.empty((electrode_count, epoch_count))
    for electrode in range(electrode_count):
        for epoch in range(epoch_count):
            if method.lower() == "shannon":
                # get PSD shannon entropy for the current electrode and epoch
                spec_entropy[electrode, epoch] = shannon_entropy(
                    psd[electrode, epoch], normalize=True
                )
            elif method.lower() == "permutation":
                # get PSD permutation entropy for the current electrode and epoch
                spec_entropy[electrode, epoch] = permutation_entropy(
                    psd[electrode, epoch], normalize=True
                )
            elif method.lower() == "sample":
                # get PSD sample entropy for the current electrode and epoch
                spec_entropy[electrode, epoch] = sample_entropy(psd[electrode, epoch])
            else:
                # unknown entropy method
                raise NotImplementedError(f'Entropy type "{method}" is unknown')
    return spec_entropy


def hurst_exponent(stage, frequency=256, freq_range=None):
    """
    Computes the Hurst exponent for one sleep stage with Anis-Lloyd-Peters correction.

    Args:
        stage: raw EEG data (electrodes x epoch steps x epochs)

    Returns:
        hurst exponent for each individual time series (electrodes x epochs)
    """
    if freq_range is not None:
        # band pass EEG data
        stage = filter_data(
            stage.transpose((0, 2, 1)).astype(float),
            frequency,
            freq_range[0],
            freq_range[1],
        ).transpose(0, 2, 1)

    hurst = np.empty((stage.shape[0], stage.shape[2]))
    for elec in range(stage.shape[0]):
        result = Parallel(n_jobs=-1)(
            delayed(complexity_hurst)(stage[elec, :, epoch])
            for epoch in range(stage.shape[2])
        )
        hurst[elec] = [r[0] for r in result]
    return hurst


def hurst_exponent_antropy(stage):
    hurst = np.empty((stage.shape[0], stage.shape[2]))
    for elec in range(stage.shape[0]):
        hurst[elec] = Parallel(n_jobs=-1)(
            delayed(detrended_fluctuation)(stage[elec, :, epoch])
            for epoch in range(stage.shape[2])
        )
    return hurst


def _compute_1_over_f(f, p, freq_range):
    fm = FOOOF()
    fm.fit(f, p, freq_range=freq_range)
    return fm.aperiodic_params_[1]


def fooof_1_over_f(stage, frequency=256, freq_range=[3, 35]):
    """
    Computes the 1/f activity (aperiodic component) using the FOOOF algorithm. The power spectrum
    is computed using Welch's method.

    Args:
        stage: raw EEG data (electrodes x epoch steps x epochs)
        frequency: the sampling frequency of the EEG
        freq_range: the frequency range to fit FOOOF on

    Returns:
        1/f activity of the EEG (electrodes x epochs)
    """
    f, psd = signal.welch(stage, frequency, axis=1)
    oof = np.empty((stage.shape[0], stage.shape[2]))
    for elec in range(stage.shape[0]):
        oof[elec] = Parallel(n_jobs=-1)(
            delayed(_compute_1_over_f)(f, psd[elec, :, epoch], freq_range)
            for epoch in range(stage.shape[2])
        )
    return oof


def _z1_chaos_test(x, sigma=1, rand_seed=0):
    np.random.seed(rand_seed)
    N = len(x)
    a = np.arange(N)
    t = np.arange(int(round(N / 10)))
    M = np.zeros(int(round(N / 10)))
    # Choose a coefficient c within the interval pi/5 to 3pi/5 to avoid
    # resonances. Do this 100 times.
    c = np.pi / 5 + np.random.random_sample(100) * 3 * np.pi / 5
    k_corr = np.zeros(100)

    for i in range(100):
        # Create a 2-d system driven by the data
        p = np.cumsum(x * np.cos(a * c[i]))
        q = np.cumsum(x * np.sin(a * c[i]))

        for n in range(int(round(N / 10))):
            # Calculate the (time-averaged) mean-square displacement,
            # subtracting a correction term (Gottwald & Melbourne, 2009)
            # and adding a noise term scaled by sigma (Dawes & Freeland, 2008)
            M[n] = (
                np.mean(
                    (p[n + 1 :] - p[: -n - 1]) ** 2 + (q[n + 1 :] - q[: -n - 1]) ** 2
                )
                - np.mean(x) ** 2 * (1 - np.cos(n * c[i])) / (1 - np.cos(c[i]))
                + sigma * (np.random.random() - 0.5)
            )

        k_corr[i], _ = stats.pearsonr(t, M)

    return np.median(k_corr)


def zero_one_chaos(stage):
    """
    Compute the 0-1 chaos test.

    Args:
        stage: raw EEG data (electrodes x epoch steps x epochs)

    Returns:
        0-1 chaos test of the EEG (electrodes x epochs)
    """
    zo = np.empty((stage.shape[0], stage.shape[2]))
    for elec in range(stage.shape[0]):
        zo[elec] = Parallel(n_jobs=-1)(
            delayed(_z1_chaos_test)(stage[elec, :, epoch])
            for epoch in range(stage.shape[2])
        )
    return zo


def avalanche_wrapper(x, frequency, max_iei, threshold):
    try:
        return detect_avalanches(x, frequency, max_iei=max_iei, threshold=threshold)
    except TypeError:
        return []


def compute_avalanches(stage, frequency=256):
    # channel-wise z-transform
    stage = (stage - stage.mean(axis=(1, 2), keepdims=True)) / stage.std(
        axis=(1, 2), keepdims=True
    )
    # detect avalanches
    result = Parallel(n_jobs=-1)(
        delayed(avalanche_wrapper)(
            stage[:, :, epoch], frequency, max_iei=0.024, threshold=2
        )
        for epoch in range(stage.shape[2])
    )
    avalanches = sum(
        [result[i][0] for i in range(len(result)) if len(result[i]) > 0], []
    )
    # compute branching ratio
    bran_rat = np.array(
        [
            [branching_ratio(result[i][1][ch], 5, 256) for i in range(len(result))]
            for ch in range(stage.shape[0])
        ]
    )
    return avalanches, bran_rat


def _compute_lziv(epoch):
    return lziv_complexity((epoch > np.median(epoch)).astype(int), normalize=True)


def compute_lziv(stage):
    lziv = np.empty((stage.shape[0], stage.shape[2]))
    for channel in range(stage.shape[0]):
        result = Parallel(n_jobs=-1)(
            delayed(_compute_lziv)(stage[channel, :, epoch])
            for epoch in range(stage.shape[2])
        )
        lziv[channel] = result
    return lziv
