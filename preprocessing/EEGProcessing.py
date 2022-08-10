import os
import glob
import numpy as np
from scipy import signal, stats
from fooof import FOOOF
from antropy import lziv_complexity, detrended_fluctuation
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


def power_spectral_density(stage, bands=True, frequency=256, freq_range=(0.5, 50)):
    """
    Computes the power spectral density for one sleep stage and (if bands is true) separates
    the spectrum into frequency bands. The power spectrum will be corrected for the 1/f-like
    aperiodic component.

    Args:
        stage: EEG data from one sleep stage (electrodes x epoch steps x epochs)
        bands: boolean indicating if PSD for frequency bands or complete PSD should be returned (changes output shape)
        frequency: sampling frequency of the EEG
        freq_range: frequency range to fit FOOOF on

    Returns:
        power spectral density (electrodes x epochs x amplitudes) or (electrodes x epochs x bands)
    """
    electrode_count = stage.shape[0]
    epoch_count = stage.shape[2]

    def _flat_spectrum(f, a, freq_range):
        fm = FOOOF(verbose=False)
        fm.fit(f, a, freq_range=freq_range)
        return fm

    freq, amp = signal.welch(stage, frequency, axis=1)

    amp = amp.transpose(0, 2, 1).reshape(-1, freq.shape[0])

    result = Parallel(n_jobs=-1)(
        delayed(_flat_spectrum)(freq, camp, freq_range) for camp in amp
    )
    freq = np.array([fm.freqs for fm in result]).reshape(
        electrode_count, epoch_count, -1
    )
    amp = np.array([fm._spectrum_flat for fm in result]).reshape(
        electrode_count, epoch_count, -1
    )

    if not bands:
        return amp

    result = np.empty((electrode_count, epoch_count, 6))
    result[:, :, 0] = (
        amp[(freq >= 0.5) & (freq < 4)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    result[:, :, 1] = (
        amp[(freq >= 4) & (freq < 8)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    result[:, :, 2] = (
        amp[(freq >= 8) & (freq < 12)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    result[:, :, 3] = (
        amp[(freq >= 12) & (freq < 16)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    result[:, :, 4] = (
        amp[(freq >= 16) & (freq < 32)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    result[:, :, 5] = (
        amp[(freq >= 32) & (freq < 50)]
        .reshape(electrode_count, epoch_count, -1)
        .sum(axis=-1)
    )
    return result


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
            elif method.lower() == "sample":
                # get PSD sample entropy for the current electrode and epoch
                spec_entropy[electrode, epoch] = sample_entropy(psd[electrode, epoch])
            else:
                # unknown entropy method
                raise NotImplementedError(f'Entropy type "{method}" is unknown')
    return spec_entropy


def compute_dfa(stage):
    """
    Compute DFA slope exponent.

    Args:
        stage: raw EEG data (electrodes x epoch steps x epochs)

    Returns:
        alpha exponent from DFA of the EEG (electrodes x epochs)
    """
    hurst = np.empty((stage.shape[0], stage.shape[2]))
    for elec in range(stage.shape[0]):
        hurst[elec] = Parallel(n_jobs=-1)(
            delayed(detrended_fluctuation)(stage[elec, :, epoch])
            for epoch in range(stage.shape[2])
        )
    return hurst


def _compute_1_over_f(f, p, freq_range):
    fm = FOOOF(verbose=False)
    fm.fit(f, p, freq_range=freq_range)
    return fm.aperiodic_params_[-1]


def fooof_1_over_f(stage, frequency=256, freq_range=[0.5, 50]):
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


def _compute_lziv(epoch):
    return lziv_complexity((epoch > np.median(epoch)).astype(int), normalize=True)


def compute_lziv(stage):
    """
    Computes the Lempel-Ziv complexity.

    Args:
        stage: raw EEG data (electrodes x epoch steps x epochs)

    Returns:
        Lempel-Ziv complexity of the EEG (electrodes x epochs)
    """
    lziv = np.empty((stage.shape[0], stage.shape[2]))
    for channel in range(stage.shape[0]):
        result = Parallel(n_jobs=-1)(
            delayed(_compute_lziv)(stage[channel, :, epoch])
            for epoch in range(stage.shape[2])
        )
        lziv[channel] = result
    return lziv
