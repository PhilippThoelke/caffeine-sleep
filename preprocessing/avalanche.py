"""Avalanche-based measures
"""

from numpy import (arange, argmax, array, asarray, ceil, concatenate, diff, dtype, histogram, interp, ix_, linspace, mean, nanmean, newaxis, ones, ptp, round, sign, square, std, sum, vstack, where, zeros)
import numpy.random as random
from scipy.optimize import curve_fit

import powerlaw
#import mrestimator as mre


def binarized_events(data, threshold=3, null_value=0):
    """Convert time series to binary where 1s are the locations of peaks 
    above threshold.
    
    Parameters
    ----------
    data : 2d array (dtype='float')
        Raw (cleaned) signal, channel x sample.
    threshold : float
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events.
    null_value : int
        The value assigned to subthreshold values. Usually 0 or -1.

    Returns
    -------
    events : 2d array of same shape as data, dtype=int
        Time series of zeros with ones at the event peaks.
    """
    # Detect events as the locations of peaks above threshold
    thresh_per_chan = ( mean(data, axis=1, keepdims=True) 
                        + threshold * std(data, axis=1, keepdims=True) )
    above_thresh = data > thresh_per_chan
    events = ones(data.shape) * null_value
    for chan in range(data.shape[0]):
        start_end = _detect_start_end(above_thresh[chan, :])
        for i in start_end:
            idx_peak = argmax(data[chan, i[0]:i[1]])
            events[chan, i[0] + idx_peak] = 1
            
    return events

def detect_avalanches(data, s_freq, time=None, max_iei=.004, threshold=3):
    """Detect avalanche in coarse-grained electrophsiological 
    recordings.
    
    Parameters
    ----------
    data : 2d array (dtype='float' or 'int')
        Raw (cleaned) signal, channel x sample. If dtype='int', the array is 
        taken as the binarized event time series.
    s_freq : int
        Sampling frequency.
    time : 1d array (dtype='float')
        Vector with the time points for each sample. If None, avalanche start
        and end times will be returned in seconds since the start of data.
    max_iei : float, : .004
        Duration, in seconds, of the maximum inter-event interval within which 
        suvccessive events are counted as belonging to the same avalanche.
    threshold : float
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events.      
        
    Returns
    -------
    avalanches : list of dict
        List of detected avalanches as dictionaries, with keys:
            - start_time : float
                Start time, in seconds since recording start.
            - end_time : float
                End time in seconds since recording start.
            - size : int
                Number of events (over all channels).
            - dur_sec : float
                Duration, in seconds, from first to last event in the 
                avalanche.
            - dur_bin : int
                Duration, in number of bins, rounded up.
            - n_chan : int
                Number of channels containing events.
            - profile : 1d array
                Number of events (active channels) in each successive time bin.
    events : 2d array of same shape as data, dtype=int
        Time series of zeros with ones at the event peaks.
                
    Notes
    -----
    Events detected at the same time in a pair of gradiometers are 
    counted as one. 
    
    References
    ----------
    ???Shriki et al. (2013) J Neurosci 33(16), 7079–90.
    """
    #z_data = ( (data - mean(data, axis=0, keepdims=True)) 
    #          / std(data, axis=0, keepdims=True) )
    
    # Compute minimum interval, in samples
    max_iei_len = int(max_iei * s_freq)
    
    if data.dtype is not dtype(int):
        # Detect events as the locations of peaks above threshold
        events_all_chan = binarized_events(data, threshold)
        
    # Collapse all channels into one
    events_one_chan = sum(events_all_chan, axis=0)
    
    # Obtain inter-event intervals
    idx_events = where(asarray(events_one_chan, dtype=bool))[0]
    iei = diff(idx_events)
    
    # Detect avalanches
    avalanches = []
    avl_start = idx_events[0]
    for i, j in enumerate(iei):
        if j > max_iei_len:
            avl_end = idx_events[i] + 1
            profile_raw = events_one_chan[avl_start:avl_end]
            if time is not None:
                start_time = time[avl_start]
                end_time = time[avl_end]
            else:
                start_time = avl_start / s_freq
                end_time = avl_end / s_freq
            size = int(sum(profile_raw))
            dur_sec = (avl_end - avl_start) / s_freq
            dur_bin = int(ceil(dur_sec / max_iei))
            n_chan = int(sum(sign(sum(events_all_chan[:, avl_start:avl_end],
                                     axis=0))))
            profile = time_bin_events(profile_raw, max_iei_len)
            avl = {'start_time': start_time,
                   'end_time': end_time,
                   'size': size,
                   'dur_sec': dur_sec,
                   'dur_bin': dur_bin,
                   'n_chan': n_chan,
                   'profile': profile}
            avalanches.append(avl)
            avl_start = idx_events[i + 1]
    # if the record cuts off in the middle of an avalanche, we discard the 
    # last avalanche

    return avalanches, events_all_chan, events_one_chan     

def fit_powerlaw(data, pmin=0.005):
    """Fit data to a power law and compare the fit to similar 
    distributions.

    Parameters
    ----------
    data : array-like
        List of avalanche features, e.g. avalanche sizes.
    pmin : float
        The probability density cutoff for the histogram. Used to decide 
        the maximum value for the fitted data.

    Returns
    -------
    power_law_alpha : float
        Slope of the power law.
    xmin: float
        The minimum value of the fitted data.
    xmax : TYPE
        The maximum value of the fitted data.
    power_law_sigma : float
        Standard error of the slope.
    truncated_power_law_alpha : float
        Slope of the truncated power law.
    truncated_power_law_xmin : float
        Minimum value of the trauncated power law.
    truncated_power_law_xmax : float
        Maximum value of the trauncated power law.
    PvE : tuple of float
        The loglikelihood ratio R and probability p for power law versus 
        exponential fit.
    PvL : tuple of float
        The loglikelihood ratio R and probability p for power law versus 
        lognormal fit.
    PvT : tuple of float
        The loglikelihood ratio R and probability p for power law versus 
        truncated power law fit.
    TvE : tuple of float
        The loglikelihood ratio R and probability p for truncated power law 
        versus exponential fit.
        exponential fit.
    TvL : tuple of float
        The loglikelihood ratio R and probability p for truncated power law 
        versus lognormal fit.

    Notes
    -----
    From Alstott et al. (2014): R is the loglikelihood ratio between the two 
    candidate distributions. This number will be positive if the data is 
    more likely in the first distribution, and negative if the data is 
    more likely in the second distribution. The significance value for 
    that direction is p.
    
    References
    ----------
    Alstott et al. (2014) PLoS one 9(1), e85777.
    Clauset et al. (2009) SIAM review 51(4), 661-703.

    Acknowledgements
    ----------------
    This function was adapted from code by Marzieh Zare.
    """
    pdf, bins = histogram(data, bins=arange(min(data), max(data) + 1), 
                          density=True)
    xmax = min(bins[1:][pdf < pmin]) - 1
    fit = powerlaw.Fit(data, discrete=True, xmax=xmax, verbose=False)

    PvE = fit.distribution_compare('power_law', 'exponential', 
                                   normalized_ratio=True)
    PvL = fit.distribution_compare('power_law', 'lognormal', 
                                   normalized_ratio=True)
    PvT = fit.distribution_compare('power_law', 'truncated_power_law', 
                                   normalized_ratio=True)

    TvE = fit.distribution_compare('truncated_power_law', 'exponential', 
                                   normalized_ratio=True)
    TvL = fit.distribution_compare('truncated_power_law', 'lognormal', 
                                   normalized_ratio=True)

    return (pdf,
            fit.power_law.alpha, 
            fit.xmin, 
            fit.xmax, 
            fit.power_law.sigma,
            fit.truncated_power_law.alpha, 
            fit.truncated_power_law.xmin, 
            fit.truncated_power_law.xmax, 
            PvE, PvL, PvT, TvE, TvL)

def fit_third_exponent(sizes, durations):
    """Estimate the third exponent from avalanche sizes and durations, using
    nonlinear least-squares fitting.

    Parameters
    ----------
    sizes : 1d array
        Sequence of avalanche sizes.
    durations : 1d array
        Corresponding sequence of avalanche durations.

    Returns
    -------
    third : float
        Estimated 'third' power-law exponent.
    """
    def func(x, a, b):
        return a * x + b
    popt, _ = curve_fit(func, sizes, durations)
    return popt[0]
    

def dcc(tau, alpha, third):
    """The deviation from criticality coefficient (DCC) measures the 
    degree of deviation of the empirical avalanche power law exponents 
    from the "crackling noise" scaling relation, which holds broadly 
    for avalanche-critical systems.

    Parameters
    ----------
    tau : float
        The empirical power law exponent of the avalanche size distribution.
    alpha : float
        The empirical power law exponent of the avalanche duration 
        distribution.
    third : float
        The empirical power law exponent of the relation between avalanche
        size and duration.

    Returns
    -------
    dcc : float
        The deviation from criticality coefficient.
        
    References
    ----------
    Ma et al. (2019) Neuron 104(4), 655-64.
    Friedman et al. (2012) PRL 108, 208102.
    """
    return abs( (alpha - 1) / (tau - 1) - third )

def select_avalanches(avalanches, size=None, dur_sec=None, dur_bin=None, 
                      n_chan=None, min_n_dur=None):
    """From a list of avalanches, return a selection according to specified
    criteria.

    Parameters
    ----------
    avalanches : list of dict
        Detected avalanches.
    size : tuple of int
        Minimum and maximum size of avalanches.
    dur_sec : tuple of float
        Minimum and maximum duration of avalanches, in seconds.
    dur_bin : tuple of int
        Minimum and maximum duration of avalanches, in time bins.
    n_chan : tuplr of int
        Minimum and maximum number of channels spanned by single avalanches.
    min_n_dur : int
        Minimum number of avalanches of a given duration. For every avalanche 
        duration T (in time bins), if the list contains less than 
        `min_n_dur` avalanches of duration T, then all avalanches of 
        duration T are discarded. Useful for shape collapse analysis.


    Returns
    -------
    selected : list of dict
        The selected avalanches.
    removed_durations : tuple of int
        If `min_n_dur` is not None and leads to the discarding of avalanches 
        by duration, the discarded durations will be listed here.
    """
    pass

def shape_collapse_error(avalanches, gamma, event_type, interp_n=1000):
    """Estimate the error of the shape collapse given a scaling factor
    gamma.
    
    Parameters
    ----------
    avalanches : list of dict
        Detected avalanches.
    gamma : float
        Scaling factor, corresponding to the critical exponent for the shape
        collapse.
    interp_n : int
        Number of points for interpolating avalanche profiles.

    Returns
    -------
    collapse_error : float
        The error of the shape collapse. A good collapse should minimize it.

    References
    ----------
    Marshall et al. (2016) Front Physiol 7, 250.
    Friedman et al. (2012) PRL 108, 208102.
    """
    # Calculate the average profile for each avalanche duration T
    T_set = sorted(set([x['dur_bin'] for x in avalanches]))
    profiles = zeros((len(T_set), interp_n))
    for i, T in enumerate(T_set):
        avls_T = [x for x in avalanches if x['dur_bin'] == T]
        avg_profile = mean(vstack([x['profile'] for x in avls_T]), axis=0)
        norm_avg_profile = ( asarray(avg_profile, dtype='float') 
                            / float(avg_profile.max()) )
        profiles[i, :] = interp(linspace(0, 1, interp_n), 
                                arange(T), norm_avg_profile)
    
    # Rescale the profiles (y-axis) by T**gamma
    rescaling_vector = array(T_set)**gamma
    rescaling_vector = rescaling_vector[..., newaxis]
    rescaled_profiles = rescaling_vector * profiles
    
    # Calculate collapse error
    mean_profile = mean(rescaled_profiles, axis=0, keepdims=True)
    variance = sum(square(rescaled_profiles - mean_profile), axis=0)
    squared_span = square(ptp(rescaled_profiles))
    collapse_error = mean(variance) / squared_span
    
    return collapse_error

def lattice_search(func, func_args, param_pos, step_0, n_levels=3, 
                   criterion='min'):
    """Systematically search a function's solution space for a minimum 
    (maximum) value by sweeping a parameter of the function at greater and 
    greater precision.

    Parameters
    ----------
    func : function
        The function on which to perform the search.
    func_args : list
        Arguments of the function. For the parameter to be sweeped, enter 
        the range of parameter values (inclusive) as a tuple, 
        e.g. [arg1, (0, 5), arg3, arg4]
    param_pos : int
        The index, in func_args, of the parameter to be swept.
    step_0 : float
        The coarsest step of the search. The first sweep will iterate through 
        the range given in `func_args` at intervals of length `step_0`.
    n_levels : int
        Number of sweeps. Each succesive sweep is at 10x precision. 
    criterion : str
        The criterion to be fulfilled by the search. 'min' or 'max'.
        
    Returns
    -------
    optimum : float
        The value of the swept parameter which optimizes the solution.

    References
    ----------
    Marshall et al. (2016) Front Physiol 7, 250.
    """
    start, end = func_args[param_pos]
    solutions = zeros((end - start) // step_0 + 1)
    for i, j in enumerate(arange(start, end + step_0, step_0)):
        func_args[param_pos] = j
        solutions[i] = func(*func_args)        
    if 'min' == criterion:
        optimum = min(solutions)
    elif 'max' == criterion:
        optimum = max(solutions)
        
    iteration = 1
    step_n = step_0
    while iteration <= n_levels:
        range_n = (max(optimum - step_n, start), 
                   min(optimum + step_n, end))
        step_n = step_n / 10
        solutions = zeros((range_n[1] - range_n[0]) // step_n + 1)
        for i, j in enumerate(arange(range_n[0], range_n[1] + step_n, step_n)):
            func_args[param_pos] = j
            solutions[i] = func(*func_args)        
        if 'min' == criterion:
            optimum = min(solutions)
        elif 'max' == criterion:
            optimum = max(solutions)
        iteration += 1
        
    return optimum
            

def dcc_collapse(gamma, third):
    """The deviation from criticality coefficient for the scaling relation
    between the shape collapse exponent and the so-called "third" avalanche 
    exponent.

    Parameters
    ----------
    gamma : float
        The empirical power law exponent for the shape collapse.
    third : float
        The empirical power law exponent of the relation between avalanche
        size and duration.

    Returns
    -------
    dcc_collapse : float
        The deviation from criticality coefficient for shape collapse.
        
    References
    ----------
    Friedman et al. (2012) PRL 108, 208102.
    """
    return abs( (third - 1) - gamma )

def branching_ratio(events, time_bin_size, s_freq):
    """Calculate the branching ratio, which is the average number of events
    generated in the next time bin by a single event, evaluated for every bin 
    containing at least one event.

    Parameters
    ----------
    events : ndarray (dtype=int)
        Signal with number of events occuring at every time step. Can be 2d 
        (chan x time, with binary values) or 1d (time, with +int values).
    time_bin_size : float
        Length of time bin in seconds.
    s_freq : int
        Sampling frequency of `events`.

    Returns
    -------
    br : float (> 0)
        The branching ratio.
    
    References
    ----------
    Beggs & Plenz (2003) J Neurosci 23(35), 11167-77.
    Priesemann et al. (2014) Front Syst Neurosci 24, 108.
    """
    bin_len = int(time_bin_size * s_freq)
    events = time_bin_events(events, bin_len)    
    # nanmean excludes instances where the first bin contains no spikes 
    # (resulting in inf)
    br = nanmean(events[1:] / events[:-1])
    
    return br

def mr_branching_parameter(events):
    
    
    pass

def uniform_chi(events):
    pass

def dfa():
    # dfa on raw signal
    # dfa on events signal
    # narrowband dfa
    pass

def shew_kappa():
    pass

def fano_factor():
    pass

def time_bin_events(events, bin_len):
    """Compress a time series of events per time point into events per time 
    bin.

    Parameters
    ----------
    events : ndarray (dtype=int)
        Signal with number of events occuring at every time step. Can be 2d 
        (chan x time, with binary values) or 1d (time, with +int values).
    bin_len : int
        Length of time bin, in samples.

    Returns
    -------
    binned : 1d array (dtype=int)
        Time series of number of events per consecutive time bin.
    """
    if events.squeeze().ndim == 2:
        events = sum(events, axis=0)    
    padded = concatenate((events, zeros(len(events) % bin_len)))
    binned = sum(padded.reshape(int(len(padded) / bin_len), bin_len), axis=1)
    
    return binned

def _detect_start_end(true_values):
    """From ndarray of bool values, return intervals of True values.

    Parameters
    ----------
    true_values : ndarray (dtype='bool')
        array with bool values

    Returns
    -------
    ndarray (dtype='int')
        N x 2 matrix with starting and ending times.
        
    Notes
    -----
    Function borrowed from Wonambi: https://github.com/wonambi-python/wonambi
    """
    neg = zeros((1), dtype='bool')
    int_values = asarray(concatenate((neg, true_values[:-1], neg)), 
                         dtype='int')
    # must discard last value to avoid axis out of bounds
    cross_threshold = diff(int_values)

    event_starts = where(cross_threshold == 1)[0]
    event_ends = where(cross_threshold == -1)[0]

    if len(event_starts):
        events = vstack((event_starts, event_ends)).T

    else:
        events = None

    return events

# def synthetic_avalanches(duration=60, n_chan=12, s_freq=100, frequency=1.0, 
#                          m=1.0, m_sd=0.5, rand_seed=1):
#     # Note to self: Gao2017, Medel2020 describe a synthetic LFP model which 
#     # may be relevant
    
#     random.seed(rand_seed)
#     dat_len = int(duration * s_freq)
#     events = zeros((n_chan, dat_len))
#     n_avl = int(duration * frequency)
#     avl_starts_x = random.randint(low=0, high=dat_len, size=n_avl)
#     avl_starts_y = random.randint(low=0, high=n_chan, size=n_avl)
#     events[ix_(avl_starts_y, avl_starts_x)] = 1
#     for t in range(dat_len - 1):
#         # m_t at each time t is drawn from a normal distribution centered at m
#         m_t = max(0, random.normal(loc=m, scale=m_sd))
#         n_new_events = int(round(sum(events[:, 1]) * m_t))
#         idx_new_events = random.randint(low=0, high=n_chan - 1, 
#                                         size=n_new_events)
#         events[idx_new_events, t + 1] = 1
    
#     return events
        
        

            
        
    
    