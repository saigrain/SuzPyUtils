'''
periodograms.py: simple routines to explore time-series in the Fourier
domain.
'''

import pylab as plt
import numpy as np
from numpy.fft import rfft
from numpy.fft.helper import fftshift, fftfreq
import scipy.linalg as sl
from norm import *

small = np.MachAr().eps

def sinefit(time, data, err = None, fmin = None, fmax = None, \
            sampling = 'linear', nfreq = None, doplot = True):
    '''
    
    Least squares fit of sine curves to data:
        data = dc + amp * sin(2 * pi * freq * time + phase)
    Frequency by brute force, other pars linear.

    Calling sequence:
        pgram, best_val = \
            sinefit(time, data, err = None, fmin = None, fmax = None, \
                    sampling = 'log', nfreq = 500)
                    
    Inputs:
        time: observation times, numpy array
        data: observable values, numpy array
        err: uncertainties on observables, if available, numpy array or scalar
        fmin: minimum frequency, scalar
            default: 1. / (max(time) - min(time)
        fmax: maximum frequency, scalar
            default: 0.5 / min(time[1:] - time[:-1])
            note that time is assumed to be sorted
        nfreq: number of frequencies, scalar
            default: int(fmax/fmin)
        sampling: frequency sampling, string
            default: 'linear',
            other options: 'log', 'inverse'
	doplot: set to False to suppress plot
	    default: True

    Outputs:
        pgram: periodogram, tuple consisiting of: 
            freq: frequency values, numpy array
            rchi2: reduced chi2 values as a function of frequency for best
                amplitude, phase and zero-point, numpy array
            amp: best amplitude as a function of frequency, numpy array
            phase: best phase as a function of frequency, numpy array
            dc: best zero-point as a function of frequency, numpy array
            Note that the first element in each output array corresponds
            to the constant model (freq = 0, amp = 0, phase = 0, dc =
            weighted mean of data), so if you requested nfreq frequencies,
            the output arrays have nfreq+1 elements
        best_val: best values (minimum rchi2), tuple consisting of:
            (best_rchi2, best_freq, best_amp, best_phase, best_dc)
    '''
    
    if fmin is None:
        fmin = 1. / (mymax(time) - mymin(time))
    if fmax is None:
        fmax = 0.5 / mymin(time[1:] - time[:-1])
    if nfreq is None:
        nfreq = int(fmax/fmin)
    if sampling is 'log':
        lfmin, lfmax = np.log10(fmin), np.log10(fmax)
        lfreq = np.r_[lfmin:lfmax:nfreq*1j]
        freq = 10.0**lfreq
    elif sampling is 'inverse':
        pmax, pmin = 1. / fmin, 1. / fmax
        per = np.r_[pmin:pmax:nfreq*1j]
        freq = np.sort(1. / per)
    else:
        freq = np.r_[fmin:fmax:nfreq*1j]
    n = len(time)
    if err is None:
        w = np.ones(n)
    else:
        w = np.ones(n) / err**2
    freq = np.append(0, freq)
    rchi2 = np.zeros(nfreq+1) + np.nan
    amp = np.zeros(nfreq+1) + np.nan
    phase = np.zeros(nfreq+1) + np.nan
    dc = np.zeros(nfreq+1) + np.nan
    sumw = w.sum()
    dataw = data * w
    sumdw = dataw.sum()
    meanw = sumdw / sumw
    dc[0] = meanw
    ndof = float(len(data)-1)
    rchi2[0] = ((data - meanw)**2 * w).sum() / ndof
    amp[0] = 0.
    phase[0] = 0.
    a = np.matrix(np.empty((3,3)))
    a[2,2] = sumw
    b = np.empty(3)
    b[2] = sumdw
    ndof -= 3
    for i in np.arange(nfreq):
        arg = 2 * np.pi * freq[i] * time
        cosarg = np.cos(arg)
        sinarg = np.sin(arg)
        a[0,0] = (sinarg**2*w).sum()
        a[0,1] = (cosarg*sinarg*w).sum()
        a[0,2] = (sinarg*w).sum()
        a[1,0] = a[0,1]
        a[1,1] = (cosarg**2*w).sum()
        a[1,2] = (cosarg*w).sum()
        a[2,0] = a[0,2]
        a[2,1] = a[1,2]
        a[abs(a)<=small] = 0.
        if sl.det(a) < small: continue
        b[0] = (dataw*sinarg).sum()
        b[1] = (dataw*cosarg).sum()
        c = sl.solve(a, b)
        amp[i+1] = np.sqrt(c[0]**2+c[1]**2)
        phase[i+1] = np.arctan2(c[1],c[0])
        dc[i+1] = c[2]
        f = amp[i+1] * np.sin(arg + phase[i+1]) + dc[i+1]
        rchi2[i+1] = ((data - f)**2 * w).sum() / ndof
    best_rchi2 = mymin(rchi2)
    i = np.where(rchi2 == best_rchi2)[0]
    best_freq = freq[i]
    best_per = 1./best_freq
    best_amp = amp[i]
    best_phase = phase[i]
    best_dc = dc[i]
    if doplot == True:
        ttl = '%.3f %.3f %.3f %.5f %.3f %.5f' % \
	    (best_rchi2, best_per, best_freq, best_amp, best_phase, best_dc)
        print ttl
	plt.figure(figsize = (6,7.5), edgecolor = 'w')
	plt.subplot(311)
	if err is None:
	    plt.plot(time, data, 'k.')
	else:
	    plt.errorbar(time, data, err, fmt = 'k.', capsize = 0)
	plt.xlabel('time')
	plt.ylabel('data')
	plt.title(ttl)
	n_p = freq[i] * (mymax(time) - mymin(time)) 
	if n_p < 20:
	    x = np.r_[mymin(time):mymax(time):101j]
	    plt.plot(x, best_amp * np.sin(2 * np.pi * best_freq * x + best_phase) + \
		     best_dc, 'r-')
	plt.xlim(mymin(time), mymax(time))
	plt.subplot(312)
	if (sampling is 'log') + (sampling is 'inverse'):
	    plt.semilogx()
	plt.axvline(best_freq, c = 'r')
	plt.plot(freq[1:], rchi2[0] - rchi2[1:], 'k-')
	plt.xlabel('frequency')
	plt.ylabel('$\delta \chi^2$')
	plt.xlim(mymin(freq), mymax(freq))
	plt.subplot(313)
	ph = (time % best_per) / best_per
	if err is None:
	    plt.plot(ph, data, 'k.')
	else:
	    plt.errorbar(ph, data, err, fmt = 'k.', capsize = 0)
	x = np.r_[0:best_per:101j]
	y = best_amp * np.sin(2 * np.pi * x / best_per + best_phase) + best_dc
	plt.plot(x/best_per, y, 'r')
	plt.xlim(0,1)
	plt.xlabel('phase')
	plt.ylabel('data')
    return (rchi2, freq, amp, phase, dc), \
	(best_rchi2, best_freq, best_amp, best_phase, best_dc)

def DftPowerSpectrum(x, dt = 1, norm = False, doplot = False):
    '''
    Compute power spectrum (squared modulus of discrete Fourier
    transform) of 1-D vector x, assumed to be sampled regularly with
    sampling interval dt, and the corresponding frequency array in
    physical units (postive frequencies only). If norm is True, the
    power specturm is "normalised", i.e. multiplied by 4, so that a
    sinusoid with semi-amplitude A gives rise to a peak of height A**2
    in the power spectrum.
    '''
    n = x.size
    amp = abs(rfft(x)) 
    ps = amp**2 / float(n)
    if norm == True: ps *= 4 / float(n)
    freq = np.arange(n/2+1) / float(n*dt)
    if doplot == True:
	plt.figure(figsize = (6,5))
	plt.subplot(211)
        plt.title('DFT power spectrum')
	t = np.arange(n)*dt
	plt.plot(t, x, 'k-')
	plt.xlabel('time')
	plt.ylabel('data')
	plt.xlim(0,n*dt)
	plt.subplot(212)
    plt.plot(freq, ps, 'k-')
    plt.xlabel('frequency')
    plt.ylabel('power')
    return ps, freq

def AcfPeriodogram(x, dt = 1, norm = False, doplot = False, \
                       smooth = False, box = 0.01):
    '''
    Compute periodogram (power spectrum of ACF) of a 1-D vector x,
    assumed to be sampled regularly with sampling interval dt, and the
    corresponding frequency array in physical units (postive
    frequencies only). The ACF is computed up to lags of N/4 where N
    is the length of the input array. If smooth is True, the ACF is
    multiplied by sinc(pi/box) before taking the power spectrum. This
    is equivalent to smoothing the power specturm by convolving it a
    top-hat function of width (box/dt) in the frequency domain.
    '''
    maxl = min(len(x), max(len(x)/4, 50))
    f = plt.figure()
    lag, corr, line, ax = plt.acorr(x, maxlags = maxl, normed = True)
    plt.close(f)
    if smooth == True:
        corr *= np.sinc(np.pi/box)
    pgram, freq = DftPowerSpectrum(corr, dt, doplot = False)
    if doplot == True:
        plt.figure(figsize = (6,7.5))
        plt.subplot(311)
        plt.title('ACF periodogram')
        t = np.arange(len(x))*dt
        plt.plot(t, x, 'k-')
        plt.xlabel('time')
        plt.ylabel('data')
        plt.xlim(0,len(x)*dt)
        plt.subplot(312)
        l = lag >= 0
        plt.plot(lag[l]*dt, corr[l], 'k-')
        plt.ylabel('ACF')
        plt.xlabel('lag (time units)')
        plt.xlim(0,lag.max()*dt)
        plt.subplot(313)
        plt.plot(freq, pgram, 'k-')
        plt.xlabel('frequency')
        if smooth == True:
            plt.ylabel('amplitude (smoothed)')
        else:
            plt.ylabel('amplitude')
        plt.xlim(freq.min(),freq.max())
    return (corr, lag*dt), (pgram, freq)

