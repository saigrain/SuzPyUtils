'''
Compute light and RV curves for spotted stars. Spots are simulated
either using the Dorren (1989) formalism (more exact but slow) or the
simplified formalism of Aigrain, Pont & Zucker (2012, more approximate
but faster). Also computes an estimate of the bisector span but I'm
not so confident about that. Also includes routines to plot the
time-series and their amplitude spectra (from sine-fitting). The code
includes routines to compute and plot regularly sampled time-series,
or data with a time-sampling more typical of ground-based observations
-- or the user can produce their own set of time values and feed them
in.
'''

import numpy as np
import pylab as plt
import scipy.interpolate as sciint
import numpy.random as npran
from multiplot import *
from norm import *
from periodograms import sinefit

RSUN = 6.96e8
DAY2S = 86400.0
DEG2RAD = np.pi / 180.0

class param():
    """Holds parameters for spots on a given star"""
    def __init__(self, nspot):
        '''Generate fiducial parameter set for nspot spots. This can
        then be modified by the user.'''
        self.nspot = nspot
        self.rstar = np.ones(nspot) * 1.0
        self.incl = np.ones(nspot) *  np.pi / 2.
        self.u = np.ones(nspot) * 0.0
        self.cspot = np.ones(nspot) * 1.0
        self.cfac = np.ones(nspot) * 0.1
        self.Q = np.ones(nspot) * 10.0
        self.vconv = np.ones(nspot) * 200.0
        self.period = np.ones(nspot) * 5.0
        self.amax = np.ones(nspot) * 0.01
        self.decay = np.zeros(nspot)
        self.pk = np.zeros(nspot)
        self.phase = np.ones(nspot) * np.pi
        self.lat = np.zeros(nspot)

    def calci(self, time, i):
        '''Calculations for one spot'''
        # Spot area
        if (self.pk[i] == 0) + (self.decay[i] == 0):
            area = np.ones(len(time)) * self.amax[i]
        else:
            area = self.amax[i] * \
                np.exp(-(time - self.pk[i])**2 / 2. / self.decay[i]**2)
        # Fore-shortening 
        long = 2 * np.pi * time / self.period[i] + self.phase[i]
        mu = np.cos(self.incl[i]) * np.sin(self.lat[i]) + \
            np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.cos(long)
        # Projected area 
        proj = area * mu
        proj[mu < 0] = 0
        # Flux
        if self.u[i] != 0:
            # Finite size spot with limb darkening (slow)
            N = len(time)
            spot = np.zeros(N)
            for j in np.arange(N):
                spot[j] = dorren_F(self.u[i], self.u[i], 1-self.cspot[i], \
                                       np.arcsin(np.sqrt(area[j])), \
                                       np.arccos(mu[j]))
        else:
            # Point-like spot without limb darkening
            spot = - proj * self.cspot[i]
        fac = proj * self.Q[i] * self.cfac[i] * (1 - mu)
        dF = np.copy(spot) # + fac
        # RV
        veq = 2 * np.pi * self.rstar[i] * RSUN / self.period[i] / DAY2S
        spot *= veq * np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.sin(long)
        fac = proj * self.Q[i] * mu * self.vconv[i]
        dRV = spot + fac
        bis = dRV * np.cos(long)
        return dF, dRV, bis

    def calc(self, time):
        '''Calculations for all spots'''
        N = len(time)
        M = len(self.lat)
        dF = np.zeros((M, N))
        dRV = np.zeros((M, N))
        bis = np.zeros((M, N))
        for i in np.arange(M):
            dFi, dRVi, bisi = self.calci(time, i)
            dF[i,:] = dFi
            dRV[i,:] = dRVi
            bis[i,:] = bisi
        return dF, dRV, bis

    def calci_pos(self, time, i):
        '''Calculations for one spot'''
        # Spot area
        try:
            am = self.amax[i]
        except AttributeError:
            am = self.alphamax[i]
        if (self.pk[i] == 0) + (self.decay[i] == 0):
            area = np.ones(len(time)) * am
        else:
            area = am * \
                np.exp(-(time - self.pk[i])**2 / 2. / self.decay[i]**2)
        # Fore-shortening 
        long = 2 * np.pi * time / self.period[i] + self.phase[i]
        mu = np.cos(self.incl[i]) * np.sin(self.lat[i]) + \
            np.sin(self.incl[i]) * np.cos(self.lat[i]) * np.cos(long)
        return area, mu, self.lat[i]

    def calc_pos(self, time):
        '''Calculations for all spots'''
        N = len(time)
        M = len(self.lat)
        area = np.zeros((M, N))
        mu = np.zeros((M, N))
        lat = np.zeros(M)
        for i in np.arange(M):
            dum1, dum2, dum3 = self.calci_pos(time, i)
            area[i,:] = dum1
            mu[i,:] = dum2
            lat[i] = dum3
        return area, mu, lat

def genTSreg(pars, nper = 20, npper = 1000, sum = False):
    '''Generate regularly sampled light curve, RV and bisector curves
    lasting nper periods, with npper points per period, for specified
    set of spot parameters, including evolving spots.'''
    permean = np.mean(pars.period)
    tmin = 0.0
    tmax = permean * nper 
    time = np.r_[tmin:tmax:permean/float(npper)]
    N = len(time)
    dF, dRV, bis = pars.calc(time)
    if sum == True:
        dF = np.reshape(np.sum(dF, 0), (1, N))
        dRV = np.reshape(np.sum(dRV, 0), (1, N))
        bis = np.reshape(np.sum(bis, 0), (1, N))
    return time, dF, dRV, bis

def genPosreg(pars, nper = 20, npper = 1000):
    '''Generate regularly sampled position curves
    lasting nper periods, with npper points per period, for specified
    set of spot parameters, including evolving spots.'''
    permean = np.mean(pars.period)
    tmin = 0.0
    tmax = permean * nper 
    time = np.r_[tmin:tmax:permean/float(npper)]
    N = len(time)
    area, mu, lat = pars.calc_pos(time)
    return time, area, mu, lat

def plotTS(time, y1, y2, y3 = None, figno = 1, discrete = True, \
                  savefig = None, period = None, xper = False):
    '''Plot light and RV curve(s)'''
    M, N = np.shape(y1)
    fac1 = 100
    fac2 = 1
    ls = ['-','--','-.',':']
    mrk = ['.',',','+','x']
    col = ['k','c','m','grey']
    if discrete == True:
        m1 = np.copy(mrk)
    else:
        m1 = np.copy(ls)
    if (xper == True) * (period != None):
        tt = time / period - 0.5
        xrange = [-0.5,0.5]
        xttl = 'Phase'
    else:
        tt = time
        xrange = min(time), max(time)
        xttl = 'time (days)'
    if y3 == None:
        ny = 2
    else:
        ny = 3
    ee = dofig(figno, 1, ny, aspect = 1)

    ax1 = doaxes(ee, 1, ny, 0, 0)
    plt.setp(ax1.get_xticklabels(), visible = False)
    for i in np.arange(M):
        plt.plot(tt, y1[i,:] * fac1, m1[i], c = col[i])
    plt.ylabel(r"$\Delta\Psi$ (\%)")
    ymin = y1.min() * fac1
    ymax = y1.max() * fac1
    yr = ymax - ymin
    plt.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

    ax2 = doaxes(ee, 1, ny, 0, 1, sharex = ax1)
    for i in np.arange(M):
        plt.plot(tt, y2[i,:] * fac2, m1[i], c = col[i])
    plt.ylabel(r"$\Delta V$ (m/s)")
    ymin = y2.min() * fac2
    ymax = y2.max() * fac2
    yr = ymax - ymin
    plt.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

    if y3 != None:
        ax3 = doaxes(ee, 1, ny, 0, 2, sharex = ax1)
        for i in np.arange(M):
            plt.plot(tt, y3[i,:] * fac2, m1[i], c = col[i])
        plt.ylabel(r"$V_{\rm{bis}}$ (m/s)")
        ymin = y3.min() * fac2
        ymax = y3.max() * fac2
        yr = ymax - ymin
        plt.ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

    plt.xlabel(xttl)
    plt.xlim(xrange[0], xrange[1])

    if savefig != None:
        plt.savefig('%s_ts.png' % savefig)
    return

def plotPer(time, y1, y2, y3 = None, figno = 2, savefig = None, \
                period = None, fmp = 8.0):
    '''Plot light curve and RV amplitude spectra'''
    M, N = np.shape(y1)
    fac1 = 100.0
    fac2 = 1.0
    ls = ['-','--','-.',':']
    col = ['k','c','m','grey']
    pmax = 2* (time.max() - time.min())
    if period == None:
        dt = np.median(time[1:]-time[:N-1])
        pmin = max([dt * 2., 0.1])
    else:
        pmin = period / fmp
    nper = 1000
    if period == None:
        fac = 1.0
    else:
        fac = period
    if y3 == None:
        ny = 2
    else:
        ny = 3
    ee = dofig(figno, 1, ny, aspect = 1)

    ax1 = doaxes(ee, 1, ny, 0, 0)
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.ylabel(r"$A_\Psi$ (\%)")
    ymax = 0
    ls = ['-','--','-','--']
    col = ['k','k','m','m']
    for i in np.arange(M):
        a, b = sinefit(time, y1[i,:], fmin = 1./pmin, fmax = 1./pmax, \
                       nfreq = nper, doplot = False)
        rchi2, freq, amp, phase, dc = a
        plt.plot(fac * freq, amp * fac1, ls[i], c = col[i])
        ymax = max(ymax, mymax(amp) * fac1)
    plt.ylim(0, 1.1 * ymax)

    ax2 = doaxes(ee, 1, ny, 0, 1, sharex = ax1)
    plt.ylabel(r"$A_V$ (m/s)")
    ymax = 0
    ls = ['-','--','-','-.']
    col = ['k','k','m','k']
    for i in np.arange(M):
        a, b = sinefit(time, y2[i,:], fmin = 1./pmin, fmax = 1./pmax, \
                       nfreq = nper, doplot = False)
        rchi2, freq, amp, phase, dc = a
        plt.plot(fac * freq, amp * fac2, ls[i], c = col[i])
        ymax = max(ymax, mymax(amp) * fac2)
    plt.ylim(0, 1.1 * ymax)

    if y3 != None:
        ax3 = doaxes(ee, 1, ny, 0, 2, sharex = ax1)
        plt.ylabel(r"$A_{\rm{bis}}$ (m/s)")
        ymax = 0
        for i in np.arange(M):
            a, b = sinefit(time, y3[i,:], fmin = 1./pmin, fmax = 1./pmax, \
                           nfreq = nper, doplot = False)
            rchi2, freq, amp, phase, dc = a
            plt.plot(fac * freq, amp * fac2, ls[i], c = col[i])
            ymax = max(ymax, mymax(amp) * fac2)
        plt.ylim(0, 1.1 * ymax)

    if period == None:
        plt.xlabel(r"Frequency (cycles/day)")
    else:
        plt.xlabel(r"Frequency (cycles/$P_{\rm rot}^{-1}$)")

    if savefig != None:
        plt.savefig('%s_per.png' % savefig)
    return
    
def plotTSPer(time, y1, y2, y3 = None, figno = [1,2], savefig = None, \
                discrete = False, period = None, xper = False, \
                  fmp = 5.):
    '''Plot both time series and amplitude spectra for light and RV'''
    plotTS(time, y1, y2, y3 = y3, figno = figno[0], discrete = discrete, \
                  savefig = savefig, period = period, xper = xper)
    plotPer(time, y1, y2, y3 = y3, figno = figno[1], savefig = savefig, \
                period = period, fmp = fmp)
    return

def gd_sample(tspan = 180, npernight = 3, drun = 10, nrun = 3, nrand = 10, \
              dnight = 8./24.):
    '''
    Compute a time array that is realistic for ground-based observations,
    e.g. with an RV spectrograph
    '''
    # One point per night
    days = np.arange(tspan)
    dt_night = dnight / float(npernight+1)
    # Multiple points per night, with small deviations from regularity
    obs = np.zeros((tspan, npernight)) 
    for i in np.arange(npernight):
        obs[:,i] = days[:] + dt_night * float(i) + \
            npran.randn(tspan) * dt_night/2.
    # Select points in "intensive" runs
    if drun == tspan:
        take = np.ones((tspan, npernight), 'int')
    else:
        take = np.zeros((tspan, npernight), 'int')
        for i in np.arange(nrun):
            ok = 0
            while ok == 0:
                tstart = np.fix(npran.rand(1) * float(tspan))
                tstart = tstart[0]
                tend = tstart + drun
                if tend > tspan: continue
                if take[tstart:tend,:].any(): continue
                take[tstart:tend,:] = 1
                ok = 1
# Select additional individual points
    ntot = tspan*npernight
    obs = np.reshape(obs, ntot)
    take = np.reshape(take, ntot)
    index = np.argsort(obs)
    obs = obs[index]
    take = take[index]
    for i in np.arange(nrand):
        ok = 0
        while ok == 0:
            t = np.fix(npran.rand(1) * float(ntot))
            t = t[0]
            if take[t] == 1: continue
            take[t] = 1
            ok = 1
    time = obs[(take==1)]
    time -= time[0]
    return time

def resample(time, y, tnew):
    '''Resample existing time series'''
    M = np.shape(y)[0]
    N = len(tnew)
    y_ = np.zeros((M, N))
    for i in np.arange(M):
        g = sciint.interp1d(time, y[i,:])
        print y_[i,:].shape
        print tnew.shape
        print g(tnew).shape
        y_[i,:] = g(tnew)
    return y_

# Dorren (1987) routines (mainly for checking accuracy of my
# simplified spot model)

def calc_bigab(alpha, beta):
    '''Calculate A & B from alpha & beta (Dorren 1987)'''
    if (beta - alpha) > (sp.pi / 2.): # spot out of view
        return 0.0, 0.0
    cosalpha = sp.cos(alpha)
    sinalpha = sp.sin(alpha)
    cosbeta = sp.cos(beta)
    sinbeta = sp.sin(beta)
    tanbeta = sinbeta / cosbeta
    if (beta + alpha) <= (sp.pi / 2.): # spot fully visible
        delta = 0.0
        sindelta = 0.0
        cosdelta = 1.0
        zeta = 0.0
        sinzeta = 0.0
        coszeta = 1.0
    else: # spot partly visible
        cosdelta = 1.0 / sp.tan(alpha) / sp.tan(beta)
        delta = sp.arccos(cosdelta)
        sindelta = sp.sin(delta)
        sinzeta = sindelta * sinalpha
        zeta = sp.arcsin(sinzeta)
    if beta <= (sp.pi / 2.): 
        T = sp.arctan(sinzeta * tanbeta)
    else:
        T = sp.pi - sp.arctan( -sinzeta * tanbeta)
    biga = zeta + (sp.pi - delta) * cosbeta * sinalpha**2 - \
        sinzeta * sinbeta * cosalpha
    bigb = (1/3.) * (sp.pi - delta) * \
        ( -2 * cosalpha**3 - 3 * sinbeta**2 * cosalpha * sinalpha**2) + \
        (2/3.) * (sp.pi - T) + (1/6.) * sinzeta * sp.sin(2 * beta) * \
        (2 - 3 * cosalpha**2)
    return biga, bigb

def calc_littleab(ustar, uspot, fratio):
    '''Calculate a & b from u_star & u_spot & F_spot/F_star (Dorren
    1987)'''
    littlea = (1 - ustar) - (1 - uspot) * fratio
    littleb = ustar - uspot * fratio
    return littlea, littleb

def dorren_F(ustar, uspot, fratio, alpha, beta):
    '''Calculate F (fraction of stellar disk hidden by spot) from
    u_star, u_spot, F_spot/F_star, alpha & beta, following Dorren (1987)'''
    biga, bigb = calc_bigab(alpha, beta)
    littlea, littleb = calc_littleab(ustar, uspot, fratio)
    F = (littlea * biga + littleb * bigb) / sp.pi / \
        (1 - ustar / 3.)
    if F < 0: F = 0
    return -F

