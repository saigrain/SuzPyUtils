import numpy as np

'''
orbit: compute sky positions and radial velocities for 2-body
Keplerian orbits
'''

# Utility functions

def rv_K(P, f): 
    """
    Calculate RV semi-amplitude of body 1 for orbital period P and mass
    function f, = m1 (sin(incl))**2 / (m1+m2)**3, where m1 and m2 are
    the masses of bodies 1 and 2, respectively, and incl is the
    inclination of the orbital plane (incl = pi/2 for a system seen
    edge-on).
    """
    return (2 * np.pi * 6.67e-11 / (P*86400)**2)**(1/3.) * f

def getT0(P, Ttr, omega = 0, Ecc = 0):
    """
    Compute time of periastron passage T0 from the orbital period P,
    time of transit centre Ttr, argument of periastron omega and
    eccentricity Ecc.
    """
    nu = np.pi/2 - omega
    cosnu = np.cos(nu)
    cosE = (Ecc + cosnu) / (1 + Ecc * cosnu)
    E = np.arccos(cosE)
    if (nu > -np.pi) and (nu < 0.0): E = -E  #mcquillan:1/4/10
    M = E - Ecc * np.sin(E)
    T0 = Ttr - M * P / (2 * np.pi)
    return T0

def phase(t, P, T0 = 0.0):
    """
    Phase-fold times t given period P and epoch T0; result in [0:1].
    """
    Phase = ((t-T0) % P) / P
# ensure > 0
    Phase[Phase <= 0.] += 1
    return Phase

# Lower level functions

f = np.MachAr()
machep = f.eps

def eccan(Ecc, M, Tol = 1.0e-8, Nmax = 50):
    """
    Calculate eccentric anomaly from eccentricity Ecc and mean
    anomaly M using Newton-Raphson process.
    """
    if M < Tol: return M
    x = Ecc * np.sin(M) / (1 - Ecc * np.cos(M))
    Eo = M + x * (1-x*x/2.)
    Diff = 1
    Flag = 0
    i = 0
    while (Diff > Tol):
        En = Eo + (M  + Ecc * np.sin(Eo) - Eo) / (1 - Ecc * np.cos(Eo))
        Diff = abs((En - Eo) / Eo)
        Eo = En
        i += 1
        if i >= Nmax:
            if Flag ==1:
                print Ecc, M
                print 'Eccan did not converge'
                return M
            Flag = 1
            i = 0
            Eo = M 
            Diff = 1
    return En

def truean(t, P, T0 = 0, Ecc = 0):
    """
    Calculate true anomaly for times t, orbital period P, time of
    periastron passage T0, and eccentricity Ecc.
    """
    Phase = phase(t, P, T0) # phases
    M = 2 * np.pi * Phase # mean anomaly
    if Ecc <= machep:
        return M
    eccanV = np.vectorize(eccan) 
    E = eccanV(Ecc, M) % (2 * np.pi)  # eccentric anomaly
    cosE = np.cos(E)
    cosNu = (cosE - Ecc) / (1 - Ecc * cosE)
    Nu = np.arccos(cosNu) # true anomaly
    Nu = np.select([E <= np.pi, np.ones(len(Nu))], \
                          [Nu, 2 * np.pi - Nu]) # E>pi cases
    return Nu

def truedist(Nu, a, Ecc = 0):
    """
    Compute distance from center of mass, given true anomaly Nu,
    semi-major axis a & eccentricity Ecc.
    """
    return a * (1 - Ecc**2) / (1 + Ecc * np.cos(Nu))

def orbitcoord(t, P, T0 = 0, Ecc = 0, a = 1):
    """
    Compute coordinates X and Y relative to center of mass, in orbital
    plane, for times t, given orbital period P, time of periastron
    passage T0, eccentricity Ecc and semi-major axis a. X is towards
    observer. The true anomaly Nu is also returned for convenience.
    """
    Nu = truean(t, P, T0, Ecc)
    r = truedist(Nu, a, Ecc)
    X = r * np.cos(Nu)
    Y = r * np.sin(Nu)
    return X, Y, Nu

# Top-level functions

def skycoord(t, P, T0 = 0, Ecc = 0, a = 1, \
             incl = np.pi/2, Omega = 0, omega = 0):
    """
    Compute coordinates in plane of sky (x=East, y=North), relative to
    centre of mass for times t, given orbital period P, time of
    periastron passage T0, eccentricity Ecc, semi-major axis a,
    inclination of orbital plane incl (incl = pi/2 for an edge-on
    orbit), longitude of ascending node Omega, and argument of
    periastron omega. The z coordinate (away from observer) is also
    returned for convenience.
    """
    X, Y, Nu = orbitcoord(t, P, T0, Ecc, a)
    cosi = np.cos(incl)
    sini = np.sin(incl)
    cosO = np.cos(Omega)
    sinO = np.sin(Omega)
    coso = np.cos(omega)
    sino = np.sin(omega)
    cosxX = - cosi * sinO * sino + cosO * coso
    cosxY = - cosi * sinO * coso - cosO * sino
    cosyX = cosi * cosO * sino + sinO * coso
    cosyY = cosi * cosO * coso - sinO * sino
    x = X * cosxX + Y * cosxY
    y = X * cosyX + Y * cosyY
    z = np.sqrt(X**2+Y**2) * sini * np.sin(omega+Nu)
    return x, y, z

def radvel(t, P, K, T0 = 0, V0 = 0, Ecc = 0, omega = 0):
    """
    Compute radial velocity for times t, given orbital period P, RV
    semi-amplitude K, time of periastron passage T0, systemic velocity
    (velocity of the center of mass) V0, eccentricity Ecc, and
    argument of periastron omega.
    """
    Nu = truean(t, P, T0, Ecc)
    Vr = V0 + K * (np.cos(omega + Nu) + Ecc * np.cos(omega))
    if (K < 0): Vr[:] = -999
    return Vr

