# Build wave function for He4 variational methods
# the 2-body factor is a linear combination of solutions of the 2-body SE
# the 3-body factor from Schmidt-Chester 1980

from numba import njit
import numpy as np
from scipy.optimize import curve_fit

from scipy.constants import hbar, physical_constants
from scipy.constants import k as kB # Boltzmann constant

epsilon = 10.22 # K (Kelvin)
sigma = 2.556 # A (Angstroms)

m = 4.002603254 # amu (atomic mass unit)
amu = physical_constants['atomic mass constant'][0] # Kg

# eta := hbar^2 / m in epsilon-sigma units
eta = hbar**2 / (m*amu) / (epsilon*kB) / (sigma**2*1e-20)


@njit # Lennard-Jones potential
def V_lj(r):
    return 4 * ((1 / r)**12 - (1 / r)**6)


@njit # Numerov's method starting from the end of the grid
def numerovR(u, V, E, dr):
    k2 = (E - V) / eta

    # copy array to create new one to avoid editing the original one
    u = np.copy(u)

    for n in range(len(u)-2, 0, -1):
        # Numerov formula
        u[n-1] = u[n]*(2-5*dr**2*k2[n]/6) - u[n+1]*(1+dr**2*k2[n+1]/12)
        u[n-1] = u[n-1] / (1 + dr**2*k2[n-1]/12)
    return u


def build_fn(L, n):

    # build grid for Numerov's method
    dr = 1e-4
    Rmin = 0.35
    Rmax = L/2
    
    n_grid = int((Rmax - Rmin) / dr)
    rgrid = np.arange(n_grid) * dr + Rmin

    i = 0 # En index
    En = np.zeros(n)
    
    V = V_lj(rgrid)

    guess0 = numerovR(rgrid, V, -1, dr)[0]
    guessE = -1
        
    # found n bound states with Numerov's method
    E = -1 + 0.01
    while En[n-1] == 0:
        u = numerovR(rgrid, V, E, dr)
        # find candidates for starting secant method
        if u[0] * guess0 < 0:
        
            # secant method
            u1, u0 = u[0], float(guess0)
            E1, E0 = float(E), float(guessE)
        
            while E1 - E0 > 1e-9:
                # updating formula
                E2 = E1 - u1 * (E1 - E0) / (u1 - u0)
        
                u2 = numerovR(rgrid, V, E2, dr)
                if u2[0] * u1 < 0:
                    E0 = float(E2)
                    u0 = u2[0]
                else:
                    E1 = float(E2)
                    u1 = u2[0]
        
            # add energy values and update indexes
            E2 = (E1 - E0) / 2 + E0
            En[i] = E2
        
            i += 1
        
        guess0 = u[0]
        guessE = E
        E += 0.01
    
    fn = np.zeros((n, n_grid))
    nR = 0
    for i in range(n):
        R = numerovR(rgrid, V_lj(rgrid), En[i], dr) / rgrid

        # find nearest zero "stable" point
        nR = max(nR, np.argmin(np.abs(R)[:int(1 / dr)]))
        
        fn[i] = np.copy(R)
        
    rgrid = rgrid[nR:]
    fn = fn[:, nR:]

    return fn, rgrid
