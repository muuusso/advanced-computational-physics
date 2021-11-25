# Build wave function for He4 variational methods
# the 2-body factor is a linear combination of solutions of the 2-body SE
# the 3-body factor from Schmidt-Chester 1980

from numba import njit
import numpy as np
from scipy.optimize import curve_fit

from interpolation.splines import UCGrid, filter_cubic, eval_cubic

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


# build wf with the rigt 2-body factor with healing distance L/2
def build_psi(L):
    # build grid for Numerov's method
    dr = 1e-4
    Rmin = 0.35
    Rmax = L/2
    
    n = int((Rmax - Rmin) / dr + 1)
    rgrid = np.linspace(Rmin, Rmax, n)
    
    i = 0 # E_jl index
    En = np.zeros(4)
    
    V = V_lj(rgrid)
    
    guess0 = numerovR(rgrid, V, -1, dr)[0]
    guessE = -1
    
    # found bound states for E in [-1, 3] epsilon with Numerov's method
    for E in np.linspace(-1, 3, 401)[1:]:
        u = numerovR(rgrid, V, E, dr)
        # find candidates for starting secant method
        if u[0] * guess0 < 0:
    
            # secant method
            u1, u0 = u[0], float(guess0)
            E1, E0 = float(E), float(guessE)
    
            while E1 - E0 > 1e-6:
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
    
    # extended grid to 0
    n_ext = int(3.5 / dr)
    ext_rgrid = np.arange(1, n_ext+1) * dr
    
    f_sol = np.zeros((4, n_ext))
    
    # due to instablity in Numerov's method near 0
    # the function are fitted with a McMillan function
    # between the nearest zero "stable" point and 1 
    mcMillan_fit = lambda r, a, b: a*np.exp(-(b / r)**5)
    
    # grid for spline cubic interpolation
    spgrid = UCGrid((dr, (n_ext+1)*dr, n_ext))

    f_n = {}
    for i in range(4):
        R = numerovR(rgrid, V_lj(rgrid), En[i], dr) / rgrid
        
        # find nearest zero "stable" point
        nR = np.argmin(np.abs(R)[:int(1 / dr)])
        
        popt, pcov = curve_fit(mcMillan_fit, rgrid[nR:int(1/dr)], R[nR:int(1/dr)])
        
        f_sol[i, -n + nR:] = R[nR:]
        f_sol[i, :-n + nR] = mcMillan_fit(ext_rgrid[:-n + nR],  *popt)
        
        # defining fn(r): every function is a cubic spline
        C = filter_cubic(spgrid, f_sol[i][:, None])
        
        f_n[i] = lambda r: eval_cubic(spgrid, C, np.expand_dims(r, 1)).flatten()
        f_n[i] = njit(f_n[i]) # jit function
        f_n[i](rgrid) # run function to compile it with the right C

    f0 = f_n[0]
    f1 = f_n[1]
    f2 = f_n[2]
    f3 = f_n[3]


    @njit # 2-body correlation factor
    def f(r_ij, cn):
        r_ij = np.tril(r_ij).flatten() # take lower triangle of matrix
        r_ij = r_ij[(r_ij < L/2) & (r_ij > 0)] # cutoff at L/2
        f_ij = cn[0]*f0(r_ij) + cn[1]*f1(r_ij) + cn[2]*f2(r_ij) + cn[3]*f3(r_ij)
        return np.prod(f_ij)
    
    
    @njit # 3-body correlation factor
    def h(R_ij, r_ij, Lambda, w, r0):
        # sum on three-body terms
        h_ijk = 0

        # cutoff at L/2
        r_ij = np.fmin(r_ij, L/2) 
        # xi on pairwise distance matrix
        xi_ij = np.exp(-(r_ij-r0)**2 / w**2) * ((2*r_ij-L) / L)**3
    
        # computing ij ik terms for every i 
        for i in range(R_ij.shape[0]):
            h_ijk += xi_ij[i].dot(R_ij[i]).dot(R_ij[i].T.dot(xi_ij[i])) 
    
        return np.exp(- Lambda / 2 * h_ijk)
    
    
    @njit
    def psi(conf, cn, Lambda, w, r0):
        # pairwise vector distance matrix
        R_ij = conf - np.expand_dims(conf, axis=1)
        R_ij = R_ij - L * np.rint(R_ij / L)
    
        # pairwise distance matrix 
        r_ij = np.sqrt(np.sum(R_ij**2, axis=2))

        return f(r_ij, cn) * h(R_ij, r_ij, Lambda, w, r0)

    return psi, f, h, f0, f1, f2, f3
