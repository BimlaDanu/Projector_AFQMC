import numpy as np
import scipy.linalg as la
import torch
import torch.linalg as tla

def mod(a, b):
    return (a + b) % b

def square_lattice(Lx, Ly):
    # Real space translation vectors
    a = 1
    a1 = np.array([a, 0])
    a2 = np.array([0, a])
    loc = np.zeros((Lx, Ly, 2), dtype=float)  # loc array now stores 2D vectors
    label = np.zeros((Lx * Ly, 2), dtype=int)
    
    site = 0
    for m in range(Lx):
        for n in range(Ly):
            # Compute real space coordinates using the lattice vectors
            loc[m, n] = m * a1 + n * a2
            label[site, 0] = m
            label[site, 1] = n
            site += 1
    return loc, label

def find_site_index(loc, target, Lx, Ly):
    """Find the index in the `loc` array corresponding to the target position."""
    for m in range(Lx):
        for n in range(Ly):
            if np.allclose(loc[m, n], target):
                return m * Ly + n
    raise ValueError("Target position not found in loc array.")

def projector(t, Lx, Ly):
    #ham = np.zeros((Lx * Ly, Lx * Ly), dtype=float)
    ham = np.zeros((Lx * Ly, Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    epsl = 0.01
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n
            # Calculate positions of the nearest neighbors
            loc_x_neighbor = loc[mod(m + 1, Lx), n]  # Neighbor to the left or right
            loc_y_neighbor = loc[m, mod(n + 1, Ly)]  # Neighbor to the top or bottom
            
            
            s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)
            s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)
            
            #interaction in x directions
            ham[s1][s1x] -= t*(1. + (-1)**(m+n)*epsl)
            ham[s1x][s1] -= t*(1. + (-1)**(m+n)*epsl)
            
            #interaction in y directions
            ham[s1][s1y] -= t*(1.-epsl)
            ham[s1y][s1] -= t*(1.-epsl)     
    return ham

def Hubbard_projector_mord_square(t, Lx, Ly, mu, m_ord, spin):
    #ham = np.zeros((Lx * Ly, Lx * Ly), dtype=float)
    ham = np.zeros((Lx * Ly, Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    epsl =0.01
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n
            # Calculate positions of the nearest neighbors
            loc_x_neighbor = loc[mod(m + 1, Lx), n]  # Neighbor to the left or right
            loc_y_neighbor = loc[m, mod(n + 1, Ly)]  # Neighbor to the top or bottom
            
            
            s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)
            s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)
            
            #interaction in x directions
            ham[s1][s1x] -= t*(1.+ (-1)**(m+n)*epsl)
            ham[s1x][s1] -= t*(1.+ (-1)**(m+n)*epsl)
            
            #interaction in y directions
            ham[s1][s1y] -= t*(1.-epsl)
            ham[s1y][s1] -= t*(1.-epsl)    
            
            ham[s1, s1] -= mu 
            ham[s1, s1] += spin * m_ord * (-1)**(m+n)    
    return ham

def projector_ladder(t, Lx, Ly, bc_x, bc_y):
#def projector_ladder(t, Lx, Ly, bc_x="open", bc_y="open"):    
    ham = np.zeros((Lx * Ly, Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    epsl = 0.01
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n

            # ---- x-direction neighbor ----
            if m + 1 < Lx:
                mx = m + 1
            elif bc_x == "periodic":
                mx = mod(m + 1, Lx)
            else:
                mx = None

            if mx is not None:
                loc_x_neighbor = loc[mx, n]
                s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)

                ham[s1][s1x] -= t * (1. + (-1)**(m+n) * epsl)
                ham[s1x][s1] -= t * (1. + (-1)**(m+n) * epsl)

            # ---- y-direction (rung) neighbor ----
            if n + 1 < Ly:
                ny = n + 1
            elif bc_y == "periodic":
                ny = mod(n + 1, Ly)
            else:
                ny = None

            if ny is not None:
                loc_y_neighbor = loc[m, ny]
                s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)

                ham[s1][s1y] -= t * (1. - epsl)
                ham[s1y][s1] -= t * (1. - epsl)

    return ham

def Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x, bc_y):
#def Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="open"):
    ham = np.zeros((Lx * Ly, Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    epsl = 0.01
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n

            # ---- x-direction neighbor ----
            if m + 1 < Lx:
                mx = m + 1
            elif bc_x == "periodic":
                mx = mod(m + 1, Lx)
            else:
                mx = None

            if mx is not None:
                loc_x_neighbor = loc[mx, n]
                s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)

                ham[s1][s1x] -= t * (1. + (-1)**(m+n) * epsl)
                ham[s1x][s1] -= t * (1. + (-1)**(m+n) * epsl)

            # ---- y-direction (rung) neighbor ----
            if n + 1 < Ly:
                ny = n + 1
            elif bc_y == "periodic":
                ny = mod(n + 1, Ly)
            else:
                ny = None

            if ny is not None:
                loc_y_neighbor = loc[m, ny]
                s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)

                ham[s1][s1y] -= t * (1. - epsl)
                ham[s1y][s1] -= t * (1. - epsl)

            # on-site terms
            ham[s1, s1] -= mu
            ham[s1, s1] += spin * m_ord * (-1)**(m+n)

    return ham

def trial_wavefunction_Hubbard(t, Lx, Ly, m_ord, mu, NFL, n_part, bc_x, bc_y):
    """
    Generate Hubbard Hamiltonians, eigenvectors, and projections for a square lattice.
    Parameters:
        t (float): Hopping parameter.
        Lx (int): Number of lattice sites along x-axis.
        Ly (int): Number of lattice sites along y-axis.
        epsl (float): Epsilon parameter for modulation.
        m_ord (float): Staggered magnetization order parameter.
        mu (float): Chemical potential.
        NFL (int): Number of Fermion flavors (spin components).
        n_part (int): Number of particles (defaults to half-filling).

    Returns:
        HPS (ndarray): 3D array of Hubbard Hamiltonians of shape (ndim, ndim, NFL).
        P0 (ndarray): 3D array of eigenvectors of shape (ndim, ndim, NFL).
        P (ndarray): 3D array of projections of shape (ndim, n_part, NFL).
    """
    ndim = Lx * Ly  # Total number of sites

    # Allocate arrays to store results
    HPS = np.zeros((ndim, ndim, NFL))  # Hamiltonians for each spin
    P0 = np.zeros((ndim, ndim, NFL))  # Eigenvectors for each spin
    P = None

    if n_part < 0:
        n_part = ndim // 2  # Default to half-filling

    P = np.zeros((ndim, n_part, NFL))  # Projections for each spin

    # Iterate over spin flavors
    for nf in range(NFL):
        spin = 1 - 2 * nf  # Alternating spin values (1, -1)
        #HPS[:, :, nf] = Hubbard_projector_mord_square(t, Lx, Ly, m_ord, spin, mu)
       # HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, m_ord, spin, mu, bc_x="open", bc_y="open")
        #HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, m_ord, spin, mu, bc_x="open", bc_y="open")
        #
        #
        if bc_x=="open" and bc_y=="open":
            HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="open")
        elif bc_x=="periodic" and bc_y=="periodic":
            HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="periodic", bc_y="periodic")
        elif bc_x=="periodic" and bc_y=="open":
            HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="periodic", bc_y="open")
        elif bc_x=="open" and bc_y=="periodic":
            HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="periodic")
        #
        #
        #HPS[:, :, nf] = Hubbard_projector_mord_ladder(t, Lx, Ly, m_ord, spin, mu, bc_x="periodic", bc_y="open")
        #
        #
        eigvals, eigvecs = la.eigh(HPS[:, :, nf])  # Solve eigenproblem
        P0[:, :, nf] = eigvecs  # Store all eigenvectors
        P[:, :, nf] = eigvecs[:, :n_part]  # Store first n_part eigenvectors (projection)
    return HPS, P0, P


def torch_Hubbard_projector_mord_square(t, Lx, Ly, mu, m_ord, spin):
    #ham = torch.zeros((Lx * Ly, Lx * Ly),  dtype = torch.double)
    ham = torch.zeros((Lx * Ly, Lx * Ly),  dtype=torch.float64)
    ham = torch.zeros((Lx * Ly, Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    epsl = 0.01
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n
            
            # Calculate positions of the nearest neighbors
            
            loc_x_neighbor = loc[mod(m + 1, Lx), n]  # Neighbor to the left or right
            loc_y_neighbor = loc[m, mod(n + 1, Ly)]  # Neighbor to the top or bottom
            
            
            s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)
            s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)
            
            #interaction in x directions
            ham[s1][s1x] -= t*(1.+ (-1)**(m+n)*epsl)
            ham[s1x][s1] -= t*(1.+ (-1)**(m+n)*epsl)
            
            #interaction in y directions
            ham[s1][s1y] -= t*(1.-epsl)
            ham[s1y][s1] -= t*(1.-epsl)    
            
            ham[s1, s1] -= mu 
            ham[s1, s1] += spin * m_ord * (-1)**(m+n)    
    return ham

#def torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="open"):
def torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x, bc_y):
    ham = torch.zeros((Lx * Ly, Lx * Ly), dtype=torch.float64)
    loc, _ = square_lattice(Lx, Ly)
    epsl = 0.01

    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n

            # ---------- x-direction neighbor ----------
            if m + 1 < Lx:
                mx = m + 1
            elif bc_x == "periodic":
                mx = mod(m + 1, Lx)
            else:
                mx = None

            if mx is not None:
                loc_x_neighbor = loc[mx, n]
                s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)

                ham[s1][s1x] -= t * (1. + (-1)**(m+n) * epsl)
                ham[s1x][s1] -= t * (1. + (-1)**(m+n) * epsl)

            # ---------- y-direction (rung) neighbor ----------
            if n + 1 < Ly:
                ny = n + 1
            elif bc_y == "periodic":
                ny = mod(n + 1, Ly)
            else:
                ny = None

            if ny is not None:
                loc_y_neighbor = loc[m, ny]
                s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)

                ham[s1][s1y] -= t * (1. - epsl)
                ham[s1y][s1] -= t * (1. - epsl)

            # ---------- on-site contributions ----------
            ham[s1, s1] -= mu
            ham[s1, s1] += spin * m_ord * (-1)**(m+n)

    return ham


def torch_trial_wavefunction_Hubbard(t, Lx, Ly, m_ord, mu, NFL, n_part,  bc_x, bc_y):
#def torch_trial_wavefunction_Hubbard1(t, Lx, Ly, m_ord, mu, NFL, n_part=-1,  bc_x="open", bc_y="open"):
    ndim = Lx * Ly  # Total number of sites

    # Allocate arrays to store results
    HPS = torch.zeros((ndim, ndim, NFL))  # Hamiltonians for each spin
    P0 = torch.zeros((ndim, ndim, NFL))  # Eigenvectors for each spin
    P = None

    if n_part < 0:
        n_part = ndim // 2  # Default to half-filling

    P = torch.zeros((ndim, n_part, NFL), dtype=torch.float64)  # Projections for each spin

    # Iterate over spin flavors
    for nf in range(NFL):
        spin = 1 - 2 * nf  # Alternating spin values (1, -1)
        #HPS[:, :, nf] = torch_Hubbard_projector_mord_square(t, Lx, Ly, mu, m_ord, spin)
        #HPS[:, :, nf] = torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x, bc_y)
        #
        #
        if bc_x=="open" and bc_y=="open":
            HPS[:, :, nf] = torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="open")
        elif bc_x=="periodic" and bc_y=="periodic":
            HPS[:, :, nf] = torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="periodic", bc_y="periodic")
        elif bc_x=="periodic" and bc_y=="open":
            HPS[:, :, nf] = torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="periodic", bc_y="open")
        elif bc_x=="open" and bc_y=="periodic":
            HPS[:, :, nf] = torch_Hubbard_projector_mord_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="periodic")
        #
        #
        #eigvals, eigvecs = la.eigh(HPS[:, :, nf])  # Solve eigenproblem
        eigvals, eigvecs = tla.eigh(HPS[:, :, nf])  # Solve eigenproblem
        P0[:, :, nf] = eigvecs  # Store all eigenvectors
        P[:, :, nf] = eigvecs[:, :n_part]  # Store first n_part eigenvectors (projection)
    return HPS, P0, P


