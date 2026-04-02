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

def Hubbard_hop_ham(t, t_prime, mu, Lx, Ly):
    ham = np.zeros((Lx * Ly, Lx * Ly), dtype=float)
    loc, label = square_lattice(Lx, Ly)
    
    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n
            
            # Calculate positions of the nearest neighbors
            loc_x_neighbor = loc[mod(m + 1, Lx), n]  # Neighbor to the left or right
            loc_y_neighbor = loc[m, mod(n + 1, Ly)]  # Neighbor to the top or bottom
            
            s1x = find_site_index(loc, loc_x_neighbor, Lx, Ly)
            s1y = find_site_index(loc, loc_y_neighbor, Lx, Ly)
            
            # Nearest neighbor hopping
            #along x axis 
            ham[s1][s1x] -= t
            ham[s1x][s1] -= t
            
            #along y axis 
            ham[s1][s1y] -= t
            ham[s1y][s1] -= t
            
            # Chemical potential
            ham[s1][s1] -= mu 
            
            # Calculate positions of the next-nearest neighbors
            xy_p = loc[mod(m + 1, Lx), mod(n + 1, Ly)]  # Neighbor to the diagonal x+y
            xy_m = loc[mod(m - 1, Lx), mod(n - 1, Ly)]  # Neighbor to the diagonal -x-y
            
            s2x = find_site_index(loc, xy_p, Lx, Ly)
            s2y = find_site_index(loc, xy_m, Lx, Ly)
            
            # Next-nearest neighbor hopping
            ham[s1][s2x] -= t_prime
            ham[s2x][s1] -= t_prime
            
            ham[s1][s2y] -= t_prime
            ham[s2y][s1] -= t_prime         
    return ham


def Hubbard_ham_morder(t, Lx, Ly, m_ord):
    # Initialize the Hamiltonian matrix
    #ham = np.zeros((2 * Lx * Ly, 2 * Lx * Ly), dtype = float)  # 2 for spin
    ham = np.zeros((2 * Lx * Ly, 2 * Lx * Ly))  # 2 for spin
    #loc, label = square_lattice(Lx, Ly)
    loc, _ = square_lattice(Lx, Ly)
    
    # Loop over lattice sites and construct the Hamiltonian
    for m in range(Lx):
        for n in range(Ly):
            s1_up = m * Ly + n  # Index for spin-up
            s1_down = s1_up + Lx * Ly  # Index for spin-down (second half of the Hamiltonian)
            
            # Nearest neighbor hopping for spin-up
            loc_x_neighbor = loc[mod(m + 1, Lx), n]  # Neighbor to the left or right
            loc_y_neighbor = loc[m, mod(n + 1, Ly)]  # Neighbor to the top or bottom
            
            s1x_up = find_site_index(loc, loc_x_neighbor, Lx, Ly)
            s1y_up = find_site_index(loc, loc_y_neighbor, Lx, Ly)
            
            # Nearest neighbor hopping (spin-up)
            ham[s1_up, s1x_up] -= t
            ham[s1x_up, s1_up] -= t
            ham[s1_up, s1y_up] -= t
            ham[s1y_up, s1_up] -= t

            # AFM ordering (spin-up)
            #ham[s1_up, s1_up] =  m_ord
            ham[s1_up, s1_up] += (-1)**(m+n) * m_ord
             

            # Nearest neighbor hopping for spin-down
            loc_x_neighbor_down = loc[mod(m + 1, Lx), n]  # Neighbor in x-direction
            loc_y_neighbor_down = loc[m, mod(n + 1, Ly)]  # Neighbor in y-direction
            
            s1x_down = find_site_index(loc, loc_x_neighbor_down, Lx, Ly) + Lx * Ly
            s1y_down = find_site_index(loc, loc_y_neighbor_down, Lx, Ly) + Lx * Ly
            
            # Nearest neighbor hopping (spin-down)
            ham[s1_down, s1x_down] -= t
            ham[s1x_down, s1_down] -= t
            ham[s1_down, s1y_down] -= t
            ham[s1y_down, s1_down] -= t

            # Chemical potential and AFM ordering (spin-down)
            #ham[s1_down, s1_down] = - m_ord
            ham[s1_down, s1_down] -=  (-1)**(m+n) * m_ord
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

def generate_hubbard_data(t, Lx, Ly, m_ord, mu, NFL, n_part=-1):
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
        HPS[:, :, nf] = Hubbard_projector_mord_square(t, Lx, Ly, m_ord, spin, mu)
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

def torch_generate_hubbard_data(t, Lx, Ly, m_ord, mu, NFL, n_part=-1):
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
        HPS[:, :, nf] = torch_Hubbard_projector_mord_square(t, Lx, Ly, m_ord, spin, mu)
        #eigvals, eigvecs = la.eigh(HPS[:, :, nf])  # Solve eigenproblem
        eigvals, eigvecs = tla.eigh(HPS[:, :, nf])  # Solve eigenproblem
        P0[:, :, nf] = eigvecs  # Store all eigenvectors
        P[:, :, nf] = eigvecs[:, :n_part]  # Store first n_part eigenvectors (projection)
    return HPS, P0, P

def generate_hubbard_hopping_nf(t, t_prime, mu, Lx, Ly, NFL):
    """
    Generate flavor depency in Hopping term of Hubbard Hamiltonians.

    Parameters:
        t (float): Hopping parameter.
        Lx (int): Number of lattice sites along x-axis.
        Ly (int): Number of lattice sites along y-axis.
        mu (float): Chemical potential.
        NFL (int): Number of Fermion flavors (spin components).

    Returns:
        Ht (ndarray): 3D array of Hubbard Hamiltonians of shape (ndim, ndim, NFL).
    """
    ndim = Lx * Ly  # Total number of sites
    Ht_nf = np.zeros((ndim, ndim, NFL))  # Hamiltonians for each spin
    for nf in range(NFL):
        spin = 1 - 2 * nf  # Alternating spin values (1, -1)
        Ht_nf[:, :, nf] =  Hubbard_hop_ham(t, t_prime, mu, Lx, Ly)
    return Ht_nf
