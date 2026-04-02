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


#def Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x="open", bc_y="open", ladder=False):
def Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x, bc_y, ladder=False):
    """
    Same structure as your original Hubbard_hop_ham,
    but now supports open/periodic boundaries along x and y,
    and optional ladder geometry (Ly→2).
    """

    # Ladder = force two legs
    #if ladder:
    #    Ly = 2
        
    #print('Ly =',Ly)
    ham = np.zeros((Lx * Ly, Lx * Ly), dtype=float)
    loc, label = square_lattice(Lx, Ly)

    for m in range(Lx):
        for n in range(Ly):
            s1 = m * Ly + n

            # ---------- Nearest neighbors ----------

            # +x neighbor (right)
            if m + 1 < Lx:
                xm = m + 1
            elif bc_x == "periodic":
                xm = mod(m + 1, Lx)
            else:
                xm = None

            if xm is not None:
                s1x = xm * Ly + n
                ham[s1][s1x] -= t
                ham[s1x][s1] -= t

            # +y neighbor (up / rung for ladder)
            if n + 1 < Ly:
                yn = n + 1
            elif bc_y == "periodic":
                yn = mod(n + 1, Ly)
            else:
                yn = None

            if yn is not None:
                s1y = m * Ly + yn
                ham[s1][s1y] -= t
                ham[s1y][s1] -= t

            # Chemical potential
            ham[s1][s1] -= mu

            # ---------- Next-nearest neighbors (diagonals) ----------

            # (m+1, n+1)
            xm = m + 1
            yn = n + 1
            if xm >= Lx:
                xm = mod(xm, Lx) if bc_x == "periodic" else None
            if yn >= Ly:
                yn = mod(yn, Ly) if bc_y == "periodic" else None

            if xm is not None and yn is not None:
                s2 = xm * Ly + yn
                ham[s1][s2] -= t_prime
                ham[s2][s1] -= t_prime

            # (m-1, n-1)
            xm = m - 1
            yn = n - 1
            if xm < 0:
                xm = mod(xm, Lx) if bc_x == "periodic" else None
            if yn < 0:
                yn = mod(yn, Ly) if bc_y == "periodic" else None

            if xm is not None and yn is not None:
                s2 = xm * Ly + yn
                ham[s1][s2] -= t_prime
                ham[s2][s1] -= t_prime
            
    return ham

def Hubbard_ham_morder_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x, bc_y):
#def Hubbard_ham_morder_ladder(t, Lx, Ly, mu, m_ord, spin, bc_x="open", bc_y="open"):
    ham = np.zeros((2*Lx * Ly, 2*Lx * Ly))
    loc, _ = square_lattice(Lx, Ly)
    for m in range(Lx):
        for n in range(Ly):
            s1_up = m * Ly + n
            s1_down =  s1_up + m * Ly + n
            # ---- x-direction neighbor ----
            if m + 1 < Lx:
                mx = m + 1
            elif bc_x == "periodic":
                mx = mod(m + 1, Lx)
            else:
                mx = None

            if mx is not None:
                loc_x_neighbor = loc[mx, n]
                s1x_up = find_site_index(loc, loc_x_neighbor, Lx, Ly)

                ham[s1_up][s1x_up] -= t 
                ham[s1x_up][s1_up] -= t 

            # ---- y-direction (rung) neighbor ----
            if n + 1 < Ly:
                ny = n + 1
            elif bc_y == "periodic":
                ny = mod(n + 1, Ly)
            else:
                ny = None

            if ny is not None:
                loc_y_neighbor = loc[m, ny]
                s1y_up = find_site_index(loc, loc_y_neighbor, Lx, Ly)

                ham[s1_up][s1y_up] -= t 
                ham[s1y_up][s1_up] -= t 

            # on-site terms
            ham[s1_up, s1_up] -= mu
            ham[s1_up, s1_up] += spin * m_ord * (-1)**(m+n)
            
            s1_down =  s1_up + m * Ly + n
            # ---- x-direction neighbor ----
            if m + 1 < Lx:
                mx = m + 1
            elif bc_x == "periodic":
                mx = mod(m + 1, Lx)
            else:
                mx = None

            if mx is not None:
                loc_x_neighbor = loc[mx, n]
                s1x_down = find_site_index(loc, loc_x_neighbor, Lx, Ly)+ Lx * Ly

                ham[s1_down][s1x_down] -= t 
                ham[s1x_down][s1_down] -= t 

            # ---- y-direction (rung) neighbor ----
            if n + 1 < Ly:
                ny = n + 1
            elif bc_y == "periodic":
                ny = mod(n + 1, Ly)
            else:
                ny = None

            if ny is not None:
                loc_y_neighbor = loc[m, ny]
                s1y_down = find_site_index(loc, loc_y_neighbor, Lx, Ly)+ Lx * Ly

                ham[s1_down][s1y_down] -= t 
                ham[s1y_down][s1_down] -= t 

            # on-site terms
            ham[s1_down, s1_down] -= mu
            ham[s1_down, s1_down] -= spin * m_ord * (-1)**(m+n)

    return ham

def Hubbard_hop_ham1(t, t_prime, mu, Lx, Ly):
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

def Hubbard_ham_morder1(t, Lx, Ly, m_ord):
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

def hubbard_hopping_nf(t, t_prime, mu, Lx, Ly, NFL, bc_x, bc_y, ladder):
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
    #print('bc_x, bc_y, ladder, ndim =',bc_x, bc_y, ladder, ndim)
    Ht_nf = np.zeros((ndim, ndim, NFL))  # Hamiltonians for each spin
    for nf in range(NFL):
        spin = 1 - 2 * nf  # Alternating spin values (1, -1)
        if bc_x=="open" and bc_y=="open":
            Ht_nf[:, :, nf] =  Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x="open", bc_y="open", ladder=True)
        elif bc_x=="periodic" and bc_y=="periodic":
            Ht_nf[:, :, nf] =  Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x="periodic", bc_y="periodic")   
        elif bc_x=="open" and bc_y=="periodic":
            Ht_nf[:, :, nf] =  Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x="open", bc_y="periodic")
        elif bc_x=="periodic" and bc_y=="open":
            Ht_nf[:, :, nf] =  Hubbard_hop_ham(t, t_prime, mu, Lx, Ly, bc_x="periodic", bc_y="open")
            
    return Ht_nf
