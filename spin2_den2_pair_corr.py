import numpy as np
from hopping_ham_mod import square_lattice


def mod(a, b):
    return (a + b) % b

def spin_spin_corr(GR, GRC, Lx, Ly):
    """
    Compute r-resolved spin-spin correlations on a square lattice.
    Returns 2D arrays (dx, dy) of averaged correlations.
    """
    Nsite = Lx * Ly
    ZZ_r  = np.zeros((Lx, Ly), dtype=np.float64)
    ZXY_r = np.zeros((Lx, Ly), dtype=np.float64)
    count = np.zeros((Lx, Ly), dtype=int)

    # Generate lattice coordinates
    loc, label = square_lattice(Lx, Ly)

    for i in range(Nsite):
        xi, yi = label[i]  # integer coordinates of site i
        for j in range(Nsite):
            xj, yj = label[j]  # integer coordinates of site j

            # compute periodic displacements
            dx = mod(xj - xi, Lx)
            dy = mod(yj - yi, Ly)

            # minimum image convention (shortest distance)
            if dx > Lx // 2:
                dx -= Lx
            if dy > Ly // 2:
                dy -= Ly

            # store correlation in 0..Lx-1, 0..Ly-1 indexing
            idx = dx % Lx
            idy = dy % Ly

            ZZ_r[idx, idy] += (
                GRC[i, j, 0] * GR[i, j, 0]
              + GRC[i, j, 1] * GR[i, j, 1]
              + (GRC[i, i, 1] - GRC[i, i, 0]) * (GRC[j, j, 1] - GRC[j, j, 0])
            )

            ZXY_r[idx, idy] += (
                GRC[i, j, 0] * GR[i, j, 1]
              + GRC[i, j, 1] * GR[i, j, 0]
            )

            count[idx, idy] += 1

    # normalize to get average per displacement
    ZZ_r  /= count
    ZXY_r /= count
    ZZ2XY_r = ZZ_r + 2.0 * ZXY_r

    return ZZ_r, ZXY_r, ZZ2XY_r

def spin_spin_corr_full(GR, GRC, Lx, Ly):
    """
    Compute full site-site spin-spin correlations.
    Output arrays are (Nsite, Nsite) = (Lx*Ly, Lx*Ly)
    """
    Nsite = Lx * Ly

    ZZ = np.zeros((Nsite, Nsite), dtype=np.float64)
    ZXY = np.zeros((Nsite, Nsite), dtype=np.float64)

    for i in range(Nsite):
        for j in range(Nsite):
            ZZ[i, j] = (
                GRC[i, j, 0] * GR[i, j, 0]
              + GRC[i, j, 1] * GR[i, j, 1]
              + (GRC[i, i, 1] - GRC[i, i, 0]) * (GRC[j, j, 1] - GRC[j, j, 0])
            )
            ZXY[i, j] = (
                GRC[i, j, 0] * GR[i, j, 1]
              + GRC[i, j, 1] * GR[i, j, 0]
            )

    ZZ2XY = ZZ + 2.0 * ZXY
    return ZZ, ZXY, ZZ2XY

def spin_spin_corr_placeholder(GR, GRC, Lx, Ly):
    Nsite = Lx * Ly

    ZZ_r  = np.zeros((Lx, Ly), dtype=np.float64)
    ZXY_r = np.zeros((Lx, Ly), dtype=np.float64)
    count = np.zeros((Lx, Ly), dtype=int)

    for i in range(Nsite):
        xi, yi = divmod(i, Ly)
        for j in range(Nsite):
            xj, yj = divmod(j, Ly)
            dx = (xj - xi) % Lx
            dy = (yj - yi) % Ly

            ZZ_r[dx, dy] += (
                GRC[i, j, 0] * GR[i, j, 0]
              + GRC[i, j, 1] * GR[i, j, 1]
              + (GRC[i, i, 1] - GRC[i, i, 0])
                * (GRC[j, j, 1] - GRC[j, j, 0])
            )

            ZXY_r[dx, dy] += (
                GRC[i, j, 0] * GR[i, j, 1]
              + GRC[i, j, 1] * GR[i, j, 0]
            )

            count[dx, dy] += 1

    # normalize
    ZZ_r  /= count
    ZXY_r /= count
    ZZ2XY_r = ZZ_r + 2.0 * ZXY_r

    return ZZ_r, ZXY_r, ZZ2XY_r

#Density-Density correlations
def density_density_corr(GR, GRC, Lx, Ly):
    """
    Compute r-resolved density-density correlations on a square lattice.
    Returns 2D arrays (dx, dy) of averaged correlations.
    
    GR, GRC: Green's function arrays, shape (Nsite, Nsite, 2) for spin up/down or orbital indices
    Lx, Ly: lattice dimensions
    """
    Nsite = Lx * Ly
    DD_r = np.zeros((Lx, Ly), dtype=np.float64)
    count = np.zeros((Lx, Ly), dtype=int)

    # Generate lattice coordinates
    loc, label = square_lattice(Lx, Ly)

    for i in range(Nsite):
        xi, yi = label[i]
        for j in range(Nsite):
            xj, yj = label[j]

            # periodic displacements
            dx = (xj - xi) % Lx
            dy = (yj - yi) % Ly
            if dx > Lx // 2:
                dx -= Lx
            if dy > Ly // 2:
                dy -= Ly
            idx = dx % Lx
            idy = dy % Ly

            # density-density correlation: <n_i n_j> - <n_i><n_j>
            n_i = GRC[i, i, 0] + GRC[i, i, 1]
            n_j = GRC[j, j, 0] + GRC[j, j, 1]
            DD_r[idx, idy] += (n_i * n_j) - ((GR[i, i, 0] + GR[i, i, 1]) * (GR[j, j, 0] + GR[j, j, 1]))
            count[idx, idy] += 1

    DD_r /= count
    return DD_r

#Pair-pair correlations functions
def pair_corr(GR, GRC, Lx, Ly, pairing='s'):
    """
    Compute r-resolved superconducting pair correlations.
    pairing: 's', 'px', 'py', 'd'
    """
    Nsite = Lx * Ly
    P_r = np.zeros((Lx, Ly), dtype=np.complex128)
    count = np.zeros((Lx, Ly), dtype=int)

    loc, label = square_lattice(Lx, Ly)

    # Define pairing form factors for nearest neighbors
    def form_factor(dx, dy, pairing):
        if pairing == 's':
            return 1
        elif pairing == 'px':
            return dx
        elif pairing == 'py':
            return dy
        elif pairing == 'd':
            return dx**2 - dy**2
        else:
            raise ValueError("Unknown pairing type")

    for i in range(Nsite):
        xi, yi = label[i]
        for j in range(Nsite):
            xj, yj = label[j]

            dx = (xj - xi) % Lx
            dy = (yj - yi) % Ly
            if dx > Lx // 2:
                dx -= Lx
            if dy > Ly // 2:
                dy -= Ly
            idx = dx % Lx
            idy = dy % Ly

            # Pair-pair correlation: <c_i↑ c_i↓ c_j↓† c_j↑†>
            # Approximated as product of anomalous Greens functions
            delta_i = GRC[i, i, 0] * GRC[i, i, 1]  # local pair
            delta_j = GRC[j, j, 0] * GRC[j, j, 1]  # local pair
            P_r[idx, idy] += form_factor(dx, dy, pairing) * delta_i * np.conj(delta_j)
            count[idx, idy] += 1

    P_r /= count
    return P_r
