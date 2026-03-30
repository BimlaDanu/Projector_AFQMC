import numpy as np
from hopping_ham_mod import square_lattice


def mod(a, b):
    return (a + b) % b


def get_displacement(xi, yi, xj, yj, Lx, Ly, bc_x, bc_y):
    dx = xj - xi
    dy = yj - yi

    # ---- X direction ----
    if bc_x == "periodic":
        dx = dx % Lx
        if dx > Lx // 2:
            dx -= Lx
        idx = dx % Lx
    elif bc_x == "open":
        if abs(dx) >= Lx:
            return None
        idx = dx + (Lx - 1)
    else:
        raise ValueError("bc_x must be 'open' or 'periodic'")

    # ---- Y direction ----
    if bc_y == "periodic":
        dy = dy % Ly
        if dy > Ly // 2:
            dy -= Ly
        idy = dy % Ly
    elif bc_y == "open":
        if abs(dy) >= Ly:
            return None
        idy = dy + (Ly - 1)
    else:
        raise ValueError("bc_y must be 'open' or 'periodic'")

    return idx, idy


def spin_spin_corr_bc(GR, GRC, Lx, Ly,
                      bc_x="periodic",
                      bc_y="periodic"):
    """
    Computes translationally averaged spin-spin correlation
    C(dx,dy) = < S_i · S_{i+r} >
    consistent with Hubbard_hop_ham indexing.

    Assumes site index = m*Ly + n
    """

    Nsite = Lx * Ly

    # --- allocate displacement-resolved arrays ---
    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    ZZ_r  = np.zeros((size_x, size_y), dtype=np.float64)
    ZXY_r = np.zeros((size_x, size_y), dtype=np.float64)
    count = np.zeros((size_x, size_y), dtype=int)

    # --- loop over all sites i ---
    for m in range(Lx):
        for n in range(Ly):

            i = m * Ly + n

            # --- loop over all sites j ---
            for mp in range(Lx):
                for np_ in range(Ly):

                    j = mp * Ly + np_

                    # ==========================
                    # Displacement calculation
                    # ==========================

                    dx = mp - m
                    dy = np_ - n

                    if bc_x == "periodic":
                        dx = dx % Lx
                        idx = dx
                    else:
                        if abs(dx) >= Lx:
                            continue
                        idx = dx + (Lx - 1)

                    if bc_y == "periodic":
                        dy = dy % Ly
                        idy = dy
                    else:
                        if abs(dy) >= Ly:
                            continue
                        idy = dy + (Ly - 1)

                    # ==========================
                    # Spin correlations
                    # ==========================

                    ZZ_r[idx, idy] += (
                        GRC[i, j, 0] * GR[i, j, 0]
                        + GRC[i, j, 1] * GR[i, j, 1]
                        + (GRC[i, i, 1] - GRC[i, i, 0])
                          * (GRC[j, j, 1] - GRC[j, j, 0])
                    )

                    ZXY_r[idx, idy] += (
                        GRC[i, j, 0] * GR[i, j, 1]
                        + GRC[i, j, 1] * GR[i, j, 0]
                    )

                    count[idx, idy] += 1

    # --- normalize ---
    mask = count > 0
    ZZ_r[mask]  /= count[mask]
    ZXY_r[mask] /= count[mask]

    ZZ2XY_r = ZZ_r + 2.0 * ZXY_r

    return ZZ_r, ZXY_r, ZZ2XY_r

def spin_spin_corr_bc1(GR, GRC, Lx, Ly, bc_x="periodic", bc_y="periodic"):

    Nsite = Lx * Ly
    loc, label = square_lattice(Lx, Ly)

    # Output size depends on BC
    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    ZZ_r  = np.zeros((size_x, size_y), dtype=np.float64)
    ZXY_r = np.zeros((size_x, size_y), dtype=np.float64)
    count = np.zeros((size_x, size_y), dtype=int)

    for i in range(Nsite):
        xi, yi = label[i]

        for j in range(Nsite):
            xj, yj = label[j]

            disp = get_displacement(
                xi, yi, xj, yj, Lx, Ly, bc_x, bc_y
            )

            if disp is None:
                continue

            idx, idy = disp

            ZZ_r[idx, idy] += (
                GRC[i, j, 0] * GR[i, j, 0]
              + GRC[i, j, 1] * GR[i, j, 1]
              + (GRC[i, i, 1] - GRC[i, i, 0])
                * (GRC[j, j, 1] - GRC[j, j, 0])
            )

            ZXY_r[idx, idy] += (
                GRC[i, j, 0] * GR[i, j, 1]
              + GRC[i, j, 1] * GR[i, j, 0]
            )

            count[idx, idy] += 1

    mask = count > 0
    ZZ_r[mask]  /= count[mask]
    ZXY_r[mask] /= count[mask]

    ZZ2XY_r = ZZ_r + 2.0 * ZXY_r

    return ZZ_r, ZXY_r, ZZ2XY_r





def spin_spin_corr_full_bc(GR, GRC, Lx, Ly):
    Nsite = Lx * Ly

    ZZ = np.zeros((Nsite, Nsite), dtype=np.float64)
    ZXY = np.zeros((Nsite, Nsite), dtype=np.float64)

    for i in range(Nsite):
        for j in range(Nsite):
            ZZ[i, j] = (
                GRC[i, j, 0] * GR[i, j, 0]
              + GRC[i, j, 1] * GR[i, j, 1]
              + (GRC[i, i, 1] - GRC[i, i, 0])
                * (GRC[j, j, 1] - GRC[j, j, 0])
            )
            ZXY[i, j] = (
                GRC[i, j, 0] * GR[i, j, 1]
              + GRC[i, j, 1] * GR[i, j, 0]
            )

    ZZ2XY = ZZ + 2.0 * ZXY
    return ZZ, ZXY, ZZ2XY

def density_density_corr_bc(GR, GRC, Lx, Ly, bc_x="periodic", bc_y="periodic"):

    Nsite = Lx * Ly
    loc, label = square_lattice(Lx, Ly)

    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    DD_r = np.zeros((size_x, size_y), dtype=np.float64)
    count = np.zeros((size_x, size_y), dtype=int)

    for i in range(Nsite):
        xi, yi = label[i]
        n_i = GRC[i, i, 0] + GRC[i, i, 1]

        for j in range(Nsite):
            xj, yj = label[j]

            disp = get_displacement(
                xi, yi, xj, yj, Lx, Ly, bc_x, bc_y
            )
            if disp is None:
                continue

            idx, idy = disp
            n_j = GRC[j, j, 0] + GRC[j, j, 1]

            DD_r[idx, idy] += n_i * n_j
            count[idx, idy] += 1

    mask = count > 0
    DD_r[mask] /= count[mask]

    return DD_r


def pair_corr_bc(GR, GRC, Lx, Ly,
              bc_x="periodic", bc_y="periodic",
              pairing='s'):

    Nsite = Lx * Ly
    loc, label = square_lattice(Lx, Ly)

    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    P_r = np.zeros((size_x, size_y), dtype=np.complex128)
    count = np.zeros((size_x, size_y), dtype=int)

    def form_factor(dx, dy, pairing):
        if pairing == 's':
            return 1.0
        elif pairing == 'px':
            return dx
        elif pairing == 'py':
            return dy
        elif pairing == 'd':
            return dx**2 - dy**2
        else:
            raise ValueError("Unknown pairing")

    for i in range(Nsite):
        xi, yi = label[i]
        delta_i = GRC[i, i, 0] * GRC[i, i, 1]

        for j in range(Nsite):
            xj, yj = label[j]

            disp = get_displacement(
                xi, yi, xj, yj, Lx, Ly, bc_x, bc_y
            )
            if disp is None:
                continue

            idx, idy = disp

            # recover physical dx, dy for form factor
            dx = xj - xi
            dy = yj - yi

            delta_j = GRC[j, j, 0] * GRC[j, j, 1]

            P_r[idx, idy] += (
                form_factor(dx, dy, pairing)
                * delta_i * np.conj(delta_j)
            )

            count[idx, idy] += 1

    mask = count > 0
    P_r[mask] /= count[mask]

    return P_r


#ZZ, ZXY, Z = spin_spin_corr_bc(GR, GRC, Lx, Ly, self.bc_x, self.bc_y)
# Open / Open
#ZZ, ZXY, Z = spin_spin_corr_bc(GR, GRC, Lx, Ly, bc_x="open", bc_y="open")
# Periodic / Periodic
#ZZ, ZXY, Z = spin_spin_corr_bc(GR, GRC, Lx, Ly, bc_x="periodic", bc_y="periodic")
# Open / Periodic
#ZZ, ZXY, Z = spin_spin_corr_bc(GR, GRC, Lx, Ly, bc_x="open", bc_y="periodic")
# Periodic / Open
#ZZ, ZXY, Z = spin_spin_corr_bc(GR, GRC, Lx, Ly, bc_x="periodic", bc_y="open")