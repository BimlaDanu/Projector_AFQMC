import numpy as np
import os

# --------------------------------------------------------
# Convert array indices or displacements to real-space dx, dy
# --------------------------------------------------------
def get_real_displacement(dx, dy, Lx, Ly, bc_x, bc_y):
    if bc_x == "periodic":
        dx_p = dx
        if dx_p > Lx // 2:
            dx_p -= Lx
    elif bc_x == "open":
        dx_p = dx - (Lx - 1)
    else:
        raise ValueError("bc_x must be 'open' or 'periodic'")

    if bc_y == "periodic":
        dy_p = dy
        if dy_p > Ly // 2:
            dy_p -= Ly
    elif bc_y == "open":
        dy_p = dy - (Ly - 1)
    else:
        raise ValueError("bc_y must be 'open' or 'periodic'")

    return dx_p, dy_p

# --------------------------------------------------------
# Compute full Nsite x Nsite spin-spin correlations
# --------------------------------------------------------
def spin_spin_corr_full_bc(GR, GRC, Lx, Ly):
    Nsite = Lx * Ly
    ZZ = np.zeros((Nsite, Nsite), dtype=np.float64)
    ZXY = np.zeros((Nsite, Nsite), dtype=np.float64)

    for i in range(Nsite):
        for j in range(Nsite):
            ZZ[i, j] = (
                GRC[i, j, 0] * GR[i, j, 0] +
                GRC[i, j, 1] * GR[i, j, 1] +
                (GRC[i, i, 1] - GRC[i, i, 0]) *
                (GRC[j, j, 1] - GRC[j, j, 0])
            )
            ZXY[i, j] = (
                GRC[i, j, 0] * GR[i, j, 1] +
                GRC[i, j, 1] * GR[i, j, 0]
            )

    ZZ2XY = ZZ + 2.0 * ZXY
    return ZZ/4., ZXY/4., ZZ2XY/4.

# --------------------------------------------------------
# Compute full Nsite x Nsite density-density correlations
# --------------------------------------------------------
def density_density_corr_full_bc(GR, GRC, Lx, Ly):
    Nsite = Lx * Ly
    DD_r = np.zeros((Nsite, Nsite), dtype=np.float64)

    for i in range(Nsite):
        for j in range(Nsite):
            n_i = GRC[i, i, 0] + GRC[i, i, 1]
            n_j = GRC[j, j, 0] + GRC[j, j, 1]
            DD_r[i, j] = (n_i * n_j) - ((GR[i, i, 0] + GR[i, i, 1]) * (GR[j, j, 0] + GR[j, j, 1]))
    return DD_r

# --------------------------------------------------------
# Compute full Nsite x Nsite pair-pair correlations
# --------------------------------------------------------
def pair_pair_corr_full_bc(GR, GRC, Lx, Ly, pairing='s'):
    Nsite = Lx * Ly
    P_r = np.zeros((Nsite, Nsite), dtype=np.float64)
    
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
        for j in range(Nsite):
            # Pair-pair correlation: <c_i↑ c_i↓ c_j↓† c_j↑†>
            # Approximated as product of anomalous Greens functions
            delta_i = GRC[i, i, 0] * GRC[i, i, 1]  # local pair
            delta_j = GRC[j, j, 0] * GRC[j, j, 1]  # local pair
            P_r[i, j] = form_factor(i, j, pairing) * delta_i * np.conj(delta_j)
    return P_r
# --------------------------------------------------------
# Save spin-spin correlations with distances r
# --------------------------------------------------------
def save_spin_correlations_full_r(self, 
                                   observable_full_array,
                                   observable_std_array,
                                   Ndim,
                                   bc_x,
                                   bc_y,
                                   epoch):
    """
    Save full Nsite x Nsite correlations with real-space distance r.
    """
    Lx, Ly = self.Lx, self.Ly
    Nsite = Lx * Ly

    os.makedirs("Data_storage", exist_ok=True)

    def write_file(filename, corr_array, std_array):
        path = os.path.join("Data_storage", filename)
        with open(path, "w") as f:
            f.write("# dx dy r corr std\n")
            for i in range(Nsite):
                xi = i % Lx
                yi = i // Lx
                for j in range(Nsite):
                    xj = j % Lx
                    yj = j // Lx

                    dx = xj - xi
                    dy = yj - yi
                    dx_p, dy_p = get_real_displacement(dx, dy, Lx, Ly, bc_x, bc_y)
                    r = np.sqrt(dx_p**2 + dy_p**2)
                    corr = corr_array[i, j]
                    std = std_array[i, j]

                    f.write(f"{dx_p:4d} {dy_p:4d} {r:10.6f} {corr: .6f} {std: .6f}\n")

    # Unpack arrays
    avg_spinZZ_total = observable_full_array[0]
    avg_spinXY_total = observable_full_array[1]
    avg_spinT_total  = observable_full_array[2]

    avg_spinZZ_std = observable_std_array[0]
    avg_spinXY_std = observable_std_array[1]
    avg_spinT_std  = observable_std_array[2]

    # Save files
    write_file(f"spin-spin-zz_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
               avg_spinZZ_total, avg_spinZZ_std)

    write_file(f"spin-spin-xy_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
               avg_spinXY_total, avg_spinXY_std)

    write_file(f"spin-total_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
               avg_spinT_total, avg_spinT_std)
    
    
    
# --------------------------------------------------------
# Save spin-spin correlations as (i, j, r=j-i>0)
# --------------------------------------------------------
def save_spin_correlations_full_r_1(self, 
                                  observable_full_array,
                                  observable_std_array,
                                  Ndim,
                                  bc_x,
                                  bc_y,
                                  epoch):
    """
    Save full Nsite x Nsite correlations.
    Output columns:
    i  j  r=j-i  corr  std
    Only for j > i (so r > 0).
    """

    Lx, Ly = self.Lx, self.Ly
    Nsite = Lx * Ly

    os.makedirs("Data_storage", exist_ok=True)

    def write_file(filename, corr_array, std_array):

        path = os.path.join("Data_storage", filename)

        with open(path, "w") as f:

            f.write("# i  j  r=j-i  corr  std\n")

            for i in range(Nsite):
                for j in range(i + 1, Nsite):

                    r = j - i

                    corr = corr_array[i, j]
                    std  = std_array[i, j]

                    # print sites starting from 1
                    f.write(
                        f"{i+1:4d} "
                        f"{j+1:4d} "
                        f"{r:4d} "
                        f"{corr: .8f} "
                        f"{std: .8f}\n"
                    )

    # Unpack arrays
    avg_spinZZ_total = observable_full_array[0]
    avg_spinXY_total = observable_full_array[1]
    avg_spinT_total  = observable_full_array[2]

    avg_spinZZ_std = observable_std_array[0]
    avg_spinXY_std = observable_std_array[1]
    avg_spinT_std  = observable_std_array[2]

    # Save files
    write_file(
        f"spin-spin-zz_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
        avg_spinZZ_total,
        avg_spinZZ_std
    )

    write_file(
        f"spin-spin-xy_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
        avg_spinXY_total,
        avg_spinXY_std
    )

    write_file(
        f"spin-total_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt",
        avg_spinT_total,
        avg_spinT_std
    )
    
def save_den_correlations_full_r_1(self, DD_array, DD_std_array, Ndim, bc_x, bc_y, epoch):
    """
    Save full Nsite x Nsite correlations.
    Output columns:
    i  j  r=j-i  corr  std
    Only for j > i (so r > 0).
    """
    Lx, Ly = self.Lx, self.Ly
    Nsite = Lx * Ly
    os.makedirs("Data_storage", exist_ok=True)
    def write_file(filename, corr_array, std_array):
        path = os.path.join("Data_storage", filename)
        with open(path, "w") as f:
            f.write("# i  j  r=j-i  corr  std\n")
            for i in range(Nsite):
                for j in range(i + 1, Nsite):
                    r = j - i
                    corr = corr_array[i, j]
                    std  = std_array[i, j]
                    # print sites starting from 1
                    f.write(
                        f"{i+1:4d} "
                        f"{j+1:4d} "
                        f"{r:4d} "
                        f"{corr: .8f} "
                        f"{std: .8f}\n")
    write_file(
        f"density-density_Ndim{Ndim}_NPART{self.N_part}_U{self.Ham_U}_epoch{epoch}_file.txt",
        DD_array, DD_std_array)

def save_pair_correlations_full_r_1(self, P_array, P_std_array, Ndim, bc_x, bc_y, epoch, pairing='s'):
    """
    Save full Nsite x Nsite correlations.
    Output columns:
    i  j  r=j-i  corr  std
    Only for j > i (so r > 0).
    """
    Lx, Ly = self.Lx, self.Ly
    Nsite = Lx * Ly
    os.makedirs("Data_storage", exist_ok=True)
    def write_file(filename, corr_array, std_array):
        path = os.path.join("Data_storage", filename)
        with open(path, "w") as f:
            f.write("# i  j  r=j-i  corr  std\n")
            for i in range(Nsite):
                for j in range(i + 1, Nsite):
                    r = j - i
                    corr = corr_array[i, j]
                    std  = std_array[i, j]
                    
                    if P_std_array is not None:
                        std = P_std_array[i, j]
                    else:
                        std = 0.0 + 0.0j
                    
                    # print sites starting from 1
                    f.write(
                        f"{i+1:4d} "
                        f"{j+1:4d} "
                        f"{r:4d} "
                        f"{corr: .8f} "
                        f"{std: .8f}\n")
    write_file(
        f"pair-{pairing}_{Ndim}_{self.N_part}_{self.Ham_U}_{epoch}_file.txt",
        P_array, P_std_array)
