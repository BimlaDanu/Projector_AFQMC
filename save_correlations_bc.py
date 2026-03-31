import numpy as np
import os

def get_real_displacement1(dx, dy, Lx, Ly, bc_x, bc_y):
    """
    Convert stored displacement indices (dx,dy)
    into physical displacement values.
    """

    # ----- X direction -----
    if bc_x == "periodic":
        dx_p = dx
        if dx_p >= (Lx + 1) // 2:
            dx_p -= Lx
    elif bc_x == "open":
        dx_p = dx - (Lx - 1)
    else:
        raise ValueError("bc_x must be 'open' or 'periodic'")

    # ----- Y direction -----
    if bc_y == "periodic":
        dy_p = dy
        if dy_p >= (Ly + 1) // 2:
            dy_p -= Ly
    elif bc_y == "open":
        dy_p = dy - (Ly - 1)
    else:
        raise ValueError("bc_y must be 'open' or 'periodic'")

    return dx_p, dy_p

def get_real_displacement(dx, dy, Lx, Ly, bc_x, bc_y):
    """
    Convert array indices (dx,dy) into physical displacement
    consistent with boundary conditions.
    """

    # ----- X direction -----
    if bc_x == "periodic":
        dx_p = dx
        if dx_p > Lx // 2:
            dx_p -= Lx
    elif bc_x == "open":
        dx_p = dx - (Lx - 1)
    else:
        raise ValueError("bc_x must be 'open' or 'periodic'")

    # ----- Y direction -----
    if bc_y == "periodic":
        dy_p = dy
        if dy_p > Ly // 2:
            dy_p -= Ly
    elif bc_y == "open":
        dy_p = dy - (Ly - 1)
    else:
        raise ValueError("bc_y must be 'open' or 'periodic'")

    return dx_p, dy_p

def save_spin_correlations_bc(self,
                              observable_eq_array,
                              observable_eq_std_array,
                              Ndim,
                              bc_x,
                              bc_y,
                              epoch):

    Lx, Ly = self.Lx, self.Ly

    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    avg_spinZZ_total = observable_eq_array[0]
    avg_spinXY_total = observable_eq_array[1]
    avg_spinT_total  = observable_eq_array[2]

    avg_spinZZ_std = observable_eq_std_array[0]
    avg_spinXY_std = observable_eq_std_array[1]
    avg_spinT_std  = observable_eq_std_array[2]

    os.makedirs("Data_storage", exist_ok=True)

    def write_file(filename, corr_array, std_array):

        path = os.path.join("Data_storage", filename)

        with open(path, "w") as f:

            f.write("# dx dy r corr std\n")

            for dx in range(size_x):
                for dy in range(size_y):

                    dx_p, dy_p = get_real_displacement(
                        dx, dy, Lx, Ly, bc_x, bc_y
                    )

                    r = np.sqrt(dx_p**2 + dy_p**2)

                    corr = corr_array[dx, dy]
                    std  = std_array[dx, dy]

                    f.write(
                        f"{dx_p:4d} {dy_p:4d} "
                        f"{r:10.6f} "
                        f"{corr: .6f} "
                        f"{std: .6f}\n"
                    )

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
    
def save_density_correlations_bc(self,
                                 DD_array,
                                 DD_std_array,
                                 Ndim,
                                 bc_x,
                                 bc_y,
                                 epoch):

    Lx, Ly = self.Lx, self.Ly

    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    os.makedirs("Data_storage", exist_ok=True)

    filename = (
        f"density-density_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt"
    )

    path = os.path.join("Data_storage", filename)

    with open(path, "w") as f:

        f.write("# dx dy r corr std\n")

        for dx in range(size_x):
            for dy in range(size_y):

                dx_p, dy_p = get_real_displacement(
                    dx, dy, Lx, Ly, bc_x, bc_y
                )

                r = np.sqrt(dx_p**2 + dy_p**2)

                corr = DD_array[dx, dy]

                std = (
                    DD_std_array[dx, dy]
                    if DD_std_array is not None else 0.0
                )

                f.write(
                    f"{dx_p:4d} {dy_p:4d} "
                    f"{r:10.6f} "
                    f"{corr: .6f} "
                    f"{std: .6f}\n"
                )
                
def save_pair_correlations_bc(self,
                              P_array,
                              P_std_array,
                              Ndim,
                              epoch,
                              bc_x,
                              bc_y,
                              pairing='s'):

    Lx, Ly = self.Lx, self.Ly

    size_x = Lx if bc_x == "periodic" else 2 * Lx - 1
    size_y = Ly if bc_y == "periodic" else 2 * Ly - 1

    os.makedirs("Data_storage", exist_ok=True)

    filename = (
        f"pair-{pairing}_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}.txt"
    )

    path = os.path.join("Data_storage", filename)

    with open(path, "w") as f:

        f.write("# dx dy r Re(corr) Im(corr) Re(std) Im(std)\n")

        for dx in range(size_x):
            for dy in range(size_y):

                dx_p, dy_p = get_real_displacement(
                    dx, dy, Lx, Ly, bc_x, bc_y
                )

                r = np.sqrt(dx_p**2 + dy_p**2)

                corr = P_array[dx, dy]

                if P_std_array is not None:
                    std = P_std_array[dx, dy]
                else:
                    std = 0.0 + 0.0j

                f.write(
                    f"{dx_p:4d} {dy_p:4d} "
                    f"{r:10.6f} "
                    f"{corr.real: .6f} {corr.imag: .6f} "
                    f"{std.real: .6f} {std.imag: .6f}\n"
                )