import numpy as np
from hopping_ham_mod import square_lattice
import os


def save_spin_correlations_placeholder(self, observable_eq_array, observable_eq_std_array, Ndim):
    """
    Save r-resolved spin-spin correlations (ZZ, XY, T).
    Coordinates are relative displacements (dx, dy) with periodic distance r.
    
    File format:
    dx dy r corr std
    """
    avg_spinZZ_total = observable_eq_array[0]
    avg_spinXY_total = observable_eq_array[1]
    avg_spinT_total  = observable_eq_array[2]

    avg_spinZZ_total_std = observable_eq_std_array[0]
    avg_spinXY_total_std = observable_eq_std_array[1]
    avg_spinT_total_std  = observable_eq_std_array[2]

    Lx, Ly = self.Lx, self.Ly
    Data_folder_name = f"Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    # ====== ZZ ======
    filename_zz = (
        f"spin-spin-zz_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name, filename_zz)
    #with open(filename_zz, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p = min(dx, Lx - dx)
                dy_p = min(dy, Ly - dy)
                r = np.sqrt(dx_p**2 + dy_p**2)

                corr = avg_spinZZ_total[dx, dy]
                std  = avg_spinZZ_total_std[dx, dy]

                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

    # ====== XY ======
    filename_xy = (
        f"spin-spin-xy_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name, filename_xy)
    #with open(filename_xy, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p = min(dx, Lx - dx)
                dy_p = min(dy, Ly - dy)
                r = np.sqrt(dx_p**2 + dy_p**2)

                corr = avg_spinXY_total[dx, dy]
                std  = avg_spinXY_total_std[dx, dy]

                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

    # ====== T ======
    filename_T = (
        f"spinT-spinT_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name,  filename_T)
    #with open(filename_T, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p = min(dx, Lx - dx)
                dy_p = min(dy, Ly - dy)
                r = np.sqrt(dx_p**2 + dy_p**2)

                corr = avg_spinT_total[dx, dy]
                std  = avg_spinT_total_std[dx, dy]

                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")
                
def save_density_correlations_placeholder(self, DD_array, DD_std_array, Ndim, epoch):
    """
    Save r-resolved density-density correlations.
    
    File format:
    dx dy r corr std
    """
    Lx, Ly = self.Lx, self.Ly
    _, label = square_lattice(Lx, Ly)
    Data_folder_name = "Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    def min_image(dx, dy):
        """Apply minimum-image convention for periodic boundaries"""
        if dx > Lx // 2:
            dx -= Lx
        if dy > Ly // 2:
            dy -= Ly
        return dx, dy

    filename = f"density-density_Ndim{Ndim}_NPART{self.N_part}_U{self.Ham_U}_epoch{epoch}_file.txt"
    path = os.path.join(Data_folder_name, filename)

    with open(path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                #dx_p, dy_p = min_image(dx, dy)
                dx_p = min(dx, Lx - dx)
                dy_p = min(dy, Ly - dy)
                
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = DD_array[dx, dy]
                std = DD_std_array[dx, dy] if DD_std_array is not None else 0.0
                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

def save_pair_correlations_placeholder(self, P_array, P_std_array, Ndim, epoch, pairing='s'):
    """
    Save r-resolved superconducting pair correlations.
    
    File format:
    dx dy r Re(corr) Im(corr) Re(std) Im(std)
    """
    Lx, Ly = self.Lx, self.Ly
    _, label = square_lattice(Lx, Ly)
    Data_folder_name = "Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    def min_image(dx, dy):
        """Apply minimum-image convention for periodic boundaries"""
        if dx > Lx // 2:
            dx -= Lx
        if dy > Ly // 2:
            dy -= Ly
        return dx, dy

    filename = f"pair-{pairing}_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_epoch{epoch}_file.txt"
    path = os.path.join(Data_folder_name, filename)

    with open(path, "w") as f:
        f.write("# dx dy r Re(corr) Im(corr) Re(std) Im(std)\n")
        for dx in range(Lx):
            for dy in range(Ly):
                #dx_p, dy_p = min_image(dx, dy)
                dx_p = min(dx, Lx - dx)
                dy_p = min(dy, Ly - dy)
                
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = P_array[dx, dy]
                std = P_std_array[dx, dy] if P_std_array is not None else 0.0
                f.write(f"{dx} {dy} {r:.6f} {corr.real:.6f} {corr.imag:.6f} {std.real:.6f} {std.imag:.6f}\n")

def save_spin_correlations(self, observable_eq_array, observable_eq_std_array, Ndim):
    """
    Save r-resolved spin-spin correlations (ZZ, XY, T).
    Coordinates are relative displacements (dx, dy) with periodic distance r.
    
    File format:
    dx dy r corr std
    """

    avg_spinZZ_total = observable_eq_array[0]
    avg_spinXY_total = observable_eq_array[1]
    avg_spinT_total  = observable_eq_array[2]

    avg_spinZZ_total_std = observable_eq_std_array[0]
    avg_spinXY_total_std = observable_eq_std_array[1]
    avg_spinT_total_std  = observable_eq_std_array[2]

    Lx, Ly = self.Lx, self.Ly

    # Generate lattice labels
    _, label = square_lattice(Lx, Ly)
    Data_folder_name = f"Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    def min_image(dx, dy):
        """Apply minimum-image convention for periodic boundaries"""
        if dx > Lx // 2:
            dx -= Lx
        if dy > Ly // 2:
            dy -= Ly
        return dx, dy

    # ====== Save ZZ correlation ======
    filename_zz = (
        f"spin-spin-zz_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name, filename_zz)
    #with open(filename_zz, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p, dy_p = min_image(dx, dy)
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = avg_spinZZ_total[dx, dy]
                std  = avg_spinZZ_total_std[dx, dy]
                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

    # ====== Save XY correlation ======
    filename_xy = (
        f"spin-spin-xy_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name,  filename_xy)
    #with open(filename_xy, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p, dy_p = min_image(dx, dy)
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = avg_spinXY_total[dx, dy]
                std  = avg_spinXY_total_std[dx, dy]
                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

    # ====== Save T correlation ======
    filename_T = (
        f"spinT-spinT_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    )
    info_path = os.path.join(Data_folder_name, filename_T)
    #with open(filename_T, "w") as f:
    with open(info_path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p, dy_p = min_image(dx, dy)
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = avg_spinT_total[dx, dy]
                std  = avg_spinT_total_std[dx, dy]
                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

def save_density_correlations(self, DD_array, DD_std_array=None, Ndim=0):
    """
    Save r-resolved density-density correlations.
    
    File format:
    dx dy r corr std
    """
    Lx, Ly = self.Lx, self.Ly
    _, label = square_lattice(Lx, Ly)
    Data_folder_name = "Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    def min_image(dx, dy):
        """Apply minimum-image convention for periodic boundaries"""
        if dx > Lx // 2:
            dx -= Lx
        if dy > Ly // 2:
            dy -= Ly
        return dx, dy

    filename = f"density-density_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    path = os.path.join(Data_folder_name, filename)

    with open(path, "w") as f:
        f.write("# dx dy r corr std\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p, dy_p = min_image(dx, dy)
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = DD_array[dx, dy]
                std = DD_std_array[dx, dy] if DD_std_array is not None else 0.0
                f.write(f"{dx} {dy} {r:.6f} {corr:.6f} {std:.6f}\n")

def save_pair_correlations(self, P_array, P_std_array=None, Ndim=0, pairing='s'):
    """
    Save r-resolved superconducting pair correlations.
    
    File format:
    dx dy r Re(corr) Im(corr) Re(std) Im(std)
    """
    Lx, Ly = self.Lx, self.Ly
    _, label = square_lattice(Lx, Ly)
    Data_folder_name = "Data_storage"
    os.makedirs(Data_folder_name, exist_ok=True)

    def min_image(dx, dy):
        """Apply minimum-image convention for periodic boundaries"""
        if dx > Lx // 2:
            dx -= Lx
        if dy > Ly // 2:
            dy -= Ly
        return dx, dy

    filename = f"pair-{pairing}_{Ndim}_{self.N_part}_{self.Ham_U}_file.txt"
    path = os.path.join(Data_folder_name, filename)

    with open(path, "w") as f:
        f.write("# dx dy r Re(corr) Im(corr) Re(std) Im(std)\n")
        for dx in range(Lx):
            for dy in range(Ly):
                dx_p, dy_p = min_image(dx, dy)
                r = np.sqrt(dx_p**2 + dy_p**2)
                corr = P_array[dx, dy]
                std = P_std_array[dx, dy] if P_std_array is not None else 0.0
                f.write(f"{dx} {dy} {r:.6f} {corr.real:.6f} {corr.imag:.6f} {std.real:.6f} {std.imag:.6f}\n")
