import numpy as np
import scipy.linalg as la
from hopping_ham_mod import hubbard_hopping_nf
from trial_wavefunction_mod import trial_wavefunction_Hubbard
from hopping_ham_mod import square_lattice

from spin2_den2_pair_corr import *
from spin2_den2_pair_corr_bc import *
from  spin_spin_corr_all import *
#from  Hubbard_mord import *
#import scipy.linalg
#scipy.linalg.solve
#import math
#import os
#import torch
#import torch.linalg as tla
#import time
 
def Obser(self,  GR, phase, Weight, obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, obs_scal_len, obs_eq_len, obs_geq_len, N_meas):
        N_FL = self.N_FL
        Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
        K = hubbard_hopping_nf(self.Ham_t, self.Ham_tp, self.Ham_mu, self.Lx, self.Ly, self.N_FL, self.bc_x, self.bc_y, self.ladder)
        # Initialization
        GRC = np.zeros_like(GR, dtype=np.float32)
        #n_list = np.zeros((N_FL, Ndim), dtype=np.float32)
        n_list = np.empty(N_FL, dtype = object)
        # Observables: scalar quantities (obs_scal_len should be 1D)
        obs_scal_list = np.zeros(obs_scal_len, dtype=np.float32)
        obs_scal_avg_list = np.zeros(obs_scal_len, dtype=np.float32)

        # Equal time observables (obs_eq_len might involve 4D tensors)
        
        obs_geq_list = np.zeros((obs_geq_len,  Ndim ,  Ndim , N_FL), dtype=np.float32)
        obs_geq_avg_list = np.zeros((obs_geq_len, Ndim , Ndim , N_FL), dtype=np.float32)
        
        N_meas += 1
        
        Re_phase = np.real(phase)
        AbsRe_phase = np.abs(Re_phase)
        Zp_sign = phase / AbsRe_phase
        sign = Re_phase / AbsRe_phase
        
          
        id = np.identity(Ndim)
        #phase_sum += phase
        Weight_sum += np.log(Weight)
        phase_sum = ((N_meas - 1) * phase_sum + sign) / N_meas
        
        
        
        #Weight_sum += Weight

        # Calculate GRC and n_list for all flavors (nf)
        for nf in range(N_FL):
            GRC[:, :, nf] = id - GR[:,:,nf].T
            n_list[nf] = np.diag(GRC[:,:,nf])
            
        obs_geq_list[0] = GRC
     
        # Calculate scalar observables
        nu_nd = np.ones_like(n_list[0])  # Initialization
        for nf in range(N_FL):  # Loop over spin flavors
            nu_nd *= n_list[nf]  # Element-wise multiplication for density profile
            #obs_scal_list[2] += np.sum(np.trace(np.matmul(K[:, :, nf], GRC[:, :, nf])))  # Kinetic energy
            #obs_scal_list[2] += np.trace(K[:, :, nf] @ GRC[:, :, nf])  # Kinetic energy

        obs_scal_list[0] = np.sum(nu_nd)  # Density profile
        #obs_scal_list[1] = self.Ham_U * obs_scal_list[0]  # Potential energy 
        #obs_scal_list[3] = obs_scal_list[2] + obs_scal_list[1]  # Total energy = Kinetic energy + Potential energy 
        

        obs_scal_list[3],   obs_scal_list[2],   obs_scal_list[1]  = Eloc_fun(self, GR, K)
        
        
        
        n_up = np.diag(GRC[:, :, 0])
        n_dn = np.diag(GRC[:, :, 1])
        sz = (n_up - n_dn) / 2 # magnetization
        rho = (n_up + n_dn) / 2 # density
        obs_scal_list[4] =  np.sum(sz)
        obs_scal_list[5] =  np.sum(rho)


        #Zp_sign
        # Update observables with phase factors
        for j in range(obs_scal_len):
            #obs_scal_phase_sum_list[j] += phase * obs_scal_list[j]
            #obs_scal_phase_sum_list[j] += Zp_sign * obs_scal_list[j]
            
            obs_scal_phase_sum_list[j] = ((N_meas - 1) * obs_scal_phase_sum_list[j]  + np.real(Zp_sign * obs_scal_list[j])) / N_meas
            obs_scal_avg_list[j] = obs_scal_phase_sum_list[j] / phase_sum

        # Handle equilibrium observables (multidimensional arrays)
        for j in range(obs_geq_len):
            #obs_geq_phase_sum_list[j] += phase * obs_geq_list[j]
            #obs_geq_phase_sum_list[j] += Zp_sign * obs_geq_list[j]
            
            obs_geq_phase_sum_list[j] =  ((N_meas - 1) * obs_geq_phase_sum_list[j] + np.real(Zp_sign * obs_geq_list[j])) / N_meas
            obs_geq_avg_list[j] = obs_geq_phase_sum_list[j] / phase_sum
            
        # ====== Initialize r-resolved spin observables ======  
        #obs_eq_list = np.zeros((obs_eq_len, self.Lx, self.Ly), dtype=np.float32)
        #obs_eq_avg_list = np.zeros((obs_eq_len, self.Lx, self.Ly), dtype=np.float32)
        # ====== Initialize r-resolved spin observables ======    
       # ZZ_r, ZXY_r, ZZ2XY_r = spin_spin_corr(GR, GRC, self.Lx, self.Ly)  # r-resolved arrays
        #ZZ_r, ZXY_r, ZZ2XY_r = spin_spin_corr_placeholder(GR, GRC, self.Lx, self.Ly)  # r-resolved arrays
        
        #obs_eq_list[0] += ZZ_r * phase
        #obs_eq_list[1] += ZXY_r * phase
        #obs_eq_list[2] += ZZ2XY_r * phase
        
        # ====== Initialize r-resolved spin observables ======  
        if not self.Corr_all:
            size_x = self.Lx if self.bc_x == "periodic" else 2 * self.Lx - 1
            size_y = self.Ly if self.bc_y == "periodic" else 2 * self.Ly - 1
            obs_eq_list = np.zeros((obs_eq_len,  size_x,  size_y), dtype=np.float32)
            obs_eq_avg_list = np.zeros((obs_eq_len,  size_x,  size_y), dtype=np.float32)
        else:
            obs_eq_list = np.zeros((obs_eq_len,  Ndim,  Ndim), dtype=np.float32)
            obs_eq_avg_list = np.zeros((obs_eq_len,  Ndim,  Ndim), dtype=np.float32)
   
            
            # ====== Initialize r-resolved spin observables ======  
          
        #if not self.ladder:
        #    ZZ_r, ZXY_r, ZZ2XY_r = spin_spin_corr(GR, GRC, self.Lx, self.Ly) #spin_spin_corr(torch_GR, GRC, self.Lx, self.Ly)  # r-resolved arrays    
        #else: 
        #    ZZ_r, ZXY_r, ZZ2XY_r  = spin_spin_corr_bc(GR, GRC, self.Lx, self.Ly, self.bc_x, self.bc_y)  
        
        if not self.Corr_all:
            #ZZ_r, ZXY_r, ZZ2XY_r  = spin_spin_corr_bc(GR, GRC, self.Lx, self.Ly, self.bc_x, self.bc_y) 
            ZZ_r, ZXY_r, ZZ2XY_r  = spin_spin_corr(GR, GRC, self.Lx, self.Ly)    
            DD_r = density_density_corr(GR, GRC, self.Lx, self.Ly)
            P_r = pair_corr(GR, GRC, self.Lx, self.Ly, pairing='s')
        else: 
            ZZ_r, ZXY_r, ZZ2XY_r  = spin_spin_corr_full_bc(GR, GRC, self.Lx, self.Ly) 
            DD_r  = density_density_corr_full_bc(GR, GRC, self.Lx, self.Ly) 
            P_r  = pair_pair_corr_full_bc(GR, GRC, self.Lx, self.Ly, pairing='s') 
    
        
        obs_eq_list[0] += ZZ_r * Zp_sign
        obs_eq_list[1] += ZXY_r * Zp_sign
        obs_eq_list[2] += ZZ2XY_r * Zp_sign
        obs_eq_list[3] += DD_r
        obs_eq_list[4] += P_r
        
        
        # ====== Update phase sums and averages ======
        for i in range(obs_eq_len):
            #obs_eq_phase_sum_list[i] += obs_eq_list[i]
            obs_eq_phase_sum_list[i] = ((N_meas - 1) *obs_eq_phase_sum_list[i] + obs_eq_list[i]) / N_meas
            obs_eq_avg_list[i] = obs_eq_phase_sum_list[i] / phase_sum
            
        return obs_scal_avg_list, obs_eq_avg_list, obs_geq_avg_list, obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, N_meas

def Eloc_fun(self, torch_GR, torch_K):
    # Number of sites
    U = self.Ham_U
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    torch_GRC = np.zeros_like(torch_GR)
    torch_id = np.eye(Ndim, dtype = torch_GR.dtype)
    for nf in range(N_FL):
        # <<ci^\dagger cj>>_C
        torch_GRC[:, :, nf] = torch_id - torch_GR[:, :, nf].T
        
    # Compute density per site
    torch_nu_nd_total = np.ones(Ndim, dtype=float)
    for nf in range(N_FL):
        torch_nu_nd_total *= np.diag(torch_GRC[:, :, nf])
    # Kinetic energy
    torch_KE = 0.0
    torch_PE = 0.0
    for nf in range(N_FL):
        torch_KE += np.trace(torch_K[:, :, nf] @ torch_GRC[:, :, nf])  # select flavor slice
    # Potential energy
    torch_PE = U * np.sum(torch_nu_nd_total)
    # Total energy
    torch_Eloc = torch_KE + torch_PE
    return torch_Eloc, torch_KE, torch_PE





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


 
 
 