import numpy as np
import scipy.linalg as la
#from  Hubbard_mord import *
from hopping_ham_mod import hubbard_hopping_nf
from trial_wavefunction_mod import trial_wavefunction_Hubbard
import matplotlib.pyplot as plt

def Hamiltonian(self, This, tun_params):
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers   
    N_part = self.N_part
    N_FL = self.N_FL
        
    if self.ham_model == 'Hubbard':                         
        K =   hubbard_hopping_nf(self.Ham_t, self.Ham_tp, self.Ham_mu, self.Lx, self.Ly, self.N_FL, self.bc_x, self.bc_y, self.ladder)
        m_ord =0.
        _, _, P = trial_wavefunction_Hubbard(self.Ham_t, self.Lx, self.Ly, m_ord, self.Ham_mu, self.N_FL, self.N_part, self.bc_x, self.bc_y)
    #elif self.ham_model == 'Periodic_Anderson':
                
    if not tun_params:
        Theta = self.Theta
    else:
        Theta = This

    dtau = self.dtau

    Thtrot = int(round(Theta / dtau))
    Thtrot = max(Thtrot, 1)
    # total Trotter slices: ramp + flat + ramp
    #Ltrot = Thtrot + 2 * Thtrot
    Ltrot =    self.slice_m + 2 * Thtrot
    

    if not self.Hint_tau:
        U_array = self.Ham_U * np.ones(Ltrot)
        lambda_array = np.arccosh(np.exp(U_array * dtau / 2))
        #lambda_array = np.zeros(Ltrot, dtype=np.float64)
        #lambda_array =  np.arccosh(np.exp(self.Ham_U*(dtau/2)))
        plt.plot(lambda_array, marker = 'o', linestyle= 'none')
        plt.savefig(f'lambda_versus_U_{self.Adiabatic}_{Ltrot}.pdf')
        plt.clf()
        #print('1')
    else:
        Adiabatic = True
        #lambda_array, U_array, Ltrot = Lambdat_couplings(self, self.Adiabatic, Theta,  self.slice_m)
        lambda_array, U_array, Ltrot = Lambdat_couplings(self,  Adiabatic, Theta,  self.slice_m)
        #
        plt.plot(lambda_array, marker = 'o', linestyle= 'none')
        plt.savefig(f'lambda_versus_U_{self.Adiabatic}_{Ltrot}.pdf')
        #plt.show()
        plt.clf()
        
 
    ## Time dependent hopping
    #if not self.Adiabatic and  not self.Hop_tau:
    if  not self.Hop_tau:
       # None
        #ti_array[:] = np.ones(Ltrot)
        #ti_array  = dtau
        ti_array = np.zeros(Ltrot, dtype=np.float64)
        ti_array[:] = dtau
        Ltrot = len(ti_array)
        plt.plot(ti_array, marker = 'o', linestyle= 'none')
        plt.savefig(f'hopt_versus_U_{self.Adiabatic}_{Ltrot}.pdf')
        plt.clf()
    else:
        #ti_array, Ltrot1 = Hopt_couplings(self,   self.Adiabatic, Theta,   self.slice_m)
        Adiabatic = True
        ti_array, Ltrot = Hopt_couplings(self,   Adiabatic, Theta,   self.slice_m)
        plt.plot(ti_array, marker = 'o', linestyle= 'none')
        plt.savefig(f'hopt_versus_U_{self.Adiabatic}_{Ltrot}.pdf')
        plt.clf()
        
    #print('len(ti_array), Ltrot, int(Ltrot/2), ti_array, lambda_array  =',len(ti_array), Ltrot, int(Ltrot/2), ti_array, lambda_array)
    if self.ham_model == 'Hubbard':
        info = [Theta, dtau, Ltrot, N_part, self.Ham_t, self.Ham_tp, self.Ham_mu, self.Ham_U, self.Lx, self.Ly, Ndim, self.ham_model]
    elif self.ham_model == 'Periodic_Anderson':
        info = [Theta, dtau, Ltrot, N_part, self.Ham_t, self.Ham_tp, self.Ham_V, self.Ham_muf, self.Ham_muc, self.Ham_Uf, self.Lx, self.Ly, Ndim, self.ham_model]

    if self.verbose:
        print('info, np = ', info)
        
    #print('info, np, self.Hop_tau, self.Hint_tau = ', info, self.Hop_tau, self.Hint_tau)
    print('ham, Ltrot,  lambdaU_array,  ti_array = ',Ltrot, lambda_array, ti_array)
    

    Bk_t, inv_Bk_t, Bk_root_t, inv_Bk_root_t = time_dependent_hopping_nf(self, K, ti_array) 
    return K, Bk_t, inv_Bk_t, Bk_root_t, inv_Bk_root_t, P, lambda_array, Ltrot


def time_dependent_hopping_nf(self, K, ti):
    """
    K  : array of shape (Ndim, Ndim, N_FL)
    ti : iterable of time slices
    """
    Ndim = K.shape[0]
    Nt = len(ti)
    N_FL  = self.N_FL

    Bk_t = np.zeros((Nt, Ndim, Ndim, N_FL))
    inv_Bk_t = np.zeros((Nt, Ndim, Ndim, N_FL))
    Bk_root_t = np.zeros((Nt, Ndim, Ndim, N_FL))
    inv_Bk_root_t = np.zeros((Nt, Ndim, Ndim, N_FL))

    for nl_count, t in enumerate(ti):
        for nf in range(N_FL):
            Bk_t[nl_count, :, :, nf] = la.expm(-K[:, :, nf] * t)
            inv_Bk_t[nl_count, :, :, nf] = la.expm(K[:, :, nf] * t)

            Bk_root_t[nl_count, :, :, nf] = la.expm(-K[:, :, nf] * t / 2)
            inv_Bk_root_t[nl_count, :, :, nf] = la.expm(K[:, :, nf] * t / 2)
    return Bk_t, inv_Bk_t, Bk_root_t, inv_Bk_root_t

def Lambdat_couplings(self, Adiabatic, Theta, slice_m):
    beta = self.Beta
    dtau = self.dtau
    ham_U = self.Ham_U
    Nwrap = self.Nwrap
    # number of slices in ramp
    Thtrot = int(round(Theta / dtau))
    Thtrot = max(Thtrot, 1)
    # total Trotter slices: ramp + flat + ramp
   # Ltrot = Thtrot + 2 * Thtrot
    Ltrot =    slice_m + 2 * Thtrot
    #Ltrot = max(Ltrot, Nwrap)

    #dtau = (beta + Theta) / Ltrot
    dtau = self.dtau
    #print("beta, Theta, Ltrot, dtau =", beta, Theta, Ltrot, dtau)
    # HS coupling (lambda)
    lambda_array = np.zeros(Ltrot, dtype=np.float64)
    # flat middle value ->  measurements slices
    if not  self.hirsch:
        lambda_flat = np.sqrt(dtau*ham_U / 2.0)
    else:
        lambda_flat = np.arccosh(np.exp(ham_U*(dtau/2)))
        
    lambda_array[:] = lambda_flat
    
    # adiabatic ramps
    #if   self.Adiabatic:
    if  Adiabatic:
        for nt in range(1, Thtrot + 1):
            if not self.hirsch:
                val = np.sqrt(dtau*float(nt)/float(Thtrot)*ham_U / 2.0)
            else:
                val = np.arccosh(np.exp(ham_U*(dtau/2)*(float(nt)/float(Thtrot))))
                
            lambda_array[nt-1] = val
            lambda_array[Ltrot - nt] = val
     
    # reconstruct U(t) if needed invert the HS 
    U_array = (2.0 / dtau) * np.log(np.cosh(lambda_array))
    return lambda_array, U_array, Ltrot

def Hopt_couplings(self, Adiabatic, Theta,   slice_m):
    beta = self.Beta
    dtau = self.dtau
    ham_U = self.Ham_U
    Nwrap = self.Nwrap
    # number of slices in ramp
    Thtrot = int(round(Theta / dtau))
    Thtrot = max(Thtrot, 1)
    # total Trotter slices: ramp + flat + ramp
    #Ltrot = Thtrot + 2 * Thtrot
    Ltrot =    slice_m + 2 * Thtrot
    #Ltrot = max(Ltrot, Nwrap)
    # measurements slices
    hamt_array = np.zeros(Ltrot, dtype=np.float64)
    hamt_flat = dtau   
    hamt_array[:] =  hamt_flat
    # adiabatic ramps
    #if   self.Adiabatic:
    if  Adiabatic:
        for nt in range(1, Thtrot + 1):
            val = dtau*float(nt)/float(Thtrot)
            hamt_array[nt-1] = val
            hamt_array[Ltrot - nt] = val
    return hamt_array, Ltrot

