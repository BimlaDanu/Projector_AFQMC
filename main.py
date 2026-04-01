import numpy as np
import scipy.linalg as la
import time
import math
from copy import deepcopy
#import scipy.linalg
#scipy.linalg.solve
#import os
#import torch
#import torch.linalg as tla
from hamiltonian_main_mod import Hamiltonian
from hop_mod import Hop_mod_symm
from wrap_mod import Wrap
from cgr_mod import GR_init, GR_fun#, UDV_init, GR_fun1
from wrapGR_mod import WrapGRup0, WrapGRdo0
from stack_mod import Update_stack_up, Update_stack_do
from control_mod import Control_precisionG, Control_precisionP
from obser import Obser


def Main(self, This, tun_params):
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    N_FL = self.N_FL
    Nbin = self.Nbin
    Nsweep = self.Nsweep
    Nwrap = self.Nwrap
    Theta = self.Theta
    CPU_MAX = self.CPU_MAX
    N_skip = self.N_skip
    #print('self.bcx, self.bcy, self.ladder =',self.bc_x, self.bc_y, self.ladder)

    
    deltaG_threshold = 10
    obs_scal_len = 7
    obs_eq_len = 5
    obs_geq_len = 1
    #nobs =  obs_scal_len
    
    if not tun_params:
        Theta = Theta
    else:
        Theta = This

    if CPU_MAX > 0:
        Nbin = 1e4
    Nbin_eff = Nbin
    
    N_skip = N_skip
    nbin_meas = Nbin - N_skip
    
    K, Bk, inv_Bk, Bk_root, inv_Bk_root, P, lambda_array, Ltrot = Hamiltonian(self, This, tun_params)
    
    #print('K =',K)
    
    phase_avg_barray = np.zeros(nbin_meas, dtype = np.float32)    
    Weight_avg_barray = np.zeros(nbin_meas, dtype = np.float32)    

    obs_geq_avg_blist = np.zeros((obs_geq_len, nbin_meas, Ndim,  Ndim, N_FL), dtype = np.float32)
    
    if not self.Corr_all:
        size_x = self.Lx if self.bc_x == "periodic" else 2 * self.Lx - 1
        size_y = self.Ly if self.bc_y == "periodic" else 2 * self.Ly - 1
        obs_eq_avg_blist = np.zeros((obs_eq_len, nbin_meas, size_x,  size_y), dtype = np.float32)
    else:
        obs_eq_avg_blist = np.zeros((obs_eq_len, nbin_meas, Ndim,  Ndim), dtype = np.float32)
    
        
    obs_scal_avg_blist = np.zeros((obs_scal_len, nbin_meas), dtype = np.float32)

  
    nstm = math.ceil(Ltrot / Nwrap)
    lobs = int(Ltrot / 2 - 1)
    if (lobs + 1) % Nwrap != 0:
        #print('check')
        nstm += 1

    stab_up = np.full(Ltrot, False)
    stab_up[(Nwrap - 1)::Nwrap] = True
    stab_up[-1] = True
    stab_up[lobs] = True
    stab_do = np.roll(stab_up, 1)
    #stab_do = np.roll(stab_up, Ltrot % Nwrap)
    len_array = np.diff(np.insert(np.where(stab_up)[0], 0, -1))
    #print('  nstm , stab_up=',  nstm , stab_up)

    # -------------------------------
    #nstm = len(len_array)
    #print('nstm,len(len_array) =',nstm,len(len_array))
    
    
    #len_array = np.ones(nstm) * Nwrap   
    #len_array[-1] = Ltrot - (nstm - 1) * Nwrap
    #len_array = len_array.astype(np.int32)
    
    # Initialize auxiliary field with random configuration (row for site number, column for time slice)
    rng = np.random.default_rng()
    HS_field = 2 * rng.integers(2, size = [Ndim, Ltrot]) - 1 # Convert [0 1] to [-1 1]
    hv = HS_field @ np.diag(lambda_array)
    #print('hv.shape, HS_field.shape, lambda_array.shape =',hv.shape, HS_field.shape, lambda_array.shape)
    

    F1 = {'U': 1., 'D': 1., 'V': 1.}
    #F1_list = np.full(n_fl, F1)
    F1_list = [F1.copy() for _ in range(N_FL)]
    #print('F1_list =',   F1_list)        
    FP_list = []
    for nf in range(N_FL):
        P_slice = P[:, :, nf]  # Extract the i-th slice of P
        FP = Wrap(self, P_slice, F1.copy())  # Process each slice independently
        FP_list.append(FP)
        
    #for i, fp in enumerate(FP_list):  
    #    print(f"FP state {i}: {fp}")
    

    GR, phase, UDVst, Weight = GR_init(self, Bk, hv, nstm, Ltrot, stab_do, P, FP)

    stab_count = 0
    
    acceptance_rate = 0.0
    propose_count = 0
    
    deltaG_max = 0.
    deltaG_mean = 0.
    deltaP_max = 0.
    deltaP_mean = 0.
    t_total_elapsed = 0
    nst = 0
    for bin in range(Nbin):
        t_bin_start = time.time()
        bin_count = bin - N_skip
        if self.verbose:
            print(f'\nBin {bin + 1}, running for this input {This}')
        
        #Initilization     
        Weight_sum = 0.
        phase_sum = 0.
        N_meas =0
        obs_scal_phase_sum_list = np.zeros(obs_scal_len, dtype = np.float32)   
        size_x = self.Lx if self.bc_x == "periodic" else 2 * self.Lx - 1
        size_y = self.Ly if self.bc_y == "periodic" else 2 * self.Ly - 1
        if not self.Corr_all:
            obs_eq_phase_sum_list = np.zeros((obs_eq_len, size_x,  size_y), dtype = np.float32)  
        else:
             obs_eq_phase_sum_list = np.zeros((obs_eq_len, Ndim,  Ndim), dtype = np.float32)  
  
            
        obs_geq_phase_sum_list = np.zeros((obs_geq_len, Ndim, Ndim, N_FL), dtype = np.float32) 
     
        
        Nsweep = self.Nsweep
        for sweep in range(Nsweep):
            if self.verbose and Nsweep % 50 == 50 - 1:
                print(f'Sweep {Nsweep + 1} out of {Nsweep}')
            #UDVr = F1_list.copy()
            UDVr = deepcopy(F1_list)
            l_end = -1
            for l in range(Ltrot):
                phase, acceptance_rate, propose_count = WrapGRup0(self, GR, Bk, inv_Bk, hv, phase, rng, Weight, l, acceptance_rate, propose_count)
                if stab_up[l]:
                    stab_count += 1
                    len1 = len_array[nst]
                     # Update stack and nst; UDVst and UDVr changed by mutation
                    l_start = l_end + 1
                    l_end = l
                    #print('1, l, len1, nst, nstm, stab_up[l],l_start,l_end  =', l, len1, nst, nstm, stab_up[l],l_start,l_end)
                    _, _, UDVl, nst = Update_stack_up(self, Bk[l_start:(l_end + 1), :, :, :], hv[:,l_start:(l_end + 1)], UDVst, len1, P, UDVr, FP_list, nst, nstm, l_start, l_end)

                    # Calculate the equal-time Green's function from stack
                    GR_test = GR
                    phase_test = phase
                    Weight_test = Weight
                    GR, phase, Weight = GR_fun(self, UDVr, UDVl)
                    deltaG_max, deltaG_mean = Control_precisionG(self, GR, GR_test, deltaG_threshold, deltaG_max, deltaG_mean, stab_count)
                    deltaP_max, deltaP_mean = Control_precisionP(self, phase, phase_test, deltaP_max, deltaP_mean, stab_count)
                if bin_count >= 0 and l == lobs:
                    ##################### MEASUREMENTS FOR CURRENT AUXILIARY FIELD CONFIGURATION  #####################
                    if self.Sym:
                        GR_tilde = Hop_mod_symm(self, GR.copy(), Bk_root, inv_Bk_root,l)
                    else:
                        GR_tilde = GR
                    obs_scal_avg_list, obs_eq_avg_list, obs_geq_avg_list,  obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, N_meas= Obser(self, GR_tilde, phase, Weight, obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, obs_scal_len, obs_eq_len, obs_geq_len,  N_meas)


            #UDVl = F1_list.copy()
            UDVl = deepcopy(F1_list)
            l_start = Ltrot
            
            for l in range(Ltrot - 1, -1, -1):
                phase, acceptance_rate, propose_count =   WrapGRdo0(self, GR, Bk, inv_Bk, hv, phase, rng, Weight,  l, acceptance_rate, propose_count)
  
            
                if stab_do[(l)]:
                    stab_count += 1
                    len1 = len_array[nst + 1]
                    # Update stack and nst; UDVst and UDVl changed by mutation
                    l_end = l_start - 1
                    l_start = l
                    
                    _, UDVr, _, nst = Update_stack_do(self, Bk[l_start:(l_end + 1),:,:,:], hv[:,l_start:(l_end + 1)], UDVst, len1, P, UDVl, FP_list, nst, nstm, l_start, l_end)
                    
                    
                    GR_test = GR
                    phase_test = phase
                    Weight_test = Weight
                    GR, phase, Weight = GR_fun(self, UDVr, UDVl)
                    deltaG_max, deltaG_mean = Control_precisionG(self, GR, GR_test, deltaG_threshold, deltaG_max, deltaG_mean, stab_count)
                    deltaP_max, deltaP_mean = Control_precisionP(self, phase, phase_test, deltaP_max, deltaP_mean, stab_count)

    
                if bin_count >= 0 and l == lobs+1:
                    ##################### MEASUREMENTS FOR CURRENT AUXILIARY FIELD CONFIGURATION  #####################
                    if self.Sym:
                        GR_tilde = Hop_mod_symm(self, GR.copy(), Bk_root, inv_Bk_root,l)
                    else:
                        GR_tilde = GR
                    obs_scal_avg_list, obs_eq_avg_list, obs_geq_avg_list,  obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, N_meas = Obser(self, GR_tilde, phase, Weight, obs_scal_phase_sum_list, obs_eq_phase_sum_list, obs_geq_phase_sum_list, phase_sum, Weight_sum, obs_scal_len, obs_eq_len, obs_geq_len,  N_meas)
        
        if bin_count >= 0:
            
            phase_avg_barray[bin_count] = phase_sum #/ Nsweep
            Weight_avg_barray[bin_count]  = Weight_sum #/ Nsweep
            #print('bin_count, Weight_avg_barray[bin_count] =',bin_count, Weight_avg_barray[bin_count])

            # List of all average scalar observables
            for j in range(obs_scal_len):
                obs_scal_avg_blist[j, bin_count] = obs_scal_avg_list[j]
                
            # List of all average equal time observables    
            for j in range(obs_eq_len):
                obs_eq_avg_blist[j, bin_count] = obs_eq_avg_list[j]
                
            # List of all average equal time observables    
            for j in range(obs_geq_len):
                obs_geq_avg_blist[j, bin_count] = obs_geq_avg_list[j]

                
        #print('obs_avg_scal_list.shape, obs_avg_eq_list.shape, phase_avg_array.shape =',obs_avg_scal_list.shape, obs_avg_eq_list.shape, phase_avg_array.shape)
        t_bin_elapsed = time.time() - t_bin_start
        t_total_elapsed += t_bin_elapsed
    
        if (CPU_MAX > 0) and (t_total_elapsed + 1.5 * t_bin_elapsed > 3600 * CPU_MAX):
            Nbin_eff = bin
            return
    return K, Nbin_eff,obs_scal_avg_blist, obs_eq_avg_blist, obs_geq_avg_blist, phase_avg_barray, Weight_avg_barray, deltaG_max, deltaG_mean, deltaP_max, deltaP_mean, t_total_elapsed, acceptance_rate
