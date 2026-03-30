import numpy as np
import scipy.linalg as la
#import scipy.linalg
#scipy.linalg.solve
#import math
#import os
#import torch
#import torch.linalg as tla
#import time

def Ana(self, Nbin_eff, K, obs_avg_scal_list, obs_avg_eq_list, obs_avg_geq_list, phase_avg_array,  Weight_avg_array):
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    Nbin_meas = Nbin_eff - self.N_skip
    obs_eq_len = 5
    N_FL = self.N_FL
    
    energy_total_array = np.zeros(Nbin_meas, dtype = np.float32)
   
    d_array = np.zeros(Nbin_meas, dtype = np.float32)
    Pot_array = np.zeros(Nbin_meas, dtype = np.float32)
    Kin_array = np.zeros(Nbin_meas, dtype = np.float32)
    Eng_array = np.zeros(Nbin_meas, dtype = np.float32)
    Sz_array = np.zeros(Nbin_meas, dtype = np.float32)
    Rho_array = np.zeros(Nbin_meas, dtype = np.float32)
    
    GRC_avg_list = obs_avg_geq_list[0]
    
    nu_nd_avg_array_list = obs_avg_scal_list[0]
    Pot_avg_array_list = obs_avg_scal_list[1]
    Kin_avg_array_list = obs_avg_scal_list[2]
    Eng_avg_array_list = obs_avg_scal_list[3]
    Sz_avg_array_list = obs_avg_scal_list[4]
    Rho_avg_array_list = obs_avg_scal_list[5]


    for j in range(Nbin_meas):
        nu_nd_sumoversites = nu_nd_avg_array_list[j]
        d_array[j] = nu_nd_sumoversites / Ndim
        energy_total = 0
        for nf in range(N_FL):  # Loop over spin flavors
            energy_total += np.sum(np.multiply(K[:, :, nf], GRC_avg_list[j][:, :, nf]))
        energy_total_array[j] = energy_total + self.Ham_U * nu_nd_sumoversites

        # energy_total_array[j] = self.Ham_U * nu_nd_sumoversites + sum(
        #     np.sum(np.multiply(K[:, :, nf], GRC_avg_list[j][:, :, nf]))
        #    for nf in range(N_FL)
        #)
        Pot_array[j] = Pot_avg_array_list[j]
        Kin_array[j] = Kin_avg_array_list[j]
        Eng_array[j] = Eng_avg_array_list[j]
        
        Sz_array[j] =  Sz_avg_array_list[j]
        Rho_array[j] = Rho_avg_array_list[j]
           

    print('energy_total_array = ',  energy_total_array)
    print('d_array = ',d_array)
    energy_total = np.mean(energy_total_array)
    energy_total_std = np.std(energy_total_array, ddof=1)/np.sqrt(Nbin_meas)

    
    d = np.mean(d_array)
    d_std = np.std(d_array, ddof=1)/np.sqrt(Nbin_meas)

    Pot_total = np.mean(Pot_array)
    Pot_std =   np.std(Pot_array, ddof=1)/np.sqrt(Nbin_meas)

    Kin_total = np.mean(Kin_array)
    Kin_std =   np.std(Kin_array, ddof=1)/np.sqrt(Nbin_meas)
    
    Eng_total = np.mean(Eng_array)
    Eng_std =   np.std(Eng_array, ddof=1)/np.sqrt(Nbin_meas) 
    
    Sz_total = np.mean(Sz_array)
    Sz_total_std = np.std(Sz_array, ddof=1)/np.sqrt(Nbin_meas)
    
    Rho_total = np.mean(Rho_array)
    Rho_total_std = np.std(Rho_array, ddof=1)/np.sqrt(Nbin_meas)

    
    #Eng_total = np.mean(Eng_array)
    #Eng_std   = np.std(Eng_array, ddof=1)
    #Eng_err   = Eng_std / np.sqrt(len(Eng_array))


    
    observable_array = [energy_total, d, Pot_total, Kin_total, Eng_total,  Sz_total,   Rho_total]
    observable_std_array = [energy_total_std, d_std, Pot_std,  Kin_std, Eng_std,  Sz_total_std,  Rho_total_std]
    
    abs_phase = np.mean(np.abs(phase_avg_array))
    abs_phase_std = np.std(np.abs(phase_avg_array), ddof=1)/np.sqrt(Nbin_meas)

    
    abs_Weight = np.mean(Weight_avg_array)
    abs_Weight_std = np.std(Weight_avg_array, ddof=1)/np.sqrt(Nbin_meas)
    #
    #
    #
    #
    #
    #
    # 
   
    # ====== Start equal-time spin-spin correlations ======
    # Infer spatial and internal dimensions directly from data
    # Example shape: (Lx, Ly, 2)
    data_shape = obs_avg_eq_list[0, 0].shape

    spin_total_array = np.zeros((obs_eq_len, Nbin_meas, *data_shape), dtype=np.float64)

    for obs in range(obs_eq_len):
        for bin_idx in range(Nbin_meas):
            spin_total_array[obs, bin_idx, ...] = obs_avg_eq_list[obs, bin_idx]

    # Mean over bins (axis=1)
    avg_spin_total = np.mean(spin_total_array, axis=1)
    # Std over bins
    std_spin_total = np.std(spin_total_array, axis=1, ddof=1)/np.sqrt(Nbin_meas)
    #std_spin_total =  np.std(spin_total_array, axis=1, ddof=1)/np.sqrt(len(spin_total_array.shape[1]))
    
    
    #std_spin_total = np.std(spin_total_array, axis=1, ddof=1)
    #err_spin_total = std_spin_total / np.sqrt(Nbin_meas)
    
    # The bin-to-bin standard deviation is $\sigma(r) = \sqrt{\langle O^2 \rangle - \langle O\rangle^2 }$
    # If bins are statistically independent the error bar on the mean should be:
    # $ SEM (r) =\farc{\sigma(r)}{\sqrt{N_{bin}}}$ 
    # Definitions of ddof $Var = \frac{1}{N-ddof} \sum (x_i-\bar{c})^2$
    #1. It assumes independent samples
    #2. It assumes the estimator is the sample mean
    #3. It assumes roughly Gaussian fluctuations
    # Jacknife: Jacknife is a resampling method
    # 1. Remove one bin at a time
    # 2. Recompute the observable
    # 3. Measure how much it fluctuates
    #a.  Works for nonlinear estimators
    #b. Handles correlations better
    #c. Automatically includes correct normalization
    #d. Is robust when $\langle O \rangle$ is small 

    
    

    # observables
    avg_spinZZ_total = avg_spin_total[0]
    avg_spinXY_total = avg_spin_total[1]
    avg_spinT_total  = avg_spin_total[2]
    #
    avg_den_total  = avg_spin_total[3]
    avg_pair_total  = avg_spin_total[4]

    avg_spinZZ_total_std = std_spin_total[0]
    avg_spinXY_total_std = std_spin_total[1]
    avg_spinT_total_std  = std_spin_total[2]
    #
    avg_den_total_std  = std_spin_total[3]
    avg_pair_total_std  = std_spin_total[4]

    observable_eq_array = [
        avg_spinZZ_total,
        avg_spinXY_total,
        avg_spinT_total,
        avg_den_total,
        avg_pair_total
    ]

    observable_eq_std_array = [
        avg_spinZZ_total_std,
        avg_spinXY_total_std,
        avg_spinT_total_std,
        avg_den_total_std,
        avg_pair_total_std 
    ]

    #
    #print("spin_total_array shape =", spin_total_array.shape)
    #print("avg_spinZZ_total shape =", avg_spinZZ_total.shape)
    #print("avg_spinXY_total shape =", avg_spinXY_total.shape)
    #print("avg_spinT_total  shape =", avg_spinT_total.shape)
    # ====== End spin-spin correlations part ======






     #spin_total_array = np.zeros((3, Nbin_meas, self.Lx, self.Ly), dtype=np.float64)
     #for obs in range(3):
     #    for bin_idx in range(Nbin_meas):
      #       spin_total_array[obs, bin_idx, :, :] = obs_avg_eq_list[obs, bin_idx, :, :]          
    # Mean over bins (axis=1)
     #avg_spin_total = np.mean(spin_total_array, axis=1)  # shape (3, Lx, Ly)
    # Std over bins
     #std_spin_total = np.std(spin_total_array, axis=1)   # shape (3, Lx, Ly)

     #avg_spinZZ_total = avg_spin_total[0,:,:]
     #avg_spinXY_total = avg_spin_total[1,:,:]
     #avg_spinT_total =  avg_spin_total[2,:,:]
    
     #avg_spinZZ_total_std = std_spin_total [0,:,:]
     #avg_spinXY_total_std = std_spin_total [1,:,:]
     #avg_spinT_total_std =  std_spin_total [2,:,:]
     #observable_eq_array  = [avg_spinZZ_total,  avg_spinXY_total,  avg_spinT_total]
     #observable_eq_std_array  = [avg_spinZZ_total_std, avg_spinXY_total_std,  avg_spinT_total_std]
     #print('avg_spinZZ_total[0] =',avg_spinZZ_total)
     #print('avg_spinXY_total[0] =',avg_spinXY_total)
     #print('avg_spinT_total[0] =',avg_spinT_total)
    # end spin-spin correlations part
    #
    #
    #
    # 
    #
    #
    #
    # 
    return observable_array, observable_std_array, observable_eq_array, observable_eq_std_array, abs_phase, abs_phase_std,  abs_Weight,  abs_Weight_std

    

def jackknife_ratio(num, den):
    """
    num, den shape: (Nbin, ...)
    returns: mean, error
    """
    N = num.shape[0]
    jk = np.empty((N,) + num.shape[1:])

    for i in range(N):
        num_i = np.mean(np.delete(num, i, axis=0), axis=0)
        den_i = np.mean(np.delete(den, i, axis=0), axis=0)
        jk[i] = num_i / den_i

    jk_mean = np.mean(jk, axis=0)
    jk_err = np.sqrt((N - 1) * np.mean((jk - jk_mean)**2, axis=0))
    return jk_mean, jk_err




def autocorr(x, max_lag=None):
    """
    Normalized autocorrelation function.
    """
    x = np.asarray(x, dtype=np.float64)
    x -= np.mean(x)
    n = len(x)

    if max_lag is None:
        max_lag = n // 2

    var = np.var(x)
    if var == 0:
        return np.ones(max_lag)

    acf = np.empty(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.dot(x[:n-lag], x[lag:]) / (n-lag) / var
    return acf


def integrated_autocorr_time(x, max_lag=None, cutoff=5):
    """
    Self-consistent windowing estimator for tau_int.
    """
    acf = autocorr(x, max_lag)
    tau_int = 0.5

    for t in range(1, len(acf)):
        if acf[t] < 0:
            break
        tau_int += acf[t]
        if t > cutoff * tau_int:
            break
    return tau_int

# calling
#series = obs_time_series[:, rx, ry]
#tau = integrated_autocorr_time(series)
#print("tau_int =", tau)

