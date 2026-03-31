import numpy as np
import matplotlib.pyplot as plt
#from main import Main
#from main_mpi_pqmc import Main
from main import Main
#from main_mpi import Main
from ana import Ana
import os
from hopping_ham_mod import square_lattice
from save_correlations import *
from save_correlations_bc import *
from  spin_spin_corr_all import *
#from main import Main
#from ana import Ana


#%matplotlib inline

def save_data(self, filename, param_x, O_y):
        with open(filename, 'w') as f:
            f.write("param_x  <O_y>\n")
            for i in range(len(param_x)):
                #f.write(f"{param_x} {O_y}\n")
                f.write(f"{param_x[i]:0.1f}  {O_y[i]:0.8f}\n") 
     
        
def check_for_nan_inf(self, matrix, name="matrix"):
        if np.isnan(matrix).any() or np.isinf(matrix).any():
                raise ValueError(f"Input matrix {name} contains NaN or Inf values.")

 
def main_run(self):
        Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
        N_part = self.N_part
       # theta_array = np.array([0.05, 0.1, 0.5, 1, 2, 4, 10, 15, 20, 25, 30])
        #theta_array = np.array([0.05, 0.1, 0.5, 1, 2, 4, 10, 20, 25, 30])
        #theta_array = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 25, 30])
        # 
        theta_array = np.array([1.]) 
        #theta_array = np.array([0.1, 0.5, 1, 5, 10, 20])
        #theta_array = np.array([0.1], dtype = anp.float32)
        tun_params = True
        #print('theta_array.shape, theta_array.shape[0]  =', theta_array.shape, theta_array.shape[0])
        Ndim_theta = np.int32(theta_array.shape)

        energy_total_array = np.empty(Ndim_theta, dtype = np.float32)
        energy_total_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Pot_eng_array = np.empty(Ndim_theta, dtype = np.float32)
        Pot_eng_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Kin_eng_array = np.empty(Ndim_theta, dtype = np.float32)
        Kin_eng_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Eng_total_array = np.empty(Ndim_theta, dtype = np.float32)
        Eng_total_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Sz_total_array = np.empty(Ndim_theta, dtype = np.float32)
        Sz_total_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Rho_total_array = np.empty(Ndim_theta, dtype = np.float32)
        Rho_total_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
    
        energy_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        energy_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
    
        Pot_eng_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        Pot_eng_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Kin_eng_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        Kin_eng_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Eng_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        Eng_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Sz_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        Sz_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        Rho_persite_array = np.empty(Ndim_theta, dtype = np.float32)
        Rho_persite_std_array = np.empty(Ndim_theta, dtype = np.float32)
        
        
    
        abs_phase_array = np.empty(Ndim_theta, dtype = np.float32)
        abs_phase_std_array = np.empty(Ndim_theta, dtype = np.float32)
        elapsed_array = np.empty(Ndim_theta, dtype = np.float32)
        
        abs_Weight_array = np.empty(Ndim_theta, dtype = np.float32)
        abs_Weight_std_array = np.empty(Ndim_theta, dtype = np.float32)
    

        deltaG_max_array = np.empty(Ndim_theta, dtype = np.float32)
        deltaG_mean_array = np.empty(Ndim_theta, dtype = np.float32)
        deltaS_max_array = np.empty(Ndim_theta, dtype = np.float32)
        deltaS_mean_array = np.empty(Ndim_theta, dtype = np.float32)


        #Use NumPy for arrays with dtype=object
        #phase_avg_array_list = [0.] *int(theta_array.shape[0])
        #observable_array_list = [0.] * int(theta_array.shape[0])
        #observable_std_array_list = [0.] * int(theta_array.shape[0])
        
        txtstr_np = (f'square_{self.Lx}x{self.Ly}'f'nwrap{self.Nwrap}_nsweep{self.Nsweep}_np{self.N_part}_U{self.Ham_U}')

       # print('params, self.objec_fun_grad(params), grad(self.objec_fun_grad) =', params, self.objec_fun_grad(params), grad(self.objec_fun_grad))
        #theta_array = np.array([0.1, 0.5, 1, 5, 10])
        for j, This in enumerate(theta_array):
            print(f'\nThis = {This}')
            
            #while True:
            #    print(f'\nnsweep = {self.Nsweep}')
            # QMC algorithm
            #K, Nbin_eff, obs_avg_scal_list, obs_avg_eq_list, obs_avg_geq_list, phase_avg_array, Weight_avg_array, deltaG_max, deltaG_mean, deltaS_max, deltaS_mean, time_elapsed = Main(self, This, tun_params)
            K, Nbin_eff, obs_avg_scal_list, obs_avg_eq_list, obs_avg_geq_list, phase_avg_array, Weight_avg_array, deltaG_max, deltaG_mean, deltaS_max, deltaS_mean, time_elapsed, acceptance_rate = Main(self, This, tun_params)
            #print('This,obs_avg_scal_list, grad(self.obj_E_loc_theta) =',This,obs_avg_scal_list, grad(obs_avg_scal_list))
            #
            print(f'Elapsed = {time_elapsed} seconds')
            print('acceptance_rate =', acceptance_rate)
            #
            # Analysis
            observable_array, observable_std_array, observable_eq_array, observable_eq_std_array, abs_phase, abs_phase_std, abs_Weight, abs_Weight_std = Ana(self, Nbin_eff, K, obs_avg_scal_list, obs_avg_eq_list, obs_avg_geq_list, phase_avg_array,  Weight_avg_array)#
            print('observable_array, observable_std_array = ',observable_array, observable_std_array)
            print('abs_phase, abs_phase_std = ',abs_phase, abs_phase_std)
            print('abs_Weight, abs_Weight_std = ',abs_Weight, abs_Weight_std)
            
            #observable_array = [energy_total, d, Pot_total, Kin_total, Eng_total]
            #observable_std_array = [energy_total_std, d_std, Pot_std,  Kin_std, Eng_std]
                    
            #observable_array = [energy_total, energy_persite, d, Pot_total, Pot_persite, Kin_total,  Kin_persite, Eng_total, Eng_persite]
            #observable_std_array = [energy_total_std, energy_persite_std, d_std, Pot_std,  Pot_std_persite,  Kin_std,  Kin_std_persite, Eng_std,  Eng_std_persite]
            
            energy_total_array[j] = observable_array[0]
            energy_total_std_array[j] = observable_std_array[0]
            energy_persite_array[j] =  observable_array[0]/ Ndim
            energy_persite_std_array[j] =  observable_std_array[0]/ Ndim
            
            Pot_eng_array[j] = observable_array[2]
            Pot_eng_std_array[j] = observable_std_array[2]
            Pot_eng_persite_array[j] =  observable_array[2]/ Ndim
            Pot_eng_persite_std_array[j] =  observable_std_array[2]/ Ndim
            
            Kin_eng_array[j] = observable_array[3]
            Kin_eng_std_array[j] = observable_std_array[3]
            Kin_eng_persite_array[j] =  observable_array[3]/ Ndim
            Kin_eng_persite_std_array[j] =  observable_std_array[3]/ Ndim
            
            Eng_total_array[j] = observable_array[4]
            Eng_total_std_array[j] = observable_std_array[4]
            Eng_persite_array[j] =  observable_array[4]/ Ndim
            Eng_persite_std_array[j] =  observable_std_array[4]/ Ndim
            
            
            Sz_total_array[j] = observable_array[5]
            Sz_total_std_array[j] = observable_std_array[5]
            Sz_persite_array[j] =  observable_array[5]/ Ndim
            Sz_persite_std_array[j] =  observable_std_array[5]/ Ndim
            
            Rho_total_array[j] = observable_array[6]
            Rho_total_std_array[j] = observable_std_array[6]
            Rho_persite_array[j] =  observable_array[6]/ Ndim
            Rho_persite_std_array[j] =  observable_std_array[6]/ Ndim
            
            
            
            abs_phase_array[j] = abs_phase
            abs_phase_std_array[j] = abs_phase_std
   
            abs_Weight_array[j] = abs_Weight
            abs_Weight_std_array[j] = abs_Weight_std
            
            Weight_avg_array_list = [np.zeros(len(phase_avg_array), dtype=np.float32) for _ in range(theta_array.shape[0])]
            phase_avg_array_list = [np.zeros(len(phase_avg_array), dtype=np.float32) for _ in range(theta_array.shape[0])]
            
            observable_array_list = [np.zeros(len(observable_array), dtype=np.float32) for _ in range(theta_array.shape[0])]
            observable_std_array_list = [np.zeros(len(observable_std_array), dtype=np.float32) for _ in range(theta_array.shape[0])]
        
            Weight_avg_array_list[j] = Weight_avg_array
            phase_avg_array_list[j] = phase_avg_array
            
            observable_array_list[j] = observable_array
            observable_std_array_list[j] = observable_std_array
            
            deltaG_max_array[j] = deltaG_max
            deltaG_mean_array[j] = deltaG_mean
            deltaS_max_array[j] = deltaS_max
            deltaS_mean_array[j] = deltaS_mean

            print('This, Average total energy:', This, observable_array[0], observable_array[0] / Ndim)
            print('This, Std dev total energy:', This, observable_std_array[0], observable_std_array[0] / Ndim)
            
            print('This, Average total eng:', This, observable_array[4], observable_array[4]/ Ndim)
            print('This, Std dev total eng:', This, observable_std_array[4], observable_std_array[4]/ Ndim)
        
            print('This, Average P. E. + K. E.:', This, observable_array[2]+  observable_array[3], (observable_array[2]+  observable_array[3]) / Ndim)
            print('This, Std dev P. E. + K. E. :', This, observable_std_array[2]+  observable_std_array[3], (observable_std_array[2]+  observable_std_array[3])/ Ndim)
                
            print('This, Average P.E :', This, observable_array[2], observable_array[2] / Ndim)
            print('This, Std dev P.E :', This, observable_std_array[2], observable_std_array[2] / Ndim)
            
            print('This, Average  K. E.:', This, observable_array[3],   observable_array[3] / Ndim)
            print('This, Std dev  K. E.:', This,  observable_std_array[3],  observable_std_array[3]/ Ndim)
            
            print('This, Average total phase:', This, phase_avg_array)
            print('This, Average total log weight:', This, Weight_avg_array)
  
            print('This, Average |sign|: ', This, abs_phase)
            print('This, Average log weight: ', This, abs_Weight)

            print('This, Maximum delta G: ', This, deltaG_max)
            print('This, Average delta G: ', This, deltaG_mean)
        
        # Computing spin-spin correlations    
        #save_spin_correlations(self, observable_eq_array, observable_eq_std_array, Ndim)
        
        
        #save_spin_correlations_placeholder(self, observable_eq_array, observable_eq_std_array, Ndim) 
        if not self.Corr_all:
            #save_spin_correlations_bc(self, observable_eq_array, observable_eq_std_array, Ndim, self.bc_x, self.bc_y, 1)
            save_spin_correlations_placeholder(self, observable_eq_array, observable_eq_std_array, Ndim,1) 
            save_density_correlations_placeholder(self, observable_eq_array[3], observable_eq_std_array[3], Ndim, 1) 
            save_pair_correlations_placeholder(self, observable_eq_array[4], observable_eq_std_array[4], Ndim, 1, pairing='s') 
        else:
            #save_spin_correlations_full_r(self, observable_eq_array, observable_eq_std_array, Ndim, self.bc_x, self.bc_y, 1)
            save_spin_correlations_full_r_1(self, observable_eq_array, observable_eq_std_array, Ndim, self.bc_x, self.bc_y, 1)
            save_den_correlations_full_r_1(self, observable_eq_array[3], observable_eq_std_array[3], Ndim, self.bc_x, self.bc_y, 1)
            save_pair_correlations_full_r_1(self, observable_eq_array[3], observable_eq_std_array[4], Ndim, self.bc_x, self.bc_y, 1, pairing='s')
            
            
        
        
        # Computing density-density correlations 
        #save_density_correlations(self, DD_array, DD_std_array, Ndim)   
        # Computing pair-pair correlations 
        #save_pair_correlations(self, P_array, P_std_array, Ndim, pairing='s') 
        #save_pair_correlations(self, P_array, P_std_array, Ndim, pairing='px')  
        #save_pair_correlations(self, P_array, P_std_array, Ndim, pairing='py')    
        #save_pair_correlations(self, P_array, P_std_array, Ndim, pairing='d') 


        #Save info of parameters used in QMC simulations
        Data_folder_name = f"Data_storage"
        os.makedirs(Data_folder_name, exist_ok=True)
        info_filename = (f"info_Ndim{Ndim}_Npart{self.N_part}_U{self.Ham_U}_{txtstr_np}_file.txt")
        info_path = os.path.join(Data_folder_name, info_filename)
        with open(info_path, "w") as f:
        #with open(f"1info_hopt_{Ndim}_{N_part}_file.txt", 'w') as f:
            f.write(f"Theta values: {theta_array}\n"
                    f"Lx: {self.Lx}\n"
                    f"Ly: {self.Ly}\n"
                    f"Lattice dimension: {Ndim}\n"
                    f"Beta: {self.Beta}\n"
                    f"Theta: {self.Theta}\n"
                    f"Nsweep: {self.Nsweep}\n"
                    f"Nbin: {self.Nbin}\n"
                    f"Nwrap: {self.Nwrap}\n"
                    f"Sym: {self.Sym}\n"
                    f"dtau: {self.dtau}\n"
                    f"N_Part: {self.N_part}\n"
                    f"Ham_t: {self.Ham_t}\n"
                    f"Ham_U: {self.Ham_U}\n"
                    f"t_prime: {self.Ham_tp}\n"
                    f"Ham_mu: {self.Ham_mu}\n"
                    f"Model: {self.ham_model}\n"
                    f"N_FL: {self.N_FL}\n"
                    f"N_SUN: {self.N_SUN}\n")

        #Save data from QMC simulations
        QMCdata_filename = (f"QMC_data_Ndim{Ndim}_N_part{self.N_part}_U{self.Ham_U}_{txtstr_np}_file.txt")
        QMCdata_path = os.path.join(Data_folder_name, QMCdata_filename)
        with open(QMCdata_path, 'w') as f:
        #with open(f"1QMC_data_hopt_{Ndim}_{N_part}.txt", 'w') as f:
            f.write(f"energy_total_array: {energy_total_array}\n"
                    f"energy_total_std_array: {energy_total_std_array}\n"
                    f"abs_phase_array: {abs_phase_array}\n"
                    f"abs_phase_std_array: {abs_phase_std_array}\n"
                    f"Weight_avg_array_list: {Weight_avg_array}\n"
                    f"elapsed_array: {elapsed_array}\n"
                    f"phase_avg_array_list: {phase_avg_array_list}\n"
                    f"observable_array_list: {observable_array_list}\n"
                    f"observable_std_array_list: {observable_std_array_list}\n"
                    f"deltaG_max_array: {deltaG_max_array}\n"
                    f"deltaG_mean_array: {deltaG_mean_array}\n"
                    f"deltaS_max_array: {deltaS_max_array}\n"
                    f"deltaS_mean_array: {deltaS_mean_array}\n"
                    #
                    f"Sz_total_array: {Sz_total_array}\n"
                    f"Sz_total_std_array: {Sz_total_std_array}\n"
                    #
                    f"Rho_total_array: {Rho_total_array}\n"
                    f"Rho_total_std_array: {Rho_total_std_array}\n")
                    

        #Plot results
        output_filename = f"Average_Energy_hopt_{txtstr_np}.txt"
        #Assuming save_data is a function defined elsewhere
        save_data(self, output_filename, theta_array, energy_total_array)

        folder_name = "figure_storage_hopt"
        #folder_name = "NewAna_Thetaarray_figure_storage"
        os.makedirs(folder_name, exist_ok=True)
        
        #plt.plot(theta_array, energy_total_array, marker='s')
        plt.errorbar(theta_array, energy_total_array, yerr=energy_total_std_array, marker='s')
        plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.ylabel(r'Total energy$~(t)$', fontsize=18)
        plt.savefig(os.path.join(folder_name,f'Energy_{txtstr_np}.pdf'), bbox_inches='tight')
       # plt.savefig(os.path.join(folder_name, f'Energy_{Ndim}.pdf'),
       #     bbox_inches='tight')
        #plt.savefig(txtstr_np + '.pdf', bbox_inches='tight')
        #plt.show()
        plt.clf()
        
        plt.errorbar(theta_array,  abs_phase_array, yerr=abs_phase_std_array, label = '<sign QMC>',  marker='s')
        plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.ylabel('<sign>', fontsize=18)
        #plt.legend(loc = 'best' )
        plt.savefig(os.path.join(folder_name, f'Sign_{txtstr_np}.pdf'), bbox_inches='tight')
        #plt.show()
        plt.clf()
 
        #abs_Weight_array,   abs_Weight_std_array
        #plt.errorbar(theta_array, np.log(Weight_array), yerr=abs(np.log(Weight_std_array)), label = '<ln W QMC>',  marker='s')
        plt.errorbar(theta_array,  abs_Weight_array, yerr=abs_Weight_std_array, label = '<ln W QMC>',  marker='s')
        #plt.errorbar(theta_array,  ln_Weight_field_total_array, yerr=ln_Weight_field_total_std_array, label = '<ln W QMC field>',  marker='o')
        #plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.ylabel('ln(W(t))', fontsize=18)
        #plt.legend(loc = 'best' )
        plt.savefig(os.path.join(folder_name, f'Weight_{txtstr_np}.pdf'), bbox_inches='tight')
        #plt.show()
        plt.clf()
        
        plt.errorbar(theta_array, Sz_total_array, yerr=Sz_total_std_array, marker='s')
        plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.ylabel(r'$m =<S_z>$', fontsize=18)
        plt.savefig(os.path.join(folder_name,f'Sz_ {txtstr_np}.pdf'), bbox_inches='tight')
        #plt.show()
        plt.clf()
        
        plt.errorbar(theta_array, Rho_total_array, yerr=Rho_total_std_array, marker='s')
        plt.xlabel(r'$\Theta~(1/t)$', fontsize=18)
        plt.ylabel(r'\rho= <n>$', fontsize=18)
        plt.savefig(os.path.join(folder_name,f'Rho_{txtstr_np}.pdf'), bbox_inches='tight')
        #plt.show()
        plt.clf()

