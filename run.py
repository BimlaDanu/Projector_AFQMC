import numpy as np
import os
from QMC_params import *
from simulation import main_run, save_data#,  check_for_nan_inf

self = QMC_Run_Params(model_obj, latt_obj, ham_obj, QMC_obj, Ana_obj)
print(model_obj.ham_model, latt_obj.Lx, self.Lx, self.Ly, self.Ham_t, self.ham_model, self.N_skip, self.N_Cov, self.N_rebin)#, self.Lx, self.Ly, self.Norbs, self.Nlayers)
results = main_run(self)