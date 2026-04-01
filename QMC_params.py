import numpy as np
#from  Params_QMC_Sim_regular import *
class ModelParams():
    def __init__(self):
        # For regular Lattices
        self.ham_model: str = 'Hubbard'  
        self.lattice_type: str = 'Square'
        #self.lattice_type: str = 'Square'
        #self.lattice_type: str = 'Triangular'
        #self.lattice_type: str = 'Honeycomb'
        #self.lattice_type: str = 'Kagome'
        #self.lattice_type: str = '1d_chain'
        
        
        #For Bilayer Lattices
        #self.ham_model: str = 'Periodic_Anderson' 
        #self.lattice_type: str = 'Bilayer_Square'
        #self.lattice_type: str = 'Bilayer_Triangular'
        #self.lattice_type: str = 'Bilayer_Honeycomb'
        #self.lattice_type: str = 'Bilayer_Kagome'
        
        self.N_SUN: int = 1
        self.N_FL: int = 2
        self.Per: bool = True #False
        self.Corr_all: bool = True 
        #self.Corr_all: bool = False
        
        self.Hop_tau: bool = False
        #self.Hint_tau: bool = False 
        
        #self.Hop_tau: bool = True
        self.Hint_tau: bool = True 
        #
        #
        #
        self.ladder: bool = True # False
        self.bc_x: str = 'open' # Periodic
        self.bc_y: str = 'open' # Periodic
        #
        #
        #
        #self.ladder: bool = True # False
        #self.bc_x: str = 'open' # Periodic
        #self.bc_y: str = 'periodic' # Periodic
        #
        #
        #
        #self.ladder: bool = False # False
        #self.ladder: bool = True
        #self.bc_x: str = 'periodic' # open
        #self.bc_y: str = 'periodic' # open
model_obj = ModelParams()

class LattParams():
    def __init__(self, model_obj: ModelParams):
        self.Lx: int  = 16  # length along a_1 direction
        self.Ly: int = 2 # length along a_2 direction
        self.BC_Lx: bool = True  # boundary condition along a_1; False: open, True: periodic
        self.BC_Ly: bool = True  # boundary condition along a_2; False: open, True: periodic
        if model_obj.ham_model == 'Hubbard':
            #self.Norbs: int = 1
            self.Nlayers: int = 1
        elif model_obj.ham_model == 'Periodic_Anderson':
            #self.Norbs: int = 1
            self.Nlayers: int = 2
         
        #For regular lattices   
        if  model_obj.lattice_type == 'Square' or   model_obj.lattice_type == 'Triangular' or   model_obj.lattice_type == '1d_chain':
            self.Norbs: int = 1
        elif model_obj.lattice_type == 'Honeycomb':
            self.Norbs: int = 2
        elif model_obj.lattice_type == 'Kagome':
            self.Norbs: int = 3
            
        
        #For Bilayer lattices  
        if  model_obj.lattice_type == 'Bilayer_Square' or   model_obj.lattice_type == 'Bilayer_Triangular':
            self.Norbs: int = 1
        elif model_obj.lattice_type == 'Bilayer_Honeycomb':
            self.Norbs: int = 2
        elif model_obj. lattice_type == 'Bilayer_Kagome':
            self.Norbs: int = 3
            
latt_obj = LattParams(model_obj)

class HamParams():
    def __init__(self, model_obj: ModelParams):
        if model_obj.ham_model == 'Hubbard':
            self.Ham_t: float = 1.0  # NN hopping along x axis
            self.Ham_tp: float = 0.  # NN hopping along x axis
            self.Ham_U: float = 8.0
            self.Ham_mu: float = 0.0
            self.N_part: int = -1 #-1 #7#-1  # Number of particles per flavor (At half-filling: n_part = Ndim / 2)
        elif model_obj.ham_model == 'Periodic_Anderson':
            self.Ham_t: float = 1.0  # NN hopping along x axis
            self.Ham_tp: float = 0.0  # NN hopping along x axis
            self.Ham_V: float = 0.0
            self.Ham_U: float = 8.0 # Only on f electrons
            self.Ham_muc: float = 0.0
            self.Ham_muf: float = 0.0
            self.N_part: int = -1
ham_obj = HamParams(model_obj)

class QMCParams():
    def __init__(self):
        self.Nbin: int = 24  # number of bins
        self.Nsweep: int = 24 # number of QMC sweeps per bin
        self.dtau: float = 0.1  # Imaginary time steps
        self.Nwrap: int = 6  # Green's function will be computed from scratch after each time interval nwrap * dtau
        self.Theta: float = 0.5
        self.Beta: float = 0.5  # Inverse temperature
        self.Sym: bool = False #True  # Suzuki-Trotter symmetrization flag
        #self.Adiabatic: bool = False 
        self.Adiabatic: bool = True
        self.Projector: bool = True
        self.Ltau: int = 0  # 1 for time displaced correlations
        self.verbose: bool = False  # Flag for verbosity
        self.CPU_MAX: float = 0.0  # Maximum simulation time in hours
        self. LOBS_ST: int = 0 #1 Start measurements at time slice LOBS_ST
        self.LOBS_EN:  int = 0 #100 End measurements at time slice LOBS_EN
        self.hirsch: bool = True
        self.Ltau_v: int = 5  # 1 for time displaced correlations
        self.slice_m: int = 0 # 1 for time displaced correlations
        self.restart_enabled = False 
QMC_obj = QMCParams()

class AnaParams():
    def __init__(self):
        self.N_skip: int = 2  # Number of bins for warm-up / burn-in
        self.N_Cov: int = 0  # Flag of covariance matrix
        self.N_rebin = 1
Ana_obj = AnaParams()

class QMC_Run_Params():
    def __init__(self, model_obj, latt_obj, ham_obj, QMC_obj, Ana_obj):
        # Model parameters
        self.ham_model = str(model_obj.ham_model)
        self.lattice_type = str(model_obj.lattice_type)
        self.N_SUN = int(model_obj.N_SUN)
        self.N_FL = int(model_obj.N_FL)
        self.Per = bool(model_obj.Per)
        self.Corr_all = bool(model_obj.Corr_all)
        self.Hop_tau = bool(model_obj.Hop_tau)
        self.Hint_tau = bool(model_obj.Hint_tau)
        self.bc_x = str(model_obj.bc_x)
        self.bc_y = str(model_obj.bc_y)
        self.ladder = bool(model_obj.ladder)
        
        
        # Hamiltonian parameters
        if model_obj.ham_model == 'Hubbard':
            self.Ham_t = float(ham_obj.Ham_t)
            self.Ham_tp = float(ham_obj.Ham_tp)
            self.Ham_mu = float(ham_obj.Ham_mu)
            self.Ham_U = float(ham_obj.Ham_U)
            self.N_part = int(ham_obj.N_part)
            
        elif model_obj.ham_model == 'Periodic_Anderson':  
            self.Ham_t = float(ham_obj.Ham_t)
            self.Ham_tp = float(ham_obj.Ham_tp)
            self.Ham_V = float(ham_obj.Ham_V) 
            self.Ham_Uf = float(ham_obj.Ham_Uf) 
            self.Ham_muc = float(ham_obj.Ham_muc) 
            self.Ham_muf = float(ham_obj.Ham_muf)  
            self.N_part = int(ham_obj.N_part)
            
        
        # Lattice parameters
        self.Lx = int(latt_obj.Lx)
        self.Ly = int(latt_obj.Ly)
        self.BC_Lx = int(latt_obj.BC_Lx)
        self.BC_Ly = int(latt_obj.BC_Ly)
        self.Norbs = int(latt_obj.Norbs)
        self.Nlayers = int(latt_obj.Nlayers)
        
        # QMC parameters
        self.Nbin = int(QMC_obj.Nbin)  # number of bins
        self.Nsweep = int(QMC_obj.Nsweep)  # number of QMC sweeps per bin
        self.dtau = float(QMC_obj.dtau)  # Imaginary time steps
        self.Nwrap = int(QMC_obj.Nwrap)  # Green's function will be computed from scratch after each time interval nwrap * dtau
        self.Theta = float(QMC_obj.Theta)
        self.Beta = float(QMC_obj.Beta)  # Inverse temperature
        self.Sym = bool(QMC_obj.Sym)  # Suzuki-Trotter symmetrization flag
        self.Adiabatic = bool(QMC_obj.Adiabatic)
        self.Projector = bool(QMC_obj.Projector)
        self.Ltau = int(QMC_obj.Ltau)  # 1 for time-displaced correlations
        self.verbose = bool(QMC_obj.verbose)  # Flag for verbosity
        self.CPU_MAX = float(QMC_obj.CPU_MAX)  # Maximum simulation time in hours
        self.LOBS_ST = int(QMC_obj.LOBS_ST)  # Start measurements at time slice LOBS_ST
        self.LOBS_EN = int(QMC_obj.LOBS_EN)  # End measurements at time slice
        self.hirsch = bool(QMC_obj.hirsch)
        self.Ltau_v = int(QMC_obj.Ltau_v) 
        self.slice_m = int(QMC_obj.slice_m) 
        self.restart_enabled = bool(QMC_obj.restart_enabled)
        
        
        # Analysis parameters
        self.N_skip = int(Ana_obj.N_skip)  # Number of bins for warm-up / burn-in
        self.N_Cov = int(Ana_obj.N_Cov)  # Flag of covariance matrix
        self.N_rebin = int(Ana_obj.N_rebin)
    
