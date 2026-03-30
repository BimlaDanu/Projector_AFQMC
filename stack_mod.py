import numpy as np
#from wrap_mod import Strat_once_up, Strat_once_do, Wrap
from wrap_mod import Strat_once_up, Strat_once_do, Wrap, Strat_once_up0, Strat_once_do0
def Update_stack_up(self, Bk, hv, UDVst, len1, P, UDVr, FP_list, nst, nstm, l_start, l_end):
        N_FL = self.N_FL        
        # Update UDVr
        for nf in range(N_FL):
            spin = 1 - 2 * nf  # 1 for up, -1 for down
            B0 = P[:,:,nf] if nst == 0 else 1
            B = Strat_once_up(self, Bk, hv, spin, B0, l_start)
            #B = Strat_once_up0(self, Bk, hv, spin, B0, l_start)
            UDVr[nf] = Wrap(self, B, UDVr[nf].copy())
        # Get UDVl
        if nst == nstm - 1:
            UDVl = FP_list
            nst -= 1
        else:
            UDVl = UDVst[nst, :].copy()
            UDVst[nst, :] = UDVr
            #UDVl = [udv.copy() for udv in UDVst[nst]]
            #UDVst[nst] = [udv for udv in UDVr]
            #print('UDVst[nst]= ',UDVst[nst])
            nst += 1
        return UDVst, UDVr, UDVl, nst

def Update_stack_do(self, Bk, hv, UDVst, len1, P, UDVl, FP_list, nst, nstm, l_start, l_end):
    N_FL = self.N_FL    
    # Update UDVl
    for nf in range(N_FL):
        spin = 1 - 2 * nf  # 1 for up, -1 for down
        B0 = P[:,:,nf] if nst == nstm - 2 else 1
        Bdag = Strat_once_do(self, Bk, hv, spin, B0, l_start)
        #Bdag = Strat_once_do0(self, Bk, hv, spin, B0, l_start)
        UDVl[nf] = Wrap(self, Bdag, UDVl[nf].copy())
    # Get UDVr
    if nst == -1:
        UDVr = FP_list
        nst += 1
    else:
        UDVr = UDVst[nst, :].copy()
        UDVst[nst, :] = UDVl
        #UDVr = [udv.copy() for udv in UDVst[nst]]
        #UDVst[nst] = [udv for udv in UDVl]  
        nst -= 1
    return UDVst, UDVr, UDVl, nst
