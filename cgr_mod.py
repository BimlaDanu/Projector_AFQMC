import numpy as np
import scipy.linalg as la
import copy
#from wrap_mod import Wrap
from wrap_mod import Wrap, Strat_once_do
#import scipy.linalg
#scipy.linalg.solve
#import math
#import os
#import torch
#import torch.linalg as tla
#import time

def CGRP(self, FL, FR):
    Ndim = FL['U'].shape[0]
    id = np.identity(Ndim, dtype = np.float32)
    URdag = FR['U'].T.conj()
    tmp = URdag @ FL['U']
    G = id - FL['U'] @ la.solve(tmp, URdag)
    sign_det_DR = np.prod(np.sign(np.diag(FR['D'])))
    sign_det_tmp = np.sign(la.det(tmp))
    sign_det_DL = np.prod(np.sign(np.diag(FL['D'])))
    sign_det_BRBL = sign_det_DR * sign_det_tmp * sign_det_DL
    Weight =  np.prod(np.diag(FR['D'])) * la.det(tmp) * np.prod(np.diag(FL['D']))
    return G, sign_det_BRBL,  Weight

def Stackr(self, Bk, hv, spin, nstm, Ltrot, stab_up, P):
    stack = [0. for _ in range(nstm)]
    F = {'U': 1, 'D': 1, 'V': 1}
    nf = int((1 - spin) / 2)
    j = 0
    l_start = Ltrot
    for l in range(Ltrot - 1, -1, -1):
        if stab_up[l]:
            l_end = l_start - 1
            l_start = l
            # initial right factor
            if l_end == Ltrot - 1:
                Bdag = P[:, :, nf]
            else:
                Bdag = np.eye(P.shape[0])
            # accumulate backward over the block
            for ll in range(l_end, l - 1, -1):
                Bv = np.diag(np.exp(spin * hv[:, ll]))
                Bdag = Bk[ll, :, :, nf] @ (Bv @ Bdag)
            # wrap and store
            F = Wrap(self, Bdag, F)
            stack[j] = F.copy()
            j += 1
    return stack


def Stackr1(self, Bk, hv, spin, nstm, Ltrot, stab_up, P):
    # Initialize stack
    stack = [0. for _ in range(nstm)]
    # Placeholder F; shapes will be handled dynamically in Wrap
    F = {'U': 1, 'D': 1, 'V': 1}
    l = 0       # time slice counter
    j = 0       # stack counter
    nf = int((1 - spin) / 2)
    while l < Ltrot:
        # Initial Bdag for current slice
        Bv = np.diag(np.exp(spin * hv[:, -1 - l]))
        Bdag = Bk[-1 - l, :, :, nf] @ (Bv @ P[:, :, nf]) if l == 0 else Bk[-1 - l, :, :, nf] @ Bv
        # Accumulate Bdag until next stabilization point
        while not stab_up[l] and l < Ltrot - 1:
            l += 1
            Bv = np.diag(np.exp(spin * hv[:, -1 - l]))
            Bdag = Bk[-1 - l, :, :, nf] @ (Bv @ Bdag)

        # Perform QR decomposition & update F
        F = Wrap(self, Bdag, F)
        # Store a copy in the stack
        stack[j] = copy.deepcopy(F)
        j += 1  # increment stack index
        l += 1  # move to next time slice
    # Return only filled stack entries
    return stack[:j]




def Stackr0(self, Bk, hv, spin, nstm, Ltrot, stab_up, P):
    #stack = [0.]*nstm  # Use a list to store the dictionaries
    stack = [0. for _ in range(nstm)]
    #stack = np.empty(nstm, dtype = object)
    F = {'U': 1, 'D': 1, 'V': 1}
    l = 0
    j = 0 
    #spin = 1 - 2 * nf     
    nf = int((1-spin)/2)
    while l < (Ltrot-0):
        Bv = np.diag(np.exp(spin * hv[:, -1 - l])) 
        Bdag = Bk[-1 - l,:,:,nf] @ (Bv @ P[:,:,nf]) if l == 0 else Bk[-1 - l,:,:,nf] @ Bv   
        #Bdag = Bk[Ltrot-1 - l,:,:,nf] @ (Bv @ P[:,:,nf]) if l == 0 else Bk[Ltrot-1 - l,:,:,nf] @ Bv 
        
        #Bv = np.diag(np.exp(spin * hv[:, l])) 
        #Bdag = Bk[l,:,:,nf] @ (Bv @ P[:,:,nf]) if l == 0 else Bk[l,:,:,nf] @ Bv   
 
        #print('1,l,stab_up[l] =',l,stab_up[l])
        while not stab_up[l] and l < (Ltrot - 1):
            #print('2,l,stab_up[l] =',l,stab_up[l])
            l += 1
            Bv = np.diag(np.exp(spin * hv[:, -1 - l]))
            Bdag = Bk[-1 - l,:,:,nf] @ (Bv @ Bdag)
            #Bdag = Bk[Ltrot-1 - l,:,:,nf] @ (Bv @ Bdag)
            
            #Bv = np.diag(np.exp(spin * hv[:, l]))
            #Bdag = Bk[l,:,:,nf] @ (Bv @ Bdag)
        F = Wrap(self, Bdag, F)
        stack[j] = F.copy()
        j += 1 # update for stack
        l += 1 # update for time slice
    return stack

def GR_init(self, Bk, hv, nstm, Ltrot, stab_up, P, FP):
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    GR = np.zeros([Ndim, Ndim, N_FL], dtype=np.float32)
    phase_array = np.zeros(N_FL, dtype=np.float32)
    Weight_array = np.zeros(N_FL, dtype=np.float32)
    UDVst = np.empty([nstm, N_FL], dtype=object)  # same as your convention
    for nf in range(2):  # spin index
        spin = 1 - 2 * nf  # 1 for up, -1 for down
        stack_nf = Stackr(self, Bk, hv, spin, nstm, Ltrot, stab_up, P)
        # Fill UDVst with stack entries (up to actual stack length)
        for j, F in enumerate(stack_nf):
            UDVst[j, nf] = F
            #print('j,nstm, UDVst[-1, nf]  =',  j,nstm, UDVst[-1, nf] )
        # Use the **last stabilized F** in the stack for Green's function
        GR[:, :, nf], phase_array[nf], Weight_array[nf] = CGRP(self, FP, stack_nf[-1])
    # Total phase and weight
    phase = np.prod(phase_array)
    Weight = np.prod(Weight_array)
    return GR, phase, UDVst, Weight


def GR_init0(self, Bk, hv, nstm, Ltrot, stab_up, P, FP):
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    GR = np.zeros([Ndim, Ndim, N_FL])
    phase_array = np.zeros(N_FL)
    Weight_array = np.zeros(N_FL)
    UDVst = np.empty([nstm, N_FL], dtype = object)
    for nf in range(2):
        spin = 1 - 2 * nf # 1 for up, -1 for down
        UDVst[:, nf] = Stackr(self, Bk, hv, spin, nstm, Ltrot, stab_up, P)
        #print('nstm, UDVst[:, nf],nstm, UDVst[-1, nf]  =',  nstm, UDVst[:, nf],nstm, UDVst[-1, nf] )
        GR[:, :, nf], phase_array[nf], Weight_array[nf] = CGRP(self, FP, UDVst[-1, nf])
    phase = np.prod(phase_array)
    UDVst = UDVst[-2::-1, :]   #Exclude the last one and flip the order of the rest
    Weight = np.prod(Weight_array)
    return GR, phase, UDVst, Weight 

def GR_fun(self, UDVr, UDVl):
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    phase_array = np.zeros(N_FL, dtype = np.float32)
    Weight_array = np.zeros(N_FL, dtype = np.float32)
    GR = np.zeros((Ndim, Ndim, N_FL), dtype = np.float32)
    for nf in range(N_FL):
        #print('GUDVr[nf] =', UDVr[nf])
        #print('GUDVl[nf] =', UDVl[nf])
        GR_nf, phase_nf, Weight_nf = CGRP(self, UDVr[nf], UDVl[nf])
        GR[:, :, nf] = GR_nf
        phase_array[nf] = phase_nf
        Weight_array[nf] =  Weight_nf
    phase = np.prod(phase_array)
    Weight = np.prod(Weight_array)
    return GR, phase, Weight


















def GR_fun1(self,  torch_UDVr, torch_UDVl):
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    torch_phase_array = np.zeros(N_FL, dtype = np.float32)
    torch_GR = np.zeros([Ndim, Ndim, N_FL], dtype = np.float32)
    for nf in range(N_FL):
        torch_GR[:, :, nf], torch_phase_array[nf] = CGRP(self, torch_UDVr[nf], torch_UDVl[nf])
    torch_phase = np.prod(torch_phase_array)
    return torch_GR, torch_phase


def UDV_init(self, torch_Bk, torch_hv_mat, nstm, ltrot, stab_do, torch_P):
    N_FL = self.N_FL
    # Initialization
    torch_UDVst = np.empty([nstm - 1,  N_FL], dtype = object)
    torch_UDVl = np.empty(N_FL, dtype = object)
    torch_id = np.eye(torch_P.shape[1], dtype = np.float32)
    torch_id_ndim = np.eye(torch_P.shape[0], dtype = np.float32)
    torch_UDV1 = {'U': torch_id, 'D': np.diag(torch_id), 'V': torch_id}
    for nf in range(    N_FL):
        torch_UDVl[nf] = torch_UDV1.copy()
    # Counter for stack
    k = nstm - 2
    # Initialization
    l_start = ltrot
    
    for l in range(ltrot - 1, -1, -1):
        if stab_do[l]:
            l_end = l_start - 1
            l_start = l
            for nf in range(N_FL):
                spin = 1 - 2 * nf 
            
                if k == nstm - 2:
                    Bdag0 = torch_P[:,:,nf] 
                else:
                    Bdag0 = torch_id_ndim
                Bdag = Strat_once_do(self, torch_Bk, torch_hv_mat[:, l_start:(l_end + 1)],  spin, Bdag0)
                #Bdag = Strat_once_do(self, torch_Bk, torch_hv_mat,  spin, Bdag0)
                # Iteratively perform Q D T decomposition and fill stack
                #torch_UDVl[nf] = Wrap(self, Bdag, torch_UDVl[nf].copy()) # torch_UDVl[nf] changed by mutation
                wrap0(self, Bdag, torch_UDVl[nf]) 
                # Fill stack
                if k >= 0:
                    torch_UDVst[k, nf] = torch_UDVl[nf].copy()
            # Update counter for stack
            k -= 1
    return torch_UDVst, torch_UDVl




def wrap0(self, B, UDV):
    B = (B @ UDV['U']) @ np.diag(UDV['D'])
    # QR decomposition
    UDV['U'], R = la.qr(B, mode='economic')
    diag_R = np.diag(R)
    # Avoid division by zero
    eps = 1e-14
    D_abs = np.abs(diag_R) + eps
    # Update V
    UDV['V'] = (np.diag(1.0 / diag_R) @ R) @ UDV['V']
    # Store |D|
    UDV['D'] = D_abs.astype(diag_R.dtype)
    # Absorb phase into U
    UDV['U'] = UDV['U'] @ np.diag(diag_R / D_abs)
    return UDV


def cGRP0(self, UDVr, UDVl):

    Ur = UDVr['U']
    Ul = UDVl['U']
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    Id = np.eye(Ndim, dtype=np.float32)
    ULdag = Ul.T.conj()
    ULdag_UR = ULdag @ Ur
    # stabilization
    ULdag_UR += 1e-14 * np.eye(ULdag_UR.shape[0])
    GR = Id - Ur @ la.solve(ULdag_UR, ULdag)
    det_ULdag_UR = la.det(ULdag_UR)
    phase = det_ULdag_UR / np.abs(det_ULdag_UR)
    return GR, phase


def cGRP00(self, torch_UDVr, torch_UDVl):
    # G = id - BR (BL BR)^-1 BL
    #   = id - QR (QL_dag QR)^-1 QL_dag
    #
    # Note: QL_dag and QR is rectangular matrix
    #       UDVr['D']) and UDVl['D']) is positive real number
    #ndim = torch_UDVr['U'].shape[0]
    N_FL = self.N_FL
    Ndim = self.Lx * self.Ly * self.Norbs * self.Nlayers
    torch_id = np.eye(Ndim, dtype =  np.float32)
    torch_ULdag = torch_UDVl['U'].T.conj()
    torch_ULdag_UR = torch_ULdag @ torch_UDVr['U']
    torch_GR = torch_id - torch_UDVr['U'] @ la.solve(torch_ULdag_UR, torch_ULdag)
    torch_det_ULdag_UR = la.det(torch_ULdag_UR)
    torch_phase = torch_det_ULdag_UR / np.abs(torch_det_ULdag_UR)
    return torch_GR, torch_phase

def wrap00(self, B, UDV):
    B = (B @ UDV['U']) @ np.diag(UDV['D'])
    # Decompose B as B = Q R, where column of Q is orthonormal basis and R is upper triangular matrix
    #UDV['U'], R = la.qr(B, mode = 'reduced')
    UDV['U'], R = la.qr(B, mode = 'economic')
    # Split R into D (D^-1 R) where D is diagonal part of R
    # Now B = U D V = (U phase(D)) |D| V, where V = D^-1 R is an upper triangular matrix with diagonal part of V is 1, and |D| is positive real number
    diag_R = np.diag(R) # D in array
    UDV['V'] = (np.diag(1 / diag_R) @ R) @ UDV['V']
    UDV['D'] = np.abs(diag_R).type(diag_R.dtype) # |D| in array
    UDV['U'] = UDV['U'] @ np.diag(diag_R / UDV['D']) # U phase(D)
    return UDV

def Update_stack_do0(self, Bk, hv, UDVst, len1, P, UDVl, FP_list, nst, nstm):
    N_FL = self.N_FL
    hv = hv[:, :len1]
    #B0 = P if nst == nstm - 2 else 1
    # Update UDVl
    for nf in range(N_FL):
        spin = 1 - 2 * nf  # 1 for up, -1 for down
        B0 = P[:,:,nf] if nst == nstm - 2 else 1
        Bdag = Strat_once_do(self, Bk, hv, spin, B0)
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



#def Strat_once_do(self, Bk, hv, spin, Bdag):
#    Ltrot = hv.shape[1]
#    nf = int((1-spin)/2)
#    for l in range(Ltrot-1, -1, -1):
#        Bv = np.diag(np.exp(spin * hv[:,  l]))
#        Bdag = Bk[l,:,:,nf] @ np.dot(Bv, Bdag)
#    return Bdag

