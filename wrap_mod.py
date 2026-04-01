import numpy as np
import scipy.linalg as la

def Wrap(self, BL, F):
    B = np.dot(np.dot(BL, F['U']), F['D'])
    if np.isnan(B).any() or np.isinf(B).any():
        raise ValueError("Input matrix contains NaN or Inf values in WRAP.")
    F['U'], R = la.qr(B, mode='economic')
    D_vec = np.diag(R)
    F['D'] = np.diag(D_vec)
    inv_D = np.diag(1 / D_vec)
    F['V'] = inv_D @ np.dot(R, F['V'])
    return F

def Strat_once_up(self, Bk, hv, spin, B, l_start_global):
    Ltrot = hv.shape[1]
    nf = int((1-spin)/2)
    for l in range(Ltrot): 
        Bv = np.diag(np.exp(spin * hv[:, l]))
        B = Bv @ np.dot(Bk[l,:,:,nf], B)
    return B

def Strat_once_do(self, Bk, hv, spin, Bdag, l_start_global):
    Ltrot = hv.shape[1]
    nf = int((1-spin)/2)
    for l in range(Ltrot-1, -1, -1):
        Bv = np.diag(np.exp(spin * hv[:,  l]))
        Bdag = Bk[l,:,:,nf] @ np.dot(Bv, Bdag)
    return Bdag

def Strat_once_up0(self, Bk, hv, spin, B, l_start_global):
    nf = int((1-spin)/2)
    slice_len = hv.shape[1]
    for i in range(slice_len):
        l_global = l_start_global + i  # map local to global
        Bv = np.diag(np.exp(spin * hv[:, i]))
        B = Bv @ np.dot(Bk[l_global, :, :, nf], B)
    return B

def Strat_once_do0(self, Bk, hv, spin, Bdag, l_start_global):
    nf = int((1-spin)/2)
    slice_len = hv.shape[1]
    for i in range(slice_len-1, -1, -1):
        l_global = l_start_global + i
        Bv = np.diag(np.exp(spin * hv[:, i]))
        Bdag = Bk[l_global, :, :, nf] @ np.dot(Bv, Bdag)
    return Bdag

