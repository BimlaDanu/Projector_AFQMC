import numpy as np
import scipy.linalg as la


def UpdateGR(self, GR, hv, phase, rng, Weight, l, acceptance_rate, count):
    N_FL = self.N_FL
    Ndim = hv.shape[0]
    idm = np.identity(Ndim, dtype=np.float64)

    alpha_array = np.zeros(N_FL, dtype=np.float64)
    w_ratio_array = np.zeros(N_FL, dtype=np.float64)

    for site in range(Ndim):

        # ---- use hv[site, l] consistently ----
        for nf in range(N_FL):
            spin = 1 - 2 * nf
            alpha_array[nf] = np.exp(-2.0 * spin * hv[site, l]) - 1.0
            w_ratio_array[nf] = 1.0 + alpha_array[nf] * (1.0 - GR[site, site, nf])
        w_ratio = np.prod(w_ratio_array)
        count += 1
        # protect division by zero
        if w_ratio == 0:
            continue
        phase_change_propose = w_ratio / np.abs(w_ratio)
        phase_propose = phase * phase_change_propose
        p = np.abs(w_ratio * (1.0 + phase_propose ** -2) / (1.0 + phase ** -2))
        if rng.random() <= p:
            for nf in range(N_FL):
                GRC_T = idm - GR[:, :, nf]
                GR[:, :, nf] = GR[:, :, nf] - (alpha_array[nf] / w_ratio_array[nf]) * (
                    GR[:, [site], nf] @ GRC_T[[site], :]
                )
            hv[site, l] = -hv[site, l]
            phase = phase_propose
            acceptance_rate = ((count - 1) * acceptance_rate + 1.0) / count
        else:
            acceptance_rate = (count - 1) * acceptance_rate / count
    return GR, hv, phase, Weight, acceptance_rate, count



def Wrap_upward(self, GR, Bk, inv_Bk, hv, l):
    N_FL = self.N_FL
    for nf in range(N_FL):
        spin = 1 - 2 * nf
        Bv = np.diag(np.exp(spin * hv[:, l]))
        B_prev = Bv @ Bk[l, :, :, nf]
        inv_B_prev = inv_Bk[l, :, :, nf] @ np.diag(np.exp(-spin * hv[:, l]))
        GR[:, :, nf] = B_prev @ GR[:, :, nf] @ inv_B_prev

    return GR, hv

def Wrap_downward(self, GR, Bk, inv_Bk, hv, l):
    N_FL = self.N_FL
    for nf in range(N_FL):
        spin = 1 - 2 * nf
        Bv = np.diag(np.exp(spin * hv[:, l]))
        B_prev = Bv @ Bk[l, :, :, nf]
        inv_B_prev = inv_Bk[l, :, :, nf] @ np.diag(np.exp(-spin * hv[:, l]))
        GR[:, :, nf] = inv_B_prev @ GR[:, :, nf] @ B_prev

    return GR, hv


def WrapGRup(self, GR, Bk, inv_Bk, hv, phase, rng, Weight, l, acceptance_rate, count):
    GR, hv = Wrap_upward(self, GR, Bk, inv_Bk, hv, l)
    GR, hv, phase, Weight, acceptance_rate, count = UpdateGR(
        self, GR, hv, phase, rng, Weight, l, acceptance_rate, count
    )
    return GR, hv, phase, Weight, acceptance_rate, count


def WrapGRdo(self, GR, Bk, inv_Bk, hv, phase, rng, Weight, l, acceptance_rate, count):
    GR, hv, phase, Weight, acceptance_rate, count = UpdateGR(
        self, GR, hv, phase, rng, Weight, l, acceptance_rate, count
    )
    GR, hv = Wrap_downward(self, GR, Bk, inv_Bk, hv, l)
    return GR, hv, phase, Weight, acceptance_rate, count







def UpdateGR0(self, GR, hv, phase, rng,  Weight, l, acceptance_rate, count):
    N_FL = self.N_FL
    Ndim = hv.shape[0]
    id = np.identity(Ndim, dtype=np.float32)
    #alpha_vec = np.zeros(N_FL, dtype=np.float32)
    #p_vec = np.zeros(N_FL, dtype=np.float32)
    
    #alpha_array = np.zeros(N_FL)
    #w_ratio_array = np.zeros(N_FL)
    
    alpha_array = np.zeros(N_FL, dtype=np.float64)
    w_ratio_array = np.zeros(N_FL, dtype=np.float64)


    for site in range(Ndim):
        for nf in range(N_FL):
            spin = 1 - 2 * nf
            alpha_array[nf] = np.exp(-2 * spin * hv[site, l]) - 1
            w_ratio_array[nf] = 1 + alpha_array[nf] * (1 - GR[site, site, nf])
        
        # Total ratio
        w_ratio = np.prod(w_ratio_array)# w_ratio_up * w_ratio_down
        # Update counter for acceptance_rate
        count += 1
        
         # Metropolis algorithm
        phase_change_propose = w_ratio / np.abs(w_ratio)
        phase_propose = phase * phase_change_propose
        p = np.abs(w_ratio * (1 + phase_propose ** -2) / (1 + phase ** -2))
        
        if rng.random() <= np.abs(p):             
            for nf in range(N_FL):
                #u = id - GR[:, :, nf]
                #GR_update = (alpha_array[nf] / w_ratio_array[nf]) * GR[:,:,nf][:, [site]] @ u[[site], :]
                GRC_T = id - GR[:, :, nf]
                #GR[:, :, nf] = GR[:, :, nf] - (alpha_array[nf] / w_ratio_array[nf]) * GR[:, [site], nf] @ GRC_T[[site], :] # Matrix multiplication
                GR[:, :, nf] = GR[:, :, nf] - (alpha_array[nf] / w_ratio_array[nf]) * GR[:,:,nf][:, [site]] @ GRC_T[[site], :] # Matrix multiplication
            hv[site, l] = -hv[site, l]
            # Update phase
            phase = phase_propose
           
            # Update acceptance_rate
            acceptance_rate = ((count - 1) * acceptance_rate + 1) / count
        else:
            # Update acceptance_rate
            acceptance_rate = (count - 1) * acceptance_rate / count
    return  phase, acceptance_rate, count


def WrapGRup0(self, GR, Bk, inv_Bk, hv, phase, rng, Weight, l, acceptance_rate, count):
    GR = Wrap_upward0(self, GR, Bk, inv_Bk, hv, l)
    phase, acceptance_rate, count  = UpdateGR0(self, GR, hv, phase, rng, Weight, l, acceptance_rate, count)
    return phase, acceptance_rate, count


def WrapGRdo0(self, GR, Bk, inv_Bk, hv, phase, rng, Weight,  l, acceptance_rate, count):
    phase, acceptance_rate, count = UpdateGR0(self, GR, hv, phase, rng, Weight, l, acceptance_rate, count)
    GR = Wrap_downward0(self, GR, Bk, inv_Bk, hv, l)
    return phase, acceptance_rate, count


def Wrap_upward0(self, GR, Bk, inv_Bk, hv, l):
    N_FL = self.N_FL
    Ltrot = hv.shape[1]
    for nf in range(N_FL):
        spin = 1 - 2 * nf
        B_prev = np.diag(np.exp(spin * hv[:,  l])) @ Bk[l,:,:,nf]
        inv_B_prev = inv_Bk[l,:,:,nf] @ np.diag(np.exp(-spin * hv[:,  l]))
        GR[:, :, nf] = B_prev @ GR[:,:,nf] @ inv_B_prev
    return GR

def Wrap_downward0(self, GR, Bk, inv_Bk, hv, l):
    N_FL = self.N_FL
    Ltrot = hv.shape[1]
    for nf in range(N_FL):
        spin = 1 - 2 * nf
        B_prev = np.diag(np.exp(spin * hv[:,  l])) @ Bk[l,:,:,nf]
        inv_B_prev = inv_Bk[l,:,:,nf] @ np.diag(np.exp(-spin * hv[:,  l]))
        GR[:, :, nf] = inv_B_prev @ GR[:,:,nf] @ B_prev
    return GR


