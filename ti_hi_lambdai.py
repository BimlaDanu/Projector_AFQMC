#import autograd.numpy as np
import numpy as np
import matplotlib.pyplot as plt

def Ut_couplings(self, adiabatic_on, Theta):
    Ham_U_max  = self.Ham_U
    beta = self.Beta
    dtau = self.dtau
    Nwrap = self.Nwrap
    Ltrot = max(Nwrap, 2 * round((beta + Theta) / (2 * dtau)))  # # of time steps
    dtau = (beta + Theta) / Ltrot  # time steps
    L_num = round(beta / (2 * dtau))  # # of time steps for the evolution
    U_array = np.zeros(Ltrot)
    print('Ltrot, adiabatic_on, Theta, Ham_U_max, beta, dtau, Nwrap = ',Ltrot, adiabatic_on, Theta, Ham_U_max, beta, dtau, Nwrap)
    for l in range(1, Ltrot + 1):   
        if l * dtau <= beta/2: 
            U_array[l - 1] = Ham_U_max * (1 - adiabatic_on * (l - 1) /L_num)
        elif l * dtau > (beta / 2 + Theta):
            U_array[l - 1] = Ham_U_max * (1 - adiabatic_on * (Ltrot - l) /L_num)
            #print('l,   u_vec =', l,   U_array)
    lambda_array = np.arccosh(np.exp(U_array * dtau/2))
    #print('lambda_array =', l, lambda_array)
    return lambda_array, U_array, Ltrot

def time_dependent_int(n, beta, Ur, dtau, hirsch):
    hi = np.zeros(n)
    ti = np.zeros(2*n+1)
    lambdai = np.zeros(2*n+1)
    
    U = abs(Ur)
    nt = 2 * n + 1
    cost = beta / dtau / 2.0
    error = 1.0
    eps = 1.0e-14
    maxitr = 100
    
    if cost > float(n):
        gmin = 0.0
        gmax = 1.0
    elif cost > 1.0:
        gmin = 1.0
        gmax = cost / (cost - 1.0)
    else:
        gamma = 1.0
        error = 0.0
    
    itr = 0
    while error > eps and itr < maxitr:
        gamma = (gmin + gmax) / 2.0
        fun_val = fun(gamma, n)
        if fun_val > cost:
            gmin = gamma
        else:
            gmax = gamma
        error = abs(fun_val - cost)
        itr += 1
    
    hi[n-1] = dtau
    for i in range(n-2, -1, -1):
        hi[i] = hi[i+1] / gamma
    
    #lambdai[nt-1] = 0.0
    lambdai[nt-1] = dtau
    if hirsch:
        for i in range(1, n+1):
            costu = np.exp(Ur * hi[i-1] / 2.0)
            lambdai[i-1] = np.log(costu + np.sqrt(costu**2 - 1.0))
            lambdai[nt-i-1] = lambdai[i-1]
    else:
        for i in range(1, n+1):
            lambdai[i-1] = np.sqrt(U * hi[i-1])
            lambdai[nt-i-1] = lambdai[i-1]
    
    ti[0] = hi[0] / 2.0
    for i in range(1, n):
        ti[i] = (hi[i] + hi[i-1]) / 2.0
    ti[n] = hi[n-1]
    for i in range(1, n+1):
        ti[nt-i] = ti[i-1]
    
    return hi, ti, lambdai

def fun(p, n):
    cost = p ** (n - 1)
    return (1.0 - p * cost) / (1.0 - p) / cost

def time_dependent_int_2n(n, beta, Ur, dtau, hirsch):
    hi = np.zeros(n)
    nt = 2 * n                    # changed from 2*n+1
    ti = np.zeros(nt)
    lambdai = np.zeros(nt)
    
    U = abs(Ur)
    cost = beta / dtau / 2.0
    error = 1.0
    eps = 1.0e-14
    maxitr = 100
    
    if cost > float(n):
        gmin = 0.0
        gmax = 1.0
    elif cost > 1.0:
        gmin = 1.0
        gmax = cost / (cost - 1.0)
    else:
        gamma = 1.0
        error = 0.0
    
    itr = 0
    while error > eps and itr < maxitr:
        gamma = (gmin + gmax) / 2.0
        fun_val = fun(gamma, n)
        if fun_val > cost:
            gmin = gamma
        else:
            gmax = gamma
        error = abs(fun_val - cost)
        itr += 1
    
    # geometric hi
    hi[n-1] = dtau
    for i in range(n-2, -1, -1):
        hi[i] = hi[i+1] / gamma
    
    # symmetric lambdai (no central single point now)
    if hirsch:
        for i in range(n):
            costu = np.exp(Ur * hi[i] / 2.0)
            val = np.log(costu + np.sqrt(costu**2 - 1.0))
            lambdai[i] = val
            lambdai[nt - i - 1] = val
    else:
        for i in range(n):
            val = np.sqrt(U * hi[i])
            lambdai[i] = val
            lambdai[nt - i - 1] = val
    
    # time grid (midpoint construction)
    ti[0] = hi[0] / 2.0
    for i in range(1, n):
        ti[i] = (hi[i] + hi[i-1]) / 2.0
    
    # mirror time grid
    for i in range(n):
        ti[nt - i - 1] = ti[i]
    return hi, ti, lambdai

def extract_hi_from_lambdai(lambdai, Ur, hirsch):
    """
    Reconstruct hi from lambdai.
    Assumes lambdai is symmetric and length = 2*n+1.
    """
    U = abs(Ur)
    nt = len(lambdai)
    if  nt % 2 == 1:
        n = (nt - 1) // 2
    else:
        n = (nt - 0) // 2
    hi_rec = np.zeros(n)
    if hirsch:
        for i in range(n):
            hi_rec[i] = (2.0 / Ur) * np.log(np.cosh(lambdai[i]))
    else:
        for i in range(n):
            hi_rec[i] = (lambdai[i] ** 2) / U

    return hi_rec


def compare_hi(hi, hi_rec):
    diff = hi - hi_rec
    rel_err = np.abs(diff) / (np.abs(hi) + 1e-16)
    print("Max abs error   :", np.max(np.abs(diff)))
    print("Max rel error   :", np.max(rel_err))
    print("Mean rel error  :", np.mean(rel_err))
    return diff, rel_err

def symmetric_ramping(beta, dtau, theta, ham_U, peak_width=3):
    """
    Generate a symmetric Hirsch lambda_i profile with a flat top.
    
    Parameters
    ----------
    beta : float
        Inverse temperature (for reference, not used directly)
    dtau : float
        Time step
    theta : float
        Total imaginary time
    ham_U : float
        Interaction strength
    peak_width : int
        Number of symmetric points at the top (odd)
    
    Returns
    -------
    g_t : np.ndarray
        Symmetric lambda_i array
    peak_indices : list of int
        Indices of the flat top points
    """
    # total slices
    Thtrot = int(theta / dtau)
    Ltrot = 2 * Thtrot + 1       # odd total slices
    mid = Ltrot // 2

    if peak_width % 2 == 0:
        raise ValueError("peak_width must be odd to keep symmetry.")

    # Allocate array
    g_t = np.zeros(Ltrot, dtype=np.float32)

    half_peak = peak_width // 2
    peak_indices = [mid - half_peak + i for i in range(peak_width)]
    
    # Fill loop: symmetric ramp up to flat top
    for nt in range(1, Thtrot + 2):  # +1 to include middle
        if nt == Thtrot + 1:
            hval = dtau
            for idx in peak_indices:
                g_t[idx] = np.arccosh(np.exp(ham_U * hval / 2.0))
        else:
            hval = dtau * nt / (Thtrot + 1)
            left  = nt - 1
            right = Ltrot - nt
            g_t[left]  = np.arccosh(np.exp(ham_U * hval / 2.0))
            g_t[right] = np.arccosh(np.exp(ham_U * hval / 2.0))
    return g_t, peak_indices, Ltrot 

def fixfirst1(nt, beta, Ur, lambdai, hirsch):
    cost = beta

    if hirsch:
        for i in range(1, (nt - 1) // 2):  # Adjusting for 0-based index in Python
            dtt = np.exp(lambdai[i])
            costi = 2.0 * np.log((1.0 + dtt**2) / (2.0 * dtt))
            cost -= costi / Ur
        
        if cost > 0.0:
            cost = np.exp(cost * Ur / 2.0)
            dtt = cost + np.sqrt(cost**2 - 1.0)
            #lambdai[0] = np.log(dtt)
            lambdai[i] = np.log(dtt)
        else:
            print("ERROR: beta inconsistency !!!")
            #lambdai[0] = 0.0
            lambdai[i] = 0.0
    else:
        for i in range(1, (nt - 1) // 2):
            cost -= lambdai[i]**2 / Ur
        
        if cost > 0.0:
            #lambdai[0] = np.sqrt(cost * Ur)
            lambdai[i] = np.sqrt(cost * Ur)
        else:
            print("ERROR: beta inconsistency !!!")
            #lambdai[0] = 0.0
            lambdai[i] = 0.0

    # Symmetrically assign values to the second half of the lambdai array
    for i in range(1, (nt - 1) // 2):
        lambdai[nt - i - 1] = lambdai[i]
    return lambdai

def fixfirst(nt, beta, Ur, lambdai, hirsch):
    cost = beta

    if hirsch:
        for i in range(1, (nt - 1) // 2):  # Adjusting for 0-based index in Python
            dtt = np.exp(lambdai[i])
            costi = 2.0 * np.log((1.0 + dtt**2) / (2.0 * dtt))
            cost -= costi / Ur
        
        if cost > 0.0:
            cost = np.exp(cost * Ur / 2.0)
            dtt = cost + np.sqrt(cost**2 - 1.0)
            lambdai[0] = np.log(dtt)
        else:
            print("ERROR: beta inconsistency !!!")
            lambdai[0] = 0.0
    else:
        for i in range(1, (nt - 1) // 2):
            cost -= lambdai[i]**2 / Ur
        
        if cost > 0.0:
            lambdai[0] = np.sqrt(cost * Ur)
        else:
            print("ERROR: beta inconsistency !!!")
            lambdai[0] = 0.0

    # Symmetrically assign values to the second half of the lambdai array
    for i in range(1, (nt - 1) // 2):
        lambdai[nt - i - 1] = lambdai[i]
    return lambdai


def fixfirst_b(nt, beta, Ur, lambdai, lambdaib, hirsch):
    U = abs(Ur)

    # Step 1: Combine `lambdaib(i)` and `lambdaib(nt-i)` symmetrically
    for i in range(1, (nt - 1) // 2):
        lambdaib[i] += lambdaib[nt - i - 1]
        lambdaib[nt - i - 1] = 0.0

    # Step 2: Perform calculations based on the `hirsch` condition
    if hirsch:
        cost = beta
        # Iterate over the lambda elements to compute `cost` for hirsch case
        for i in range(1, (nt - 1) // 2):
            dtt = np.exp(lambdai[i])
            costi = 2.0 * np.log((1.0 + dtt**2) / (2.0 * dtt))
            cost -= costi / U
        
        if cost > 0.0:
            cost = np.exp(cost * U / 2.0)
            rootc = np.sqrt(cost**2 - 1.0)
            dtt = cost + rootc

            # Backpropagate derivatives
            dttb = lambdaib[0] / dtt
            costb = dttb * dtt / rootc
            costb = U / 2.0 * cost * costb
        else:
            print('ERROR: beta inconsistency!!!')
            costb = 0.0

        lambdaib[0] = 0.0

        # Further loop for updating `lambdaib`
        for i in range(1, (nt - 1) // 2):
            costib = -costb / U
            dtt = np.exp(lambdai[i])
            dttb = costib * 2.0 * (dtt**2 - 1.0) / (dtt**3 + dtt)
            lambdaib[i] += dttb * dtt

    else:
        cost = beta
        # Iterate over the lambda elements to compute `cost` for non-hirsch case
        for i in range(1, (nt - 1) // 2):
            cost -= lambdai[i]**2 / U

        if cost > 0.0:
            costb = lambdaib[0] / np.sqrt(cost / U) / 2.0
        else:
            costb = 0.0

        lambdaib[0] = 0.0

        # Further loop for updating `lambdaib`
        for i in range(1, (nt - 1) // 2):
            lambdaib[i] -= 2.0 * costb * lambdai[i] / U
    return lambdai, lambdaib



def time_dependent_int_b(Ur, n, beta, betan, dtau, dtaun, hi, hin, gamma, lambdai, lambdain, ti, tin, hirsch):
    U = abs(Ur)
    cost = beta / dtau / 2.0
    nt = 2 * n + 1
    hin = np.zeros(n)
    gamman = 0.0

    # First loop (equivalent to: do i = n, 1, -1)
    for i in range(n, 0, -1):
        tin[i-1] = tin[i-1] + tin[nt - i]
        tin[nt - i] = 0.0

    # Second loop
    hin[n-1] = hin[n-1] + tin[n]
    tin[n] = 0.0

    # Third loop (equivalent to: do i = n - 1, 1, -1)
    for i in range(n-1, 0, -1):
        hin[i-1] = hin[i-1] + tin[i] / 2.0
        hin[i] = hin[i] + tin[i] / 2.0
        tin[i] = 0.0

    # Fourth loop
    hin[0] = hin[0] + tin[0] / 2.0
    tin[0] = 0.0

    if hirsch:
        # Hirsch case (equivalent to: do i = n, 1, -1)
        for i in range(n, 0, -1):
            lambdain[i-1] = lambdain[i-1] + lambdain[nt - i]
            lambdain[nt - i] = 0.0
            costu = np.exp(U * hi[i-1] / 2.0)
            costun = 1.0 / np.sqrt(costu**2 - 1.0) * lambdain[i-1]
            hin[i-1] = hin[i-1] + U / 2.0 * costu * costun
            lambdain[i-1] = 0.0
    else:
        # Non-Hirsch case
        for i in range(n, 0, -1):
            lambdain[i-1] = lambdain[i-1] + lambdain[nt - i]
            lambdain[nt - i] = 0.0
            hin[i-1] = hin[i-1] + 0.5 / lambdai[i-1] * U * lambdain[i-1]
            lambdain[i-1] = 0.0

    lambdain[nt-1] = 0.0

    # Another loop (equivalent to: do i = 1, n - 1)
    for i in range(1, n):
        hin[i] = hin[i] + hin[i-1] / gamma
        gamman = gamman - hi[i] / gamma**2 * hin[i-1]
        hin[i-1] = 0.0

    dtaun = dtaun + hin[n-1]
    hin[n-1] = 0.0

    # Compute derivative
    cost_der = (n - 1.0 - n * gamma + gamma**n)
    if gamma != 1.0 and cost_der != 0.0:
        der = -cost_der / (gamma - 1.0)**2 / gamma**n
    else:
        der = -((n - 1) * n) / 2.0

    costn = gamman / der
    gamman = 0.0

    # Update dtaun and betan
    dtaun = dtaun - cost / dtau * costn
    betan = betan + costn / dtau / 2.0

    # Print statements for debugging
    print('dtaun, betan, costn, cost, gamman, der =', dtaun, betan, costn, cost, gamman, der)
    print('tb1 =', tin[:nt])
    print('hb1 =', hin[:n])
    print('lb1 =', lambdain[:nt])

    return betan, dtaun, hin, tin, lambdain

#if  __name__ == '__main__':
#    beta = 4.0
#    Ur = 2.0
#    dtau = 0.1
#    n = 5
#    hirsch = True

#    hi, ti, lambdai = time_dependent_int(n, beta, Ur, dtau, hirsch)
    # Printing results
#    print('t=', ti)

