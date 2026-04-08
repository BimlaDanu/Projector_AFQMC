import numpy as np

def Control_precisionG(self, GR, GR_test, deltaG_threshold, deltaG_max, deltaG_mean, count):
    if np.any(np.isnan(GR)) or np.any(np.isnan(GR_test)):
        raise Exception('Calculation aborted, NaN in GR detected')
    dG = np.abs(GR - GR_test)
    dG_max = np.max(dG)
    dG_mean = np.mean(dG)
    if dG_max <= deltaG_threshold:
        deltaG_max = np.maximum(deltaG_max, dG_max)
        deltaG_mean = ((count - 1) * deltaG_mean + dG_mean) / count
    #else:
    #    raise Exception(f'Calculation aborted, delta G = {dG_max} is larger than {deltaG_threshold} ! Please reduce nwrap')
    return deltaG_max, deltaG_mean

def Control_precisionP(self, phase, phase_test, deltaP_max, deltaP_mean, count):
    dP = np.abs(phase - phase_test)
    deltaP_max = np.maximum(deltaP_max, dP)
    deltaP_mean = ((count - 1) * deltaP_mean + dP) / count
    return deltaP_max, deltaP_mean
