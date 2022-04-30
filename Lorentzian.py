import os
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import size


def _lorentzian_sg(x, amp, cen, sigma):
    return amp / (1 + np.square((x - cen) / sigma))


def _lorentzian_multi(x, *argss):
    n = len(argss)
    p = n // 3
    total = 0
    pp = np.arange(0, p, 1)
    for i in pp:
        total += _lorentzian_sg(x, argss[3 * i], argss[3 * i + 1], argss[3 * i + 2])
    return total


x_array = np.linspace(1, 100, 500)
amp1 = 100
cen1 = 40
sigma1 = 10


# amp2 = 75
# cen2 = 65
# sigma2 = 5


# input amp, sigma and center for each peak
argus = [100, 40, 10, 75, 65, 5]
# two peaks
y1 = argus[0] * (1 / (argus[2] * (np.sqrt(2 * np.pi)))) * (
    np.exp((-1.0 / 2.0) * (((x_array - argus[1]) / argus[2]) ** 2)))
y2 = argus[3] * (1 / (argus[5] * (np.sqrt(2 * np.pi)))) * (
    np.exp((-1.0 / 2.0) * (((x_array - argus[4]) / argus[5]) ** 2)))
y_noise = np.exp((-1.0 / 2.0) * (((x_array - argus[5]) / argus[5]) ** 2))
y_array = y1 + y2 + y_noise

# single peak
# y_array = amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (
#     np.exp((-1.0 / 2.0) * (((x_array - cen1) / sigma1) ** 2))) + amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (
#               np.exp((-1.0 / 2.0) * (((x_array - cen2) / sigma2) ** 2))) + (np.exp((np.random.ranf(50)))) / 5


plt.plot(x_array, y_array, 'k*', label='original')

popt, pcov = opt.curve_fit(_lorentzian_multi, x_array, y_array, p0=argus, method='lm')
num_peaks = size(popt)//3
pars = []
y_pred_sg = []
y_sum = [0]*500
# peak k
for k in np.arange(0, 2, 1):
    pa = popt[3*k:3*k+3]
    pars.append(pa)
    yp = _lorentzian_sg(x_array, *pa)
    y_pred_sg.append(yp)
    y_sum += yp
 #   plt.plot(x_array, yp, label='peak'+str(k+1))

# curve fit
plt.plot(x_array, y_sum, label='fit')

plt.legend()
plt.xlim(-5, 105)
plt.ylim(-0.5, 8)
plt.show()
