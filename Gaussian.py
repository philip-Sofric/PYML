import os
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-np.log(2) * np.square((x - center) / width))


def _2gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
    return amp1 * np.exp(-np.log(2) * np.square((x - cen1) / sigma1)) + \
           amp2 * np.exp(-np.log(2) * np.square((x - cen2) / sigma2))


# location = 'D:\Projects\Algorithm Study\Curve fitting'
# os.chdir(location)
# df = pd.read_csv('data3.csv')
# x = df['RamanShift']
# y = df['Intensity']
# popt, pcov = opt.curve_fit(_gaussian, x, y, p0=(amp_0, center_0, width_0))
# print(popt)
# y_pred = _gaussian(x, popt[0], popt[1], popt[2])
# plt.plot(x, y, color='green', label= 'original')
# plt.plot(x, y_pred, color='blue', label='fitted')
# plt.xlabel('Raman Shift')
# plt.ylabel('Intensity')
# plt.show()

# x_array = np.linspace(1, 100, 50)
# amp1 = 100
# sigma1 = 10
# cen1 = 50
# y_array_gauss = amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (
#     np.exp((-1.0 / 2.0) * (((x_array - cen1) / sigma1) ** 2))) + (np.exp((np.random.ranf(50)))) / 5
# plt.plot(x_array, y_array_gauss, 'ro', label='original')
# popt, pcov = opt.curve_fit(_gaussian, x_array, y_array_gauss, p0=(amp1, cen1, sigma1))
# y_pred = _gaussian(x_array, popt[0], popt[1], popt[2])
# plt.plot(x_array, y_pred, label='fitted')
# popt_m, pcov_m = opt.curve_fit(_gaussian, x_array, y_array_gauss, p0=(amp1, cen1, sigma1), method='trf')
# y_pred_m = _gaussian(x_array, *popt_m) +1
# plt.plot(x_array, y_pred_m, 'k', label='fitted *')

x_array = np.linspace(1, 100, 50)
amp1 = 100
sigma1 = 10
cen1 = 40

amp2 = 75
sigma2 = 5
cen2 = 65

y_array_2gauss = amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (
    np.exp((-1.0 / 2.0) * (((x_array - cen1) / sigma1) ** 2))) + \
                 amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (
                     np.exp((-1.0 / 2.0) * (((x_array - cen2) / sigma2) ** 2)))

y_noise_2gauss = (np.exp((np.random.ranf(50)))) / 5
y_array_2gauss += y_noise_2gauss
plt.plot(x_array, y_array_2gauss, 'ro', label='original')
popt_2gauss, pcov_2gauss = opt.curve_fit(_2gaussian, x_array, y_array_2gauss, p0=[amp1, cen1, sigma1,
                                                                                  amp2, cen2, sigma2])
residual_2gauss = y_array_2gauss - (_2gaussian(x_array, *popt_2gauss))
# print(popt_2gauss)
# peak 1
pars_1 = popt_2gauss[0:3]
gauss_peak_1 = _gaussian(x_array, *pars_1)
plt.plot(x_array, gauss_peak_1, label='peak1')
plt.fill_between(x_array, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)

# peak 2
pars_2 = popt_2gauss[3:6]
gauss_peak_2 = _gaussian(x_array, *pars_2)
plt.plot(x_array, gauss_peak_2, label='peak2')
plt.fill_between(x_array, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)

# fit curve
y_pred2 = _2gaussian(x_array, *popt_2gauss)
plt.plot(x_array, y_pred2)

bb = y_pred2 == (gauss_peak_1 + gauss_peak_2)
print(bb)
# residual
plt.plot(x_array, residual_2gauss, "bo", label='residual')

plt.legend()
plt.xlim(-5, 105)
plt.ylim(-0.5, 8)
plt.show()
