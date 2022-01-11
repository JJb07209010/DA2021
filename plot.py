"""
Plot the data assimilation results
Read:
  x_t.txt
  x_b.txt
  x_a.txt
"""
import numpy as np
from settings import *
import matplotlib.pyplot as plt

dpath = './data/'
#exp = 'thin_spobb_every4'
exp = 'every1'
tskip = 1
#---------------------------------------
#dpath = './data_half_dt/'
#exp = 'avg_every2'
# full: 1, thin: 2, avg: 2
#tskip = 2

ls = '-'    # linestyle for the exp

# load data
x_t_save = np.genfromtxt(dpath + 'x_t.txt')
#x_b_save = np.genfromtxt(dpath + 'x_b.txt')
OI_x_a_save = np.genfromtxt('./OI_output/' + 'x_a_' + exp + '.txt')
EKF_x_a_save = np.genfromtxt('./EKF_output/' + 'x_a_' + exp + '.txt')
Var3D_x_a_save = np.genfromtxt('./3DVar_output/' + 'x_a_' + exp + '.txt')
inc_3DVar_x_a_save = np.genfromtxt('./inc_3DVar_output/' + 'x_a_' + exp + '.txt')
#y_o = np.genfromtxt(dpath + 'y_o_every1.txt')


## RMSE
RMSE_OI = np.sqrt(np.mean((OI_x_a_save - x_t_save[::tskip])**2, axis = 1))
RMSE_EKF = np.sqrt(np.mean((EKF_x_a_save - x_t_save[::tskip])**2, axis = 1))
RMSE_3DVar = np.sqrt(np.mean((Var3D_x_a_save - x_t_save[::tskip])**2, axis = 1))
RMSE_inc_3DVar = np.sqrt(np.mean((inc_3DVar_x_a_save - x_t_save[::tskip])**2, axis = 1))

## plot
f, [ax1,ax2] = plt.subplots(2, 1, sharex = 'col', gridspec_kw = {'height_ratios': [3,1]}, dpi = 200)
f.subplots_adjust(hspace = 0.1)

ax1.plot(np.arange(nT+1)[::tskip] * dT, RMSE_OI, label = 'OI')
ax1.plot(np.arange(nT+1)[::tskip] * dT, RMSE_EKF, label = 'EKF')
ax1.plot(np.arange(nT+1)[::tskip] * dT, RMSE_3DVar, label = '3DVar')
ax1.plot(np.arange(nT+1)[::tskip] * dT, RMSE_inc_3DVar, label = 'Inc_3DVar')

# axis settings
ax1.set_ylim(0, 6.5)
ax1.set_ylabel('RMSE', size=12, y = 0.3)
ax1.set_xlim(0, 40)
ax1.set_title(exp, loc = 'right', size = 9)
ax1.legend(loc = 'upper right')
ax1.grid()

ax2.set_xlabel(r'$t$', size=12)
ax2.plot(np.arange(nT+1)[::tskip] * dT, RMSE_OI, linestyle = ls, label = 'OI')
ax2.plot(np.arange(nT+1)[::tskip] * dT, RMSE_EKF, linestyle = ls, label = 'EKF')
ax2.plot(np.arange(nT+1)[::tskip] * dT, RMSE_3DVar, linestyle = ls, label = '3DVar')
ax2.plot(np.arange(nT+1)[::tskip] * dT, RMSE_inc_3DVar, linestyle = ls, label = 'Inc_3DVar')
ax2.grid()

ax2.set_ylim(0, 0.2)

plt.suptitle('RMSE Time series', size=15, y = 0.96)
plt.show()















