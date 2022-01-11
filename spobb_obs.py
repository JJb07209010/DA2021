"""
observation from nature(x_t)
Load:
    x_t.txt
Save:
    y_o.txt
    R.txt
    H.txt
"""
import numpy as np
from settings import *
import matplotlib.pyplot as plt

dpath = './data/'

# true state
x_t = np.genfromtxt(dpath + 'x_t.txt')

# obs every 4 grids
skip = 4
sigma = 0.2   # size of the obs standard deviation

# initial y_o: obs at the initial time (assign NaN)
temp = np.zeros(int(N/skip*3))
temp_avg = np.zeros(int(N/skip))
temp_thin = np.zeros(int(N/skip))
group = 0
for i in range(int(N/skip)):
    mid = group*skip
    temp[0+3*i] = x_t[0, mid-1]
    temp[1+3*i] = x_t[0,mid]
    temp[2+3*i]  = x_t[0,mid+1]
    temp_avg[i] = (x_t[0, mid-1] + x_t[0, mid] + x_t[0, mid+1])/3
    temp_thin[i] = x_t[0,mid]
   
    group += 1

y_o_save = np.array([temp], dtype = 'f8')   # [t, x]
y_o_save_avg = np.array([temp_avg], dtype = 'f8')   # [t, x]
y_o_save_thin = np.array([temp_thin], dtype='f8')   # [t, x]

t = 1
while t <= nT:
    y_o = np.zeros(int(N/skip*3))
    y_o_avg = np.zeros(int(N/skip))
    y_o_thin = np.zeros(int(N/skip))
    
    group = 0
    for i in range(int(N/skip)):
        # + random obs error
        mid = group*skip
        y_o[0+3*i] = x_t[t, mid-1] + sigma*np.random.randn(1)
        y_o[1+3*i] = x_t[t,mid] + sigma*np.random.randn(1)
        y_o[2+3*i]  = x_t[t,mid+1] + sigma*np.random.randn(1)
        y_o_avg[i] = (y_o[0+3*i] + y_o[1+3*i] + y_o[2+3*i])/3
        y_o_thin[i] = y_o[1+3*i]
        group += 1
    
    y_o_save = np.vstack([y_o_save, [y_o]])
    y_o_save_avg = np.vstack([y_o_save_avg, [y_o_avg]])
    y_o_save_thin = np.vstack([y_o_save_thin, [y_o_thin]])
    
    t += 1

## error covariance matrix
R = np.diag(np.full(y_o_save.shape[1], sigma**2, dtype = 'f8'))
R_avg = np.diag(np.full(y_o_save_avg.shape[1], sigma**2, dtype = 'f8'))
R_thin = np.diag(np.full(y_o_save_thin.shape[1], sigma**2, dtype = 'f8'))

## observation operator H
# H for superobbing before avg
H = np.zeros([y_o_save.shape[1], x_t.shape[1]])
# alter here to make different H
group = 0
for j in range(H.shape[0]):    
    mid = group*skip
    if (j%3 == 0):
        H[j, mid-1] = 1/2 # mid-1 & mid = 1/2
        H[j, mid] = 1/2
    if (j%3 == 1):
        H[j,mid] = 1
    if (j%3 == 2):
        H[j,mid] = 1/2
        H[j,mid+1] = 1/2    # mid & mid+1 = 1/2
        group += 1

# H for superobbing after avg
H_avg = np.zeros([y_o_save_avg.shape[1], x_t.shape[1]])
for i in range(H_avg.shape[0]):
    H_avg[i,i*skip] = 1
H_thin = H_avg

np.savetxt(dpath + 'y_o_spobb_every' + str(skip) + '.txt', y_o_save)
np.savetxt(dpath + 'R_spobb_every' + str(skip) + '.txt', R)
np.savetxt(dpath + 'H_spobb_every' + str(skip) + '.txt', H)

np.savetxt(dpath + 'y_o_avg_spobb_every' + str(skip) + '.txt', y_o_save_avg)
np.savetxt(dpath + 'R_avg_spobb_every' + str(skip) + '.txt', R_avg)
np.savetxt(dpath + 'H_avg_spobb_every' + str(skip) + '.txt', H_avg)

np.savetxt(dpath + 'y_o_thin_spobb_every' + str(skip) + '.txt', y_o_save_thin)
np.savetxt(dpath + 'R_thin_spobb_every' + str(skip) + '.txt', R_thin)
np.savetxt(dpath + 'H_thin_spobb_every' + str(skip) + '.txt', H_thin)

if __name__ == '__main__':
    plt.scatter(y_o_save_thin[:,2] - x_t[:,2*skip], y_o_save_thin[:,3] - x_t[:,3*skip])
    plt.show()