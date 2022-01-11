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

dpath = './data_half_dt/'

# true state
x_t = np.genfromtxt(dpath + 'x_t.txt')

# obs every 2 grids
skip = 2
sigma = 0.2   # size of the obs standard deviation

# initial y_o: obs at the initial time (assign NaN)
y_o_save_full = np.array([x_t[0, ::skip]], dtype='f8')   # [t, x]
y_o_save_thin = np.array([x_t[0, ::skip]], dtype='f8')   # [t, x]
y_o_save_avg = np.array([x_t[0, ::skip]], dtype='f8')   # [t, x]

t = 1
mid = 0
while t <= nT:
    y_o_full = x_t[t, ::skip] + sigma*np.random.randn(y_o_save_full.shape[1])   # + random obs error
    y_o_save_full = np.vstack([y_o_save_full, [y_o_full]])
    print('t_full: ', t)
    
    if (t%2 == 0):
        y_o_thin = y_o_full
        y_o_save_thin = np.vstack([y_o_save_thin, [y_o_thin]])
        mid = t
        print('mid: ', mid)
    
    if ((t == mid+1) & (t != 1)):
        y_o_avg = np.mean(y_o_save_full[mid-1:mid+2,:], axis = 0)
        y_o_save_avg = np.vstack([y_o_save_avg, [y_o_avg]])
        print('t_avg: ', t)
    if(t == nT):
        y_o_avg = np.mean(y_o_save_full[mid-1:mid+1,:], axis = 0)
        y_o_save_avg = np.vstack([y_o_save_avg, [y_o_avg]])
        print('final avg: ',t)
    t += 1

## error covariance matrix
R_full = np.diag(np.full(y_o_save_full.shape[1], sigma**2, dtype = 'f8'))
R_thin = R_full
R_avg = R_full
## observation operator H
H_full = np.zeros([y_o_save_full.shape[1], x_t.shape[1]])
# alter here to make different H
for i in range(H_full.shape[0]):
    H_full[i,i*skip] = 1

H_thin = H_full
H_avg = H_full

np.savetxt(dpath + 'y_o_full_every' + str(skip) + '.txt', y_o_save_full)
np.savetxt(dpath + 'R_full_every' + str(skip) + '.txt', R_full)
np.savetxt(dpath + 'H_full_every' + str(skip) + '.txt', H_full)

np.savetxt(dpath + 'y_o_thin_every' + str(skip) + '.txt', y_o_save_thin)
np.savetxt(dpath + 'R_thin_every' + str(skip) + '.txt', R_thin)
np.savetxt(dpath + 'H_thin_every' + str(skip) + '.txt', H_thin)

np.savetxt(dpath + 'y_o_avg_every' + str(skip) + '.txt', y_o_save_avg)
np.savetxt(dpath + 'R_avg_every' + str(skip) + '.txt', R_avg)
np.savetxt(dpath + 'H_avg_every' + str(skip) + '.txt', H_avg)


if __name__ == '__main__':
    plt.scatter(y_o_save_full[:,2] - x_t[:,2*skip], y_o_save_full[:,3] - x_t[:,3*skip])
    plt.show()
