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

# obs every 2 grids
skip = 4
sigma = 0.2   # size of the obs standard deviation

# initial y_o: obs at the initial time (assign NaN)
y_o_save = np.array([x_t[0, ::skip]], dtype='f8')   # [t, x]

t = 1
while t <= nT:
    y_o = x_t[t, ::skip] + sigma*np.random.randn(y_o_save.shape[1])   # + random obs error
    y_o_save = np.vstack([y_o_save, [y_o]])
    
    t += 1

## error covariance matrix
R = np.diag(np.full(y_o_save.shape[1], sigma**2, dtype = 'f8'))

## observation operator H
H = np.zeros([y_o_save.shape[1], x_t.shape[1]])
# alter here to make different H
for i in range(H.shape[0]):
    H[i,i*skip] = 1

np.savetxt(dpath + 'y_o_every' + str(skip) + '.txt', y_o_save)
np.savetxt(dpath + 'R_every' + str(skip) + '.txt', R)
np.savetxt(dpath + 'H_every' + str(skip) + '.txt', H)


if __name__ == '__main__':
    plt.scatter(y_o_save[:,2] - x_t[:,2*skip], y_o_save[:,3] - x_t[:,3*skip])
    plt.show()