import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

dpath = './data/'

## take a initial analysis (bad snalysis)
#x_a = np.genfromtxt('./OI_output/x_a_every1.txt')   # [t,x] load the data with 40 points(total)
x_a = np.genfromtxt('./OI_output/x_a_every1_R_as_Pb.txt')
cylen = x_a.shape[0]    # cycle length

## repeating forecast starting at different time
t1 = 0.4-0.05    # integrate length, 4*dt, like 24h for dt=6h
t2 = 0.4    # integrate length, 8*dt, like 48h for dt=6h
a = 0.1 # rescaling factor
P_b = np.zeros((N,N), dtype = 'f8')
for d in range(20):
    # d means distance
    summation = 0
    for t in range(1, cylen):
        ## set the different initial time
        ## past
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_a[t-1], 0).set_f_params(F)
        ## integrate for 0.4 
        solver.integrate(t2)
        yt2 = solver.y
        
        del solver
        ## now
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_a[t], 0).set_f_params(F)
        ## integrate for 0.2 
        solver.integrate(t1)
        yt1 = solver.y
        
        ## sum the difference in same valid time
        diff = (yt2 - yt1).reshape((40,1))
        Pb_mtrx = np.dot(diff, diff.transpose())
        for i in range(N):
            pos = i+d  # positive direction
            neg = i-d  # negative direction
            if (pos >= 40):
                pos = i+d -40
            summation += (Pb_mtrx[i,pos] + Pb_mtrx[i,neg])

    for k in range(N):
        pos = k+d
        neg = k-d
        if (pos >= 40):
            pos = k+d -40
        P_b[k,pos] = a*summation/(2*N*cylen-1)
        P_b[k,neg] = a*summation/(2*N*cylen-1)

#np.savetxt(dpath + 'P_b.txt', P_b)  # [40, 40]

if __name__ == '__main__':   
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    #P_b = np.genfromtxt(dpath + 'P_b.txt')
    plt.figure(figsize = (6,5), dpi = 100)
    cx, cy = np.meshgrid(np.arange(N), np.arange(N))
    norm = TwoSlopeNorm(vcenter = 0)
    plt.pcolormesh(cx, cy, P_b, cmap = 'bwr', norm = norm, shading = 'nearest')
    plt.colorbar()
    plt.xticks(np.arange(4, 40, 5), np.arange(4, 40, 5)+1)
    plt.yticks(np.arange(4, 40, 5), np.arange(4, 40, 5)+1)
    plt.title(r'$\alpha$ = %.2f'%a, loc = 'right')
    #plt.savefig('./fig/NMC_first_Pb.png')
    
    
    
    
    
    