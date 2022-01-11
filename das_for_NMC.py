"""
The data assimilation system (no assimilation example)
Load:
  x_a_init.txt
Save:
  x_b.txt
  x_a.txt
"""
import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from time import sleep

dpath = './data/'

time = 0
while time <= 7:
    # load initial condition
    x_a_init = np.genfromtxt(dpath + 'x_a_init.txt')
    #x_a_init = np.genfromtxt(dpath + 'x_t.txt')[0] + 1.e-4  # using nature run value plus a small error (for test purpose)
    
    # load observations
    y_o_save = np.genfromtxt(dpath + 'y_o_every1.txt')
    R = np.genfromtxt(dpath + 'R_every1.txt')   # obs error covariance
    H = np.genfromtxt(dpath + 'H_every1.txt')   # obs operator
    
    # initial x_b: no values at the initial time (assign NaN)
    x_b_save = np.full((1,N), np.nan, dtype='f8')
    P_b = np.genfromtxt(dpath + 'P_b.txt')    # background error covariance
    
    # initial x_a: from x_a_init
    x_a_save = np.array([x_a_init])
    
    tt = 1
    while tt <= nT:
        tts = tt - 1
        Ts = tts * dT  # forecast start time
        Ta = tt  * dT  # forecast end time (DA analysis time)
        print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))
    
        #--------------
        # forecast step
        #--------------
    
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_a_save[tts], Ts).set_f_params(F)
        solver.integrate(Ta)
        x_b_save = np.vstack([x_b_save, [solver.y]])
    
        #--------------
        # analysis step
        #--------------
    
        # background
        x_b = x_b_save[tt].transpose()
    
        # observation
        y_o = y_o_save[tt].transpose()
    
        # innovation
        d = y_o - np.dot(H, x_b)
    
        # analysis scheme (no assimilation in this example)
        #x_a = x_b
        
        # analysis scheme (OI)
        PbHt = np.dot(P_b, H.transpose())
        HPbHt = np.dot(np.dot(H, P_b), H.transpose())
        K = np.dot(PbHt, inv(HPbHt + R))
        x_a = x_b + np.dot(K, d)
    
        x_a_save = np.vstack([x_a_save, x_a.transpose()])
        tt += 1
    
    # save background and analysis data
    np.savetxt(dpath + 'x_b_every1.txt', x_b_save)
    np.savetxt(dpath + 'x_a_every1.txt', x_a_save)
    
    
    #################################################
    # NMC
    #################################################
    ## take a initial analysis (bad snalysis)
    x_a = np.genfromtxt(dpath + 'x_a_every1.txt')   # [t,x] load the data with 40 points(total)
    cylen = x_a.shape[0]    # cycle length
    
    ## repeating forecast starting at different time
    t1 = 0.8-0.05    # integrate length, 4*dt, like 24h for dt=6h
    t2 = 0.8    # integrate length, 8*dt, like 48h for dt=6h
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
        
        del solver
        ## sum the difference in same valid time
        diff = (yt2 - yt1).reshape((40,1))
        summation += np.dot(diff, diff.transpose())
    
    
    a = 0.25 # rescaling factor
    P_b = np.zeros_like(summation, dtype = 'f8')
    P_b[:,:] = a*summation/(cylen-1)
    np.savetxt(dpath + 'P_b.txt', P_b)  # [40, 40]
    
    #if __name__ == '__main__':
        
    #import matplotlib.pyplot as plt
    plt.figure(figsize = (6,5), dpi = 100)
    cx, cy = np.meshgrid(np.arange(N), np.arange(N))
    norm = TwoSlopeNorm(vcenter = 0)
    plt.pcolormesh(cx, cy, P_b, cmap = 'bwr', norm = norm, shading = 'nearest')
    plt.colorbar()
    plt.title(r'$\alpha$ = %.1f'%a, loc = 'right')
    plt.show()
    #plt.savefig('./fig/NMC_7_Pb.png')
    print(time)
    time += 1
    sleep(1)
    