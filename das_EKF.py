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

#dpath = './data/'
#exp = 'avg_spobb_every4'
#exp = 'every2'
#tskip = 1
dpath = './data_half_dt/'
exp = 'avg_every2'
# full: 1, thin: 2, avg: 2
tskip = 2

# load initial condition
x_a_init = np.genfromtxt(dpath + 'x_a_init.txt')
#x_a_init = np.genfromtxt('x_t.txt')[0] + 1.e-4  # using nature run value plus a small error (for test purpose)

# load observations
y_o_save = np.genfromtxt(dpath + 'y_o_' + exp + '.txt')
R = np.genfromtxt(dpath + 'R_' + exp + '.txt')   # obs error covariance
H = np.genfromtxt(dpath + 'H_' + exp + '.txt')   # obs operator

# initial x_b: no values at the initial time (assign NaN)
x_b_save = np.full((1,N), np.nan, dtype='f8')
P_b = np.genfromtxt(dpath + 'P_b.txt')    # background error covariance
P_a = P_b

# initial x_a: from x_a_init
x_a_save = np.array([x_a_init])

## initial TLM
f_jacob = np.zeros([P_b.shape[0],P_b.shape[1]]) # jacobian of the function

## identity matrix
I = np.diag(np.full(x_a_init.shape[0], 1))  

## inflation factor
## --------------------------------------------------------------------------------
# every1: 1.08,  every2: 1.2,  thin_spobb_4: 1.3,  spobb_4: 1.4,  avg_spobb_4: 1.5
# full_every2: 1.1,  thin_every2: 1.1, avg_every2: 1.1
## --------------------------------------------------------------------------------
rho = 1.1

## transition little time step number
trans_no = 25

nT = int(nT/tskip)
tt = 1
while tt <= nT:
    tts = (tt - 1)
    Ts = tts * dT*tskip  # forecast start time
    Ta = tt  * dT*tskip  # forecast end time (DA analysis time)
    print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))

    #--------------
    # forecast step
    #--------------
    ## apply small time step for transition model ------------
    ddT = (Ta-Ts)/trans_no
    x_ini = x_a_save[tts]
    M = I
    for i in range(trans_no):
        sTs = Ts + i*ddT
        sTa = Ts + (i+1)*ddT
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x_ini, sTs).set_f_params(F)
        solver.integrate(sTa)
        
        ## construct TLM
        # TLM
        for row in range(f_jacob.shape[0]):
            f_jacob[row, 0+row] = -1
            f_jacob[row, -39+row] = np.roll(x_ini, 1)[row]
            f_jacob[row, -2+row] = -np.roll(x_ini, 1)[row]
            f_jacob[row, -1+row] = np.roll(x_ini, -1)[row] - np.roll(x_ini, 2)[row]
        
        L = I + ddT*f_jacob
        M = np.dot(L, M)
        x_ini = solver.y
    ## apply small time step for transition model ------------
    
    x_b_save = np.vstack([x_b_save, [x_ini]])
    P_b = rho*np.dot(np.dot(M, P_a), M.transpose())
    #dPdt = np.dot(f_jacob, P_a) + np.dot(f_jacob, P_a).transpose()
    #P_b = (P_a + dPdt*dT)
    
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
    
    # analysis scheme (EKF)
    PbHt = np.dot(P_b, H.transpose())
    HPbHt = np.dot(np.dot(H, P_b), H.transpose())
    K = np.dot(PbHt, inv(HPbHt + R))
    x_a = x_b + np.dot(K, d)
    P_a = np.dot(I - np.dot(K, H), P_b)

    x_a_save = np.vstack([x_a_save, x_a.transpose()])
    tt += 1
    
# save background and analysis data
opath = './EKF_output/'
np.savetxt(opath + 'x_b_' + exp + '.txt', x_b_save)
np.savetxt(opath + 'x_a_' + exp + '.txt', x_a_save)
    
    
    
    