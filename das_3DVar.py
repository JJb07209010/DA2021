import numpy as np
from scipy.integrate import ode
from scipy.optimize import minimize
import lorenz96
from settings import *
from numpy.linalg import inv

## define the cost function
def J(x, x_b, P_b, H, R, y_o):
    d = y_o - np.dot(H, x)
    return np.dot(np.dot((x_b - x).transpose(), inv(P_b)), (x_b - x))/2 + \
           np.dot(np.dot(d.transpose(), inv(R)), d)/2

def grad_J(x, x_b, P_b, H, R, y_o):
    d = y_o - np.dot(H, x)
    return np.dot(inv(P_b), (x - x_b)) - np.dot(np.dot(H.transpose(), inv(R)), d)

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

# initial x_a: from x_a_init
x_a_save = np.array([x_a_init])

nT = int(nT/tskip)
tt = 1
while tt <= nT:
    tts = tt - 1
    Ts = tts * dT*tskip  # forecast start time
    Ta = tt  * dT*tskip  # forecast end time (DA analysis time)
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
    #d = y_o - np.dot(H, x_b)
    
    # analysis scheme (3DVar)
    # use x_b as first guess
    # innovation d is a function of x and include in the cost function 
    x0 = x_b
    x_a = minimize(J, x0, args = (x_b, P_b, H, R, y_o), jac = grad_J, method = 'CG').x

      
    x_a_save = np.vstack([x_a_save, x_a.transpose()])
    tt += 1

# save background and analysis data
opath = './3DVar_output/'
np.savetxt(opath + 'x_b_' + exp + '.txt', x_b_save)
np.savetxt(opath + 'x_a_' + exp + '.txt', x_a_save)