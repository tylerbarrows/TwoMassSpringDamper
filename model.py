## Abstract
# This looks at two trains conencted by a connector modeled as a spring and dampener
# We can specify a forcing function on either trains
# Train 1 is on the left, train 2 is on the right
# m1 is mass of train 1, m2 is mass of train 2
# k12 is spring
# c12 is mass dampener
# f1 is friction from train 1
# f2 is friction from tain 2

# We look at solving this using 5 methods:
# 1. Implicit with dual time stepping
# 2. Explicit
# 3. Discrete (One Riemann Sum)
# 4. Discrete (SVD Inverse)
# 5. Discrete (Pade Approximation)

# What the script does
# Compares the 5 methods
# Looks at using geometry constraints (how much can train coupling compress and expand)
# Looks at energy balance (not constrained and constrained)
# Looks at force and energy damper dissipation vs initial speed (no forcing function)

# Script calls solvingMethods.py to solve using different methods (also includes input forcing function calculation)

##
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import expm

# some formatting here
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
second_color = default_colors[1]

##
from solvingMethods import uInputCalc
from solvingMethods import dualTimeStepping
from solvingMethods import explicitSolution
from solvingMethods import discreteSolutionORS
from solvingMethods import discreteSolutionSVD
from solvingMethods import discreteSolutionPade

## parameters
g = 9.81 # m/s2
m1 = 500 # kg
m2 = 300 # kg
k12 = 800 # N/m
c12 = 1200 # N/(m/s)
f1 = 200 # N/(m/s)
f2 = 200 # N/(m/s)
A1 = 1750
omega_1 = 1 * 2*math.pi
# omega_1 = 0
A2 = 0
omega_2 = 0
u_params = np.array([A1, A2, omega_1, omega_2])

## geometry constraints
geo_constraint_flag = 0
# l_min < x_2 - x_1 < l_max
l_max = 0.2
l_min = -0.2
geo_const_bar = np.array([geo_constraint_flag,l_min,l_max,m1,m2])

## State space matrices
# x_dot = Ax + Bu
A = np.array([[0,0,1,0],[0,0,0,1],[-k12/m1,k12/m1,-(f1+c12)/m1,c12/m1],[k12/m2,-k12/m2,c12/m2,-(f2+c12)/m2]])
B = np.array([[0,0],[0,0],[1/m1,0],[0,1/m2]])

# Initial conditions
# x1, x2, x1_dot, x2_dot
x_initial = np.array([[0],[0],[0],[-10]])
x_initial = np.array([[0],[0],[0],[-2.2]])
x_initial = np.array([[0],[0],[0],[0]])
t_initial = 0
u_initial = np.array([[A1*math.sin(omega_1*t_initial)],[A2*math.sin(omega_2*t_initial)]])

##
t_final = 40
# t_final = 0.4 # use for troubleshooting

## Finding dt
tau1 = 1/math.sqrt(k12/m1)
tau2 = 1/math.sqrt(k12/m2)
tau = min(tau1,tau2)
dt_o = math.floor(tau/10*1000) / 1000 * 1
dt = dt_o
dt_prime_o = dt/5

convergence_error = 0.001 / 100
N = np.ceil(40/dt).astype(int)

## --------------------------------------------------------------------------
# Dual time stepping
n = 1
x_track, t_track, u_track, m_track, F_track = dualTimeStepping(A,B,x_initial,t_final,dt,n,convergence_error,k12,c12,u_params,geo_const_bar)

# Explicit
n = 2
x_track0, t_track0, u_track0, F_track0 = explicitSolution(A,B,x_initial,t_final,dt,n,convergence_error,k12,c12,u_params)

# Discrete - One Riemann Sum
alpha_A = 1
alpha_u = 0
x_track2, t_track2, u_track2, F_track2 = discreteSolutionORS(A,B,x_initial,t_final,dt,n,alpha_A,alpha_u,convergence_error,k12,c12,u_params)

# Discrete - SVD Inverse
alpha_u = 0
x_track3, t_track3, u_track3, F_track3 = discreteSolutionSVD(A,B,x_initial,t_final,dt,n,alpha_u,convergence_error,k12,c12,u_params)

# Discrete - Pade Approximation
alpha_u = 0
x_track4, t_track4, u_track4, F_track4 = discreteSolutionPade(A,B,x_initial,t_final,dt,n,alpha_u,convergence_error,k12,c12,u_params)

## Plotting that compares methods
fig, axs = plt.subplots(5, 1, figsize=(8, 5), sharex=True)

print('1st figure compares models')
axs[0].plot(t_track, x_track[0], label='Dual Time Stepping')
axs[0].plot(t_track0, x_track0[0], label='Explicit')
axs[0].plot(t_track2, x_track2[0], label='Discrete One Riemann Sum')
axs[0].plot(t_track3, x_track3[0], label='Discrete SVD Inverse',linestyle='-.')
# axs[0].plot(t_track3, x_track3[0], label='Discrete SVD Inverse')
axs[0].plot(t_track4, x_track4[0], label='Discrete Pade Approximation',linestyle='--')
axs[0].set_title(f'x1 vs time')
axs[0].set_ylabel('x1 (m)')
axs[0].set_xlabel('time (s)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_track, x_track[1], label='Dual Time Stepping')
axs[1].plot(t_track0, x_track0[1], label='Explicit')
axs[1].plot(t_track2, x_track2[1], label='Discrete One Riemann Sum')
axs[1].plot(t_track3, x_track3[1], label='Discrete SVD Inverse',linestyle='-.')
axs[1].plot(t_track4, x_track4[1], label='Discrete Pade Approximation',linestyle='--')
axs[1].set_title(f'x2 vs time')
axs[1].set_ylabel('x2 (m)')
axs[1].set_xlabel('time (s)')
# axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_track, x_track[2], label='Dual Time Stepping')
axs[2].plot(t_track0, x_track0[2], label='Explicit')
axs[2].plot(t_track2, x_track2[2], label='Discrete One Riemann Sum')
axs[2].plot(t_track3, x_track3[2], label='Discrete SVD Inverse',linestyle='-.')
axs[2].plot(t_track4, x_track4[2], label='Discrete Pade Approximation',linestyle='--')
axs[2].set_title(f'x1_dot vs time')
axs[2].set_ylabel('x1_dot (m/s)')
axs[2].set_xlabel('time (s)')
# axs[2].legend()
axs[2].grid(True)

axs[3].plot(t_track, x_track[3], label='Dual Time Stepping')
axs[3].plot(t_track0, x_track0[3], label='Explicit')
axs[3].plot(t_track2, x_track2[3], label='Discrete One Riemann Sum')
axs[3].plot(t_track3, x_track3[3], label='Discrete SVD Inverse',linestyle='-.')
axs[3].plot(t_track4, x_track4[3], label='Discrete Pade Approximation',linestyle='--')
axs[3].set_title(f'x2_dot vs time')
axs[3].set_ylabel('x2_dot (m/s)')
axs[3].set_xlabel('time (s)')
# axs[3].legend()
axs[3].grid(True)

axs[4].plot(t_track, x_track[1]-x_track[0], label='Dual Time Stepping')
axs[4].plot(t_track0, x_track0[1]-x_track0[0], label='Explicit')
axs[4].plot(t_track2, x_track2[1]-x_track2[0], label='Discrete One Riemann Sum')
axs[4].plot(t_track3, x_track3[1]-x_track3[0], label='Discrete SVD Inverse',linestyle='-.')
axs[4].plot(t_track4, x_track4[1]-x_track4[0], label='Discrete Pade Approximation',linestyle='--')
axs[4].set_title(f'x2-x1 vs time')
axs[4].set_ylabel('x2-x1 (m)')
axs[4].set_xlabel('time (s)')
# axs[4].legend()
axs[4].grid(True)

print('2nd plot shows number of dual time steps for dual time stepping')
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_track, m_track, label='Dual Time Stepping')
axs.set_title(f'# of pseudo-time steps')
axs.set_ylabel('# of pseudo-time steps')
axs.set_xlabel('time (s)')
axs.grid(True)

print('3rd plot compares force calculation on coupling')
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_track, F_track, label='Dual Time Stepping')
axs.plot(t_track0, F_track0, label='Explicit')
axs.plot(t_track2, F_track2, label='Discrete One Riemann Sum')
axs.plot(t_track3, F_track3, label='Discrete SVD Inverse',linestyle='-.')
axs.plot(t_track4, F_track4, label='Discrete Pade Approximation',linestyle='--')
axs.set_title(f'Force vs time')
axs.set_ylabel('Force (N)')
axs.set_xlabel('time (s)')
axs.set_xlim([-0.5,4])
axs.set_xlim([-0.1,5])
axs.legend()
axs.grid(True)

## --------------------------------------------------------------------------
## Looking at geo constraint
geo_constraint_flag = 0
x_initial = np.array([[0],[0],[0],[-10]])
A1 = 1750
u_params = np.array([A1, A2, omega_1, omega_2])
geo_const_bar = np.array([geo_constraint_flag,l_min,l_max,m1,m2])
dt = dt_o / 10
n = 1
x_track, t_track, u_track, m_track, F_track = dualTimeStepping(A,B,x_initial,t_final,dt,n,convergence_error,k12,c12,u_params,geo_const_bar)

geo_constraint_flag = 1
geo_const_bar = np.array([geo_constraint_flag,l_min,l_max,m1,m2])
x_trackConst, t_trackConst, u_trackConst, m_trackConst, F_trackConst = dualTimeStepping(A,B,x_initial,t_final,dt,n,convergence_error,k12,c12,u_params,geo_const_bar)

fig, axs = plt.subplots(5, 1, figsize=(8, 5), sharex=True)

print('4th plot looks at constrained and not constrained state vector')
axs[0].plot(t_track, x_track[0], color='blue', label='Not Constrained')
axs[0].plot(t_trackConst, x_trackConst[0], color=second_color, label='Constrained')
axs[0].set_title(f'x1 vs time')
axs[0].set_ylabel('x1 (m)')
axs[0].set_xlabel('time (s)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_track, x_track[1], color='blue', label='Not Constrained')
axs[1].plot(t_trackConst, x_trackConst[1], color=second_color, label='Constrained')
axs[1].set_title(f'x2 vs time')
axs[1].set_ylabel('x2 (m)')
axs[1].set_xlabel('time (s)')
# axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_track, x_track[2], color='blue', label='Not Constrained')
axs[2].plot(t_trackConst, x_trackConst[2], color=second_color, label='Constrained')
axs[2].set_title(f'x1_dot vs time')
axs[2].set_ylabel('x1_dot (m/s)')
axs[2].set_xlabel('time (s)')
# axs[2].legend()
axs[2].grid(True)

axs[3].plot(t_track, x_track[3], color='blue', label='Not Constrained')
axs[3].plot(t_trackConst, x_trackConst[3], color=second_color, label='Constrained')
axs[3].set_title(f'x2_dot vs time')
axs[3].set_ylabel('x2_dot (m/s)')
axs[3].set_xlabel('time (s)')
# axs[3].legend()
axs[3].grid(True)

axs[4].plot(t_track, x_track[1]-x_track[0], color='blue', label='Not Constrained')
axs[4].plot(t_trackConst, x_trackConst[1]-x_trackConst[0], color=second_color, label='Constrained')
axs[4].set_title(f'x2-x1 vs time')
axs[4].set_ylabel('x2-x1 (m)')
axs[4].set_xlabel('time (s)')
# axs[4].legend()
axs[4].grid(True)

fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)

print('5th plot focuses on length of coupling and speed only')
axs[0].plot(t_track, x_track[1]-x_track[0], color='blue', label='Not Constrained')
axs[0].plot(t_trackConst, x_trackConst[1]-x_trackConst[0], color=second_color, label='Constrained')
axs[0].set_title(f'x2-x1 vs time')
axs[0].set_ylabel('x2-x1 (m)')
axs[0].set_xlabel('time (s)')
# axs[0].set_ylim([-0.25,0.25])
# axs[0].set_xlim([-0.1,0.5])
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_track, x_track[2], color='blue', label='Not Constrained - m1 velocity')
axs[1].plot(t_track, x_track[3], color='blue', linestyle='--',label='Not Constrained - m2 velocity')
axs[1].plot(t_trackConst, x_trackConst[2], color=second_color, label='Constrained - m1 velocity')
axs[1].plot(t_trackConst, x_trackConst[3], color=second_color, linestyle='--', label='Constrained - m2 velocity')
axs[1].set_title(f'Velocity vs time')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].set_xlabel('time (s)')
# axs[1].set_ylim([-0.25,0.25])
# axs[1].set_xlim([-0.1,0.5])
axs[1].legend()
axs[1].grid(True)

# fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
# print('6th plot compares force between not constrained and contrained')
# axs.plot(t_track, F_track, color='blue', label='Not Constrained')
# axs.plot(t_trackConst, F_trackConst, color=second_color, label='Constrained')
# axs.set_title(f'Force vs time')
# axs.set_ylabel('Force (N)')
# axs.set_xlabel('time (s)')
# # axs.set_xlim([-0.5,4])
# # axs.set_xlim([-0.1,5])
# axs.legend()
# axs.grid(True)

## --------------------------------------------------------------------------
## Conservation of energy
# initial KE + initial PE + int of input E (not always positive) - int of loss E (always positive) = current KE + current PE (think about how to use dx or v*dt)
initial_KE = np.full(len(t_track), 1/2*m1*x_initial[2]**2 + 1/2*m2*x_initial[3]**2)
initial_PE = np.full(len(t_track), 1/2*k12*(x_initial[0]-x_initial[1])**2)
KE_bar = initial_KE.copy()
PE_bar = initial_PE.copy()
input_energy_bar = np.zeros_like(t_track)
friction_energy_bar = np.zeros_like(t_track)
damper_energy_bar = np.zeros_like(t_track)
for i in range(1,len(t_track)):
    t = t_track[i]
    t_prev = t_track[i-1]
    dt = t_track[i] - t_track[i-1]
    dx1 = x_track[0][i]-x_track[0][i-1]
    dx2 = x_track[1][i]-x_track[1][i-1]

    KE_bar[i] = 1/2*m1*x_track[2][i]**2 + 1/2*m2*x_track[3][i]**2
    PE_bar[i] = 1/2*k12*(x_track[0][i]-x_track[1][i])**2

    # One Riemann sum from left
    # input_energy_bar[i] = input_energy_bar[i-1] + u_track[0][i-1]*x_track[2][i-1]*dt + u_track[1][i-1]*x_track[3][i-1]*dt
    # friction_energy_bar[i] = friction_energy_bar[i-1] + f1*x_track[2][i-1]**2*dt + f2*x_track[3][i-1]**2*dt
    # damper_energy_bar[i] = damper_energy_bar[i-1] + c12*(x_track[2][i-1]-x_track[3][i-1])**2*dt

    # One Riemann sum from right
    # input_energy_bar[i] = input_energy_bar[i-1] + u_track[0][i]*x_track[2][i]*dt + u_track[1][i]*x_track[3][i]*dt
    # friction_energy_bar[i] = friction_energy_bar[i-1] + f1*x_track[2][i]**2*dt + f2*x_track[3][i]**2*dt
    # damper_energy_bar[i] = damper_energy_bar[i-1] + c12*(x_track[2][i]-x_track[3][i])**2*dt

    # Average of left and right sums
    input_energy_bar[i] = input_energy_bar[i-1] + 1/2 * ((u_track[0][i-1]*x_track[2][i-1]*dt + u_track[1][i-1]*x_track[3][i-1]*dt) + (u_track[0][i]*x_track[2][i]*dt + u_track[1][i]*x_track[3][i]*dt))
    friction_energy_bar[i] = friction_energy_bar[i-1] + 1/2 * ((f1*x_track[2][i-1]**2*dt + f2*x_track[3][i-1]**2*dt) + (f1*x_track[2][i]**2*dt + f2*x_track[3][i]**2*dt))
    damper_energy_bar[i] = damper_energy_bar[i-1] + 1/2 * ((c12*(x_track[2][i-1]-x_track[3][i-1])**2*dt) + (c12*(x_track[2][i]-x_track[3][i])**2*dt))
    

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
print('6th plot is conservation of energy for not constrained scenario - error in calculating total energy')
axs.plot(t_track, ((initial_KE + initial_PE + input_energy_bar - friction_energy_bar - damper_energy_bar) - (KE_bar + PE_bar))/(initial_KE + initial_PE + input_energy_bar), label='Total Energy Error (%)')
axs.set_title(f'Conservation of Energy')
axs.set_ylabel('Energy Error (%)')
axs.set_xlabel('Time (s)')
axs.legend()

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
print('7th plot is conservation of energy for not constrained scenario - total energy')
axs.plot(t_track, ((initial_KE + initial_PE + input_energy_bar - friction_energy_bar - damper_energy_bar) - (KE_bar + PE_bar)), label='Total Energy Error')
axs.set_title(f'Conservation of Energy')
axs.set_ylabel('Energy Error')
axs.set_xlabel('Time (s)')
axs.legend()

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
print('8th plot is conservation of energy for not constrained scenario - KE, PE, damper energy, etc.')
axs.plot(t_track, input_energy_bar, label='Input energy')
axs.plot(t_track, friction_energy_bar, label='Friction energy')
axs.plot(t_track, damper_energy_bar, label='Damper energy')
axs.plot(t_track, KE_bar, label='KE')
axs.plot(t_track, PE_bar, label='PE')
axs.plot(t_track, ((initial_KE + initial_PE + input_energy_bar - friction_energy_bar - damper_energy_bar) - (KE_bar + PE_bar)), label='Total Energy Error')
axs.set_title(f'Conservation of Energy')
axs.set_ylabel('Energy (N-m)')
axs.set_xlabel('Time (s)')
axs.legend(loc='upper right')

## --------------------------------------------------------------------------
v_input_bar = np.array([1,2,3,4,5,6,7,8,9,10])
geo_constraint_flag = 0
geo_const_bar = np.array([geo_constraint_flag,l_min,l_max,m1,m2])
A1 = 0
u_params = np.array([A1, A2, omega_1, omega_2])
max_F_bar = np.zeros_like(v_input_bar)
damper_energy_dissipated_bar = np.zeros_like(v_input_bar)
dt = dt_o / 2
for i in range(0,len(v_input_bar)):
    x_initial = np.array([[0],[0],[0],[-v_input_bar[i]]])

    x_track, t_track, u_track, m_track, F_track = dualTimeStepping(A,B,x_initial,t_final,dt,n,convergence_error,k12,c12,u_params,geo_const_bar)
    max_F_bar[i] = max(abs(F_track))

    # initial_KE = np.full(len(t_track), 1/2*m1*x_initial[2]**2 + 1/2*m2*x_initial[3]**2)
    # initial_PE = np.full(len(t_track), 1/2*k12*(x_initial[0]-x_initial[1])**2)
    # KE_bar = initial_KE.copy()
    # PE_bar = initial_PE.copy()
    # input_energy_bar = np.zeros_like(t_track)
    # friction_energy_bar = np.zeros_like(t_track)
    damper_energy_bar = np.zeros_like(t_track)
    for x in range(1,len(t_track)):
        # t = t_track[i]
        # t_prev = t_track[i-1]
        dt = t_track[x] - t_track[x-1]
        # dx1 = x_track[0][i]-x_track[0][i-1]
        # dx2 = x_track[1][i]-x_track[1][i-1]

        # KE_bar[i] = 1/2*m1*x_track[2][i]**2 + 1/2*m2*x_track[3][i]**2
        # PE_bar[i] = 1/2*k12*(x_track[0][i]-x_track[1][i])**2

        # One Riemann sum from left
        # input_energy_bar[i] = input_energy_bar[i-1] + u_track[0][i-1]*x_track[2][i-1]*dt + u_track[1][i-1]*x_track[3][i-1]*dt
        # friction_energy_bar[i] = friction_energy_bar[i-1] + f1*x_track[2][i-1]**2*dt + f2*x_track[3][i-1]**2*dt
        # damper_energy_bar[i] = damper_energy_bar[i-1] + c12*(x_track[2][i-1]-x_track[3][i-1])**2*dt

        # One Riemann sum from right
        # input_energy_bar[i] = input_energy_bar[i-1] + u_track[0][i]*x_track[2][i]*dt + u_track[1][i]*x_track[3][i]*dt
        # friction_energy_bar[i] = friction_energy_bar[i-1] + f1*x_track[2][i]**2*dt + f2*x_track[3][i]**2*dt
        # damper_energy_bar[i] = damper_energy_bar[i-1] + c12*(x_track[2][i]-x_track[3][i])**2*dt

        # Average of left and right sums
        # input_energy_bar[i] = input_energy_bar[i-1] + 1/2 * ((u_track[0][i-1]*x_track[2][i-1]*dt + u_track[1][i-1]*x_track[3][i-1]*dt) + (u_track[0][i]*x_track[2][i]*dt + u_track[1][i]*x_track[3][i]*dt))
        # friction_energy_bar[i] = friction_energy_bar[i-1] + 1/2 * ((f1*x_track[2][i-1]**2*dt + f2*x_track[3][i-1]**2*dt) + (f1*x_track[2][i]**2*dt + f2*x_track[3][i]**2*dt))
        damper_energy_bar[x] = damper_energy_bar[x-1] + 1/2 * ((c12*(x_track[2][x-1]-x_track[3][x-1])**2*dt) + (c12*(x_track[2][x]-x_track[3][x])**2*dt))
    
    damper_energy_dissipated_bar[i] = damper_energy_bar[-1]

fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)

print('8th plot shows force and damper energy dissipated vs speed')
axs[0].plot(v_input_bar, max_F_bar, color='blue', label='Force')
axs[0].set_title(f'Force vs Initial Speed')
axs[0].set_ylabel('Force (N)')
axs[0].set_xlabel('Speed (m/s)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(v_input_bar, damper_energy_dissipated_bar, color='blue', label='Energy')
axs[1].set_title(f'Dissipated Damper Energy vs Initial Speed')
axs[1].set_ylabel('Dissipated Damper Energy (J)')
axs[1].set_xlabel('Speed (m/s)')
axs[1].legend()
axs[1].grid(True)

plt.show()