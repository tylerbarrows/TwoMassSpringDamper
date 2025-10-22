##
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import expm


## Input Force
def uInputCalc(u_params,t):
    A1 = u_params[0]
    A2 = u_params[1]
    omega_1 = u_params[2]
    omega_2 = u_params[3]

    u = np.array([[A1*math.sin(omega_1*t)],[A2*math.sin(omega_2*t)]])

    return u

## Dual time stepping
def dualTimeStepping(A,B,x_initial,t_final,dto,n,convergence_error,k,c,u_params,geo_const_bar):
    t = 0
    dt = dto / n
    dt_prime_o = dt/5
    N = np.ceil(t_final/dt).astype(int)

    # geo constraints
    geo_constraint_flag = geo_const_bar[0]
    l_min = geo_const_bar[1]
    l_max = geo_const_bar[2]
    m1 = geo_const_bar[3]
    m2 = geo_const_bar[4]

    x_track = x_initial
    t_track = np.array([0])
    u_initial = uInputCalc(u_params,t)
    u_track = u_initial
    m_track = np.array([0])
    F_track = np.array([0])

    for i in range(1,N+1):
        x_o = x_track[:,[-1]]
        t_o = t_track[-1]
        t = t_o + dt

        u = uInputCalc(u_params,t)

        dx = np.matmul(A,x_o) * dt + np.matmul(B,u) * dt
        x = x_o + np.matmul(A,x_o) * dt + np.matmul(B,u) * dt

        dt_prime = dt_prime_o
        for m in range(1,N+1):
            dx_prev = dx.copy()
            dx = - dt_prime * (1/dt * (x-x_o) - np.matmul(A,x) - np.matmul(B,u))
            x = x - dt_prime * (1/dt * (x-x_o) - np.matmul(A,x) - np.matmul(B,u))

            if min ( abs ( 1/dt*(x-x_o) - (np.matmul(A,x) + np.matmul(B,u)) ) <=  abs (np.matmul(A,x) + np.matmul(B,u)) * convergence_error ) == True :
                break

            if min (np.sign(dx) == np.sign(dx_prev)) == False and m > 1:
                dt_prime = 0.9 * dt_prime

        ## adding geometry constraints here
        x1 = x[0]
        x2 = x[1]
        collisionFlag = 0
        if (x2 - x1 > l_max or x2 - x1 < l_min) and geo_constraint_flag == 1:
            collisionFlag = 1
            dt_max = dt
            dt_min = dt / 10
            for n in range(1,N+1):
                dt_con = 1/2*(dt_min+dt_max)

                t = t_o + dt_con
                u = uInputCalc(u_params,t)
                
                dx = np.matmul(A,x_o) * dt_con + np.matmul(B,u) * dt_con
                x = x_o + np.matmul(A,x_o) * dt_con + np.matmul(B,u) * dt_con

                ##
                dt_prime = dt_prime_o
                for l in range(1,N+1):
                    dx_prev = dx.copy()
                    dx = - dt_prime * (1/dt_con * (x-x_o) - np.matmul(A,x) - np.matmul(B,u))
                    x = x - dt_prime * (1/dt_con * (x-x_o) - np.matmul(A,x) - np.matmul(B,u))

                    if min ( abs ( 1/dt_con*(x-x_o) - (np.matmul(A,x) + np.matmul(B,u)) ) <=  abs (np.matmul(A,x) + np.matmul(B,u)) * convergence_error ) == True :
                        break

                    if min (np.sign(dx) == np.sign(dx_prev)) == False and l > 1:
                        dt_prime = 0.9 * dt_prime
                

                if abs(((x2-x1)-l_max)/l_max) < convergence_error or abs(((x2-x1)-l_min)/l_min) < convergence_error:
                    break
                
                x1 = x[0]
                x2 = x[1]
                if x2 - x1 > l_max or x2 - x1 < l_min:
                    dt_max = dt_con
                    dt_min = dt_min
                else :
                    dt_max = dt_max
                    dt_min = dt_con
            
            u1_prime = x[2]
            u2_prime = x[3]

            u1 = (m1/m2-1)/(m1/m2+1)*u1_prime + 2/(m1/m2+1)*u2_prime
            u2 = (m2/m1-1)/(m2/m1+1)*u2_prime + 2/(m2/m1+1)*u1_prime

            x_prev_collision = x.copy()

            x[2] = u1
            x[3] = u2
            
        if collisionFlag == 0:
            x_track = np.hstack((x_track,x))
            t_track = np.hstack((t_track,t_track[-1]+dt))
            u_track = np.hstack((u_track,u))
            m_track = np.hstack((m_track,m))
            F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))
        else:
            x_track = np.hstack((x_track,x_prev_collision))
            t_track = np.hstack((t_track,t_track[-1]+dt_con))
            # print('dt_con is ', dt_con)
            u_track = np.hstack((u_track,u))
            m_track = np.hstack((m_track,m))
            F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))

            x_track = np.hstack((x_track,x))
            t_track = np.hstack((t_track,t_track[-1]))
            u_track = np.hstack((u_track,u))
            m_track = np.hstack((m_track,m))
            F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))

    return x_track, t_track, u_track, m_track, F_track

## Explicit
def explicitSolution(A,B,x_initial,t_final,dto,n,convergence_error,k,c,u_params):
    t = 0
    dt = dto / n
    N = np.ceil(t_final/dt).astype(int)

    x_track = x_initial
    t_track = np.array([0])
    u_initial = uInputCalc(u_params,t)
    u_track = u_initial
    F_track = np.array([0])

    for i in range(1,N+1):
        x_o = x_track[:,[-1]]
        t = t + dt

        u = uInputCalc(u_params,t)

        x = x_o + np.matmul(A,x_o) * dt + np.matmul(B,u) * dt
            
        x_track = np.hstack((x_track,x))
        t_track = np.hstack((t_track,t_track[-1]+dt))
        u_track = np.hstack((u_track,u))
        F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))
        
    return x_track, t_track, u_track, F_track

## Discrete ORS
def discreteSolutionORS(A,B,x_initial,t_final,dto,n,alpha_A,alpha_u,convergence_error,k,c,u_params):
    t = 0
    dt = dto / n
    N = np.ceil(t_final/dt).astype(int)

    A_matrix_exp = expm(A*dt)
    A_matrix_exp_intBu = expm(A*alpha_A*dt)
    I = np.eye(*A.shape)

    x_track = x_initial
    t_track = np.array([0])
    u_initial = uInputCalc(u_params,t)
    u_track = u_initial
    F_track = np.array([0])

    for i in range(1,N+1):
        x_o = x_track[:,[-1]]
        
        u = uInputCalc(u_params,t)
        u_next = uInputCalc(u_params,t+dt)
        u = u * (1 - alpha_u) + u_next * alpha_u
        
        # x = np.matmul(A_matrix_exp,x_o) + np.matmul(A_inv,np.matmul((A_matrix_exp-I),np.matmul(B,u)))
        # x = np.matmul(A_matrix_exp,x_o) + np.matmul(np.matmul(np.matmul(A_inv,A_matrix_exp-I),B),u)
        x = np.matmul(A_matrix_exp,x_o) + np.matmul(np.matmul(A_matrix_exp_intBu,B*dt),u)
        # x = np.matmul(A_matrix_exp,x_o) + np.matmul(B*dt,u)

        t = t + dt

        x_track = np.hstack((x_track,x))
        t_track = np.hstack((t_track,t_track[-1]+dt))
        u_track = np.hstack((u_track,u))
        F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))

    return x_track, t_track, u_track, F_track

## Discrete SVD
def discreteSolutionSVD(A,B,x_initial,t_final,dto,n,alpha_u,convergence_error,k,c,u_params):
    t = 0
    dt = dto / n
    N = np.ceil(t_final/dt).astype(int)

    U, s, Vh = np.linalg.svd(A)
    s_diag = np.diag(s)
    U_inv = U.T
    tol = 1e-10
    s_inv = np.array([1/val if val > tol else 0 for val in s])
    s_diag_inv = np.diag(s_inv)
    Vh_inv = Vh.T
    A_svd_inv = np.matmul(np.matmul(Vh_inv,s_diag_inv),U_inv)
    A_matrix_exp = expm(A*dt)
    I = np.eye(*A.shape)

    x_track = x_initial
    t_track = np.array([0])
    u_initial = uInputCalc(u_params,t)
    u_track = u_initial
    F_track = np.array([0])

    for i in range(1,N+1):
        x_o = x_track[:,[-1]]
        
        u = uInputCalc(u_params,t)
        u_next = uInputCalc(u_params,t+dt)
        u = u * (1 - alpha_u) + u_next * alpha_u
        
        x = np.matmul(A_matrix_exp,x_o) + np.matmul(A_svd_inv,np.matmul((A_matrix_exp-I),np.matmul(B,u)))

        t = t + dt

        x_track = np.hstack((x_track,x))
        t_track = np.hstack((t_track,t_track[-1]+dt))
        u_track = np.hstack((u_track,u))
        F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))

    return x_track, t_track, u_track, F_track

## Discrete - Pade Approximation
def discreteSolutionPade(A,B,x_initial,t_final,dto,n,alpha_u,convergence_error,k,c,u_params):
    t = 0
    dt = dto / n
    N = np.ceil(t_final/dt).astype(int)

    v = np.array([[1],[1/2],[1/12],[-1/2],[1/12]])
    b0 = 1
    a0 = v[0][0]
    a1 = v[1][0]
    a2 = v[2][0]
    b1 = v[3][0]
    b2 = v[4][0]
    m0 = dt/1
    m1 = dt**2/math.factorial(2)
    m2 = dt**3/math.factorial(3)
    m3 = dt**4/math.factorial(4)
    m4 = dt**5/math.factorial(5)
    g = np.array([[m0],[m1],[m2],[m3],[m4]])
    dF_dx = np.array([[1,0,0,0,0],[-b1,1,0,-a0,0],[b1**2-b2,-b1,1,2*a0*b1-a1,-a0],[-b1**3+2*b1*b2,b1**2-b2,-b1,-3*a0*b1**2+2*a0*b2+2*a1*b1-a2,2*a0*b1-a1],[b1**4-3*b1**2*b2+b2**2,-b1**3+2*b1*b2,b1**2-b2,4*a0*b1**3-6*a0*b1*b2-3*a1*b1**2+2*a1*b2+2*a2*b1,-3*a0*b1**2+2*a0*b2+2*a1*b1-a2]])
    F = np.array([[a0],[-a0*b1+a1],[a0*(b1**2-b2)-a1*b1+a2],[a0*(-b1**3+2*b1*b2)+a1*(b1**2-b2)-a2*b1],[a0*(b1**4-3*b1**2*b2+b2**2)+a1*(-b1**3+2*b1*b2)+a2*(b1**2-b2)]])
    Func_track = F
    int_track = np.array([0])
    dv_mag = 1
    for i in range(1,10000):
        sign_Fg_prev = np.sign(F-g)
        dF_dx_inv = np.linalg.inv(dF_dx)
        dv = np.matmul(dF_dx_inv,g-F)
        dv_norm = np.linalg.norm(dv)
        dv = dv * 1/dv_norm
        dv = dv * dv_mag
        v = v + dv

        a0 = v[0][0]
        a1 = v[1][0]
        a2 = v[2][0]
        b1 = v[3][0]
        b2 = v[4][0]

        dF_dx = np.array([[1,0,0,0,0],[-b1,1,0,-a0,0],[b1**2-b2,-b1,1,2*a0*b1-a1,-a0],[-b1**3+2*b1*b2,b1**2-b2,-b1,-3*a0*b1**2+2*a0*b2+2*a1*b1-a2,2*a0*b1-a1],[b1**4-3*b1**2*b2+b2**2,-b1**3+2*b1*b2,b1**2-b2,4*a0*b1**3-6*a0*b1*b2-3*a1*b1**2+2*a1*b2+2*a2*b1,-3*a0*b1**2+2*a0*b2+2*a1*b1-a2]])
        F = np.array([[a0],[-a0*b1+a1],[a0*(b1**2-b2)-a1*b1+a2],[a0*(-b1**3+2*b1*b2)+a1*(b1**2-b2)-a2*b1],[a0*(b1**4-3*b1**2*b2+b2**2)+a1*(-b1**3+2*b1*b2)+a2*(b1**2-b2)]])

        sign_Fg = np.sign(F-g)
        if i > 100 and min(sign_Fg == sign_Fg_prev) == False:
            dv_mag = dv_mag * 0.9

        if max(abs((F-g)/g)) < convergence_error:
            break

        Func_track = np.hstack((Func_track,F))
        int_track = np.hstack((int_track,i))
    
    A_matrix_exp = expm(A*dt)
    I = np.eye(*A.shape)
    A_mod_pade = np.matmul(a0*I+a1*A+a2*np.matmul(A,A),np.linalg.inv(b0*I+b1*A+b2*np.matmul(A,A)))

    x_track = x_initial
    t_track = np.array([0])
    u_initial = uInputCalc(u_params,t)
    u_track = u_initial
    F_track = np.array([0])

    for i in range(1,N+1):
        x_o = x_track[:,[-1]]
        
        u = uInputCalc(u_params,t)
        u_next = uInputCalc(u_params,t+dt)
        u = u * (1 - alpha_u) + u_next * alpha_u
        
        x = np.matmul(A_matrix_exp,x_o) + np.matmul(np.matmul(A_mod_pade,B),u)

        t = t + dt

        x_track = np.hstack((x_track,x))
        t_track = np.hstack((t_track,t_track[-1]+dt))
        u_track = np.hstack((u_track,u))
        F_track = np.hstack((F_track,k*(x[1]-x[0])+c*(x[3]-x[2])))


    return x_track, t_track, u_track, F_track