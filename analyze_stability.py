# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:48:22 2023

@author: junhyeok
"""
import cvxpy as cp
import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt
import logging

def cal_Nvariable(scenario):
    
    if scenario == 'droop':
        Nval = 4
    elif scenario == 'droop1':
        Nval = 3
    elif scenario == 'secondPI':
        Nval = 4
    elif scenario == 'nudge':
        Nval = 5
    else:
        Nval = 6
        
    return Nval
def generate_Amat(gen,con, case, scenario):
    
    Ngen = case['Ngen']
    Nval = case['Nval']
    Pgen, droop = gen['Pgen'], gen['droop']
    Tv, Te, H, D = gen['Tv'],gen['Te'],gen['H'], gen['D']
    kp1, ki1 = con['kp1'], con['ki1']
    ki2, kc1, Tl1 =  con['ki2'],  con['kc1'],  con['Tl1']
    # frequency + Ngen * gen_state(Nval)
    Nmat = 1 + Ngen*Nval 
    
    Tm = 1/(Tv*Te)
    M = 2* sum(H)
    #M = 8.174
    A = np.zeros([Nmat,Nmat])
    A[0,0] = -D/M
    
    for i in range(Ngen):
        
        idx = 1 +i*Nval # ith Gen index
        A[0, idx] = Pgen[i]/M #Swing Equation
        A[idx ,idx +1] = 1 # Pdot
        A[idx +1, idx +2] = 1 # Pdotdot
        A[idx +2, 0] = (-ki1+ D * kp1 / M) * Tm[i] #Frequency Fractional
        
        for j in range(Ngen):
            idy = 1 + j*Nval
            A[idx +2, idy] = -kp1 * Pgen[j] /M * Tm[i]
        
        A[idx +2, idx +1] = (-1-kp1*droop)* Tm[i]
        A[idx +2, idx +2] = -(Tv[i] + Te[i])*Tm[i]
        
        if scenario == 'droop':
            A[idx+2, idx+3] = ki1 * Tm[i]
            A[idx+3, idx+1] = -droop
        elif scenario == 'droop1':
            A[idx+2, idx] = - ki1 * droop * Tm[i]
            
        elif scenario == 'secondPI':
            A[idx+2, idx+3] = (ki1 - droop*kp1*ki2) * Tm[i]
            A[idx+3, idx+1] = -droop
            A[idx+3, idx+3] = -droop*ki2
        elif scenario == 'nudge':
            A[idx+2, idx+3] = (ki1 - droop*kp1*ki2) * Tm[i]
            A[idx+2, idx+4] = droop*ki2*kp1* Tm[i]
            
            A[idx+3, idx+1] = -droop
            A[idx+3, idx+3] = -droop*ki2
            A[idx+3, idx+4] = droop*ki2
        
            if Nval == 5:
                A[idx +4, idx ] = -kc1/Tl1
                A[idx +4, idx +4] = -1/Tl1 
                
            elif Nval ==6:
                
                A[idx +4, idx +5] = 1
                # Need to Check
                A[idx +5, idx +1] = -kc1/Tl1 
                A[idx +5, idx +5] = -1/Tl1          
                 
    return A

def generate_Bmat(gen,con, case):
    
    Ngen,Nval = case['Ngen'], case['Nval']
    
    Nmat = 1 + Ngen*Nval
    Tv, Te, H = gen['Tv'],gen['Te'],gen['H']
    kp1, ki1 = con['kp1'], con['ki1']
    
    Tm = 1/(Tv*Te)
    
    M = 2* sum(H)
    
    B1 = np.zeros([Nmat,1]) # Load Disturbance
    B2 = np.zeros([Nmat, Ngen]) # Measurement Error
    
    B1[0,0] = -1/M
    
    for i in range(Ngen):
        idx = 1+ i*Nval
        B1[idx+2,0] = kp1 * Tm[i]/M
        B2[idx+2,i] = -ki1* Tm[i]

    B = np.c_[B1,B2]
    return B

def run_timeseries(gen, con, case, scenario):
    
    start = time.time()
    Sbase, Pgen, droop = gen['Sbase'], gen['Pgen'], gen['droop']
    Tv, Te, H, D = gen['Tv'],gen['Te'],gen['H'], gen['D']
    kp1, ki1 = con['kp1'], con['ki1']
    ki2, kc1, Tl1 =  con['ki2'],  con['kc1'],  con['Tl1']
    
    Ngen, Nval = case['Ngen'], case['Nval']
    Ttotal, Ts = case['Ttotal'], case['Ts']
    error_mat = case['error']
    load_pattern = case['load_pattern']
    
    Tsamp = case['Tsamp']
    

    Nmat = 1+Ngen*Nval
    Nsample = int(Ttotal/Ts)
    Ndist = int(20*1/Ts) #int(0.1 * Nsample) 

    X_dot = np.zeros([Nmat,1])
    del_X = np.zeros([Nmat, Nsample+1])
    U1 = np.zeros([1, Nsample+1]) # Load Disturbances
    U2 = np.zeros([Ngen, Nsample+1]) # Measurement Noises
    
    count = 0
    Samp_trig = 0
    Sampling_count = Tsamp/Ts
    trig = 0
    count1 = 0
    trig_time = np.zeros([Nsample + 1])
    
    del_X[:,0:1] = X_dot*Ts
    #initial operating point
    Winit = -0.002
    del_X[0,0] = Winit
       
    
    A = generate_Amat(gen, con, case, 'droop')
    B = generate_Bmat(gen, con, case)

    isProposed = scenario == 'nudge'    
    isDroop = scenario == 'droop'
    
    sumPg = 0
    for i in range(Ngen):
        idx = 1 + i*Nval
        Wrefk = Winit + error_mat[i]
        Pgk = - Wrefk/droop * Pgen[i]
        del_X[idx,0] = Pgk
        del_X[idx+3,0] = Wrefk 
        sumPg += Pgk
    
    
    PLinit = - Winit*D + sumPg


    for i in range(Nsample):
        
        # Switch from Droop to the secPI
        if not isDroop:
            if i == Ndist+1:              
                A = generate_Amat(gen, con, case, 'secondPI')
                #print(f'A matrix change from droop to {scenario}')
       
        if isProposed:
            # Sampling Count Update
            count = count + 1
            if (count > Sampling_count):
                count = 0
                if(abs(X_dot[0]*1/Tsamp) < 0.3):
                    trig = trig + 1
                    count1 = count1 + 1
                else:
                    trig = 0
                    
            if i > Ndist+1:
                if trig <= 5:
                    A = generate_Amat(gen, con, case, 'secondPI')
                else:
                    A = generate_Amat(gen, con, case, 'nudge')

            
        # Disturbance Update
        U2[:,i] = error_mat[:Ngen]
        sumPgen = sum(Pgen)
        
        if i <= 50*1/Ts:
            U1[0,i] = PLinit
            
        elif i>50*1/Ts and i<100*1/Ts:
            U1[0,i] = load_pattern[0] / Sbase /sumPgen
            
        elif i>100*1/Ts and i<140*1/Ts:
            U1[0,i] = load_pattern[1] / Sbase /sumPgen 
    
    
        elif i>140*1/Ts and i<170*1/Ts:
            U1[0,i] = load_pattern[2] / Sbase /sumPgen
    
        else:
            U1[0,i] = load_pattern[3] / Sbase /sumPgen 
    
        X_dot = np.dot(A, del_X[:,i]) + np.dot(B[:,0:1], U1[:,i]) + np.dot(B[:,1:1+Ngen], U2[:,i])
        del_X[:,i+1] = del_X[:,i]+ Ts*X_dot;  
    
    end = time.time()
    #print('Simulation Time:', end - start, 'secs')
    U = np.concatenate((U1.T,U2.T),1).T
    return del_X, U

def check_symmetric(matrix):
    n = matrix.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                count += 1
                print(f"({i}, {j}) and ({j}, {i}) entries are not equal: {matrix[i][j]} != {matrix[j][i]}")
    if count == 0:
        print("Matrix is symmetric")  # optional, can remove if not needed

def draw_graph(Y, gen, case, scenario):
    
    start = time.time()
    
    Sbase = gen['Sbase']
    Ngen = case['Ngen']
    Nval = case['Nval']
    Ts = case['Ts']
    
    color_list = ['red', 'green', 'blue', 'purple']
    
    X = np.arange(Y.shape[1])*Ts
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    print("Draw the Frequency Plot")
    ax1.plot(X, Y[0,:]+1, label='Wpu', color='black', alpha = 0.7)
    if not scenario == 'droop':
        print("Draw the Wref Plot")
        for i in range(Ngen):
            idx = 1+i*Nval
            ax1.plot(X, Y[idx+3,:]+1, label=f'Wref{i+1}', color=color_list[i])
            
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [p.u.]')
    ax1.set_ylim([0.985, 1.01])
    ax1.grid(linewidth = 0.5, alpha = 0.5, linestyle= '--')
    ax1.legend(loc='best', fontsize = 10)     
    ax1.set_title('Frequency Data')
    
    print("Draw the Active Power Plot")
    for i in range(Ngen):
        idx = 1 +i*Nval
        ax2.plot(X, Y[idx,:]*Sbase+100, label= f'Pgen{i+1}', color = color_list[i])
    
    ax2.set_ylim([80, 140])
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Active Power [p.u.]')
    ax2.grid(linewidth = 0.5, alpha = 0.5, linestyle= '--')
    ax2.legend(loc='best', fontsize = 10)
    ax2.set_title('Active Power Data')
    plt.show()
    
    end = time.time()
    print('Draw Graph Time:', end - start, 'secs')
    
def draw_lyapunov(matrix_set, case):
    
    print("Print_lyapunov_function for timeseries")
    X = matrix_set['X']
    P = matrix_set['P']
    A_droop = matrix_set['A_droop']
    A_mat = matrix_set['A']
    
    B = matrix_set['B']
    U = matrix_set['U']
    
    Ts = case['Ts']
    input_flag = case['input_flag']

    nSample = X.shape[1]-1
    x = np.arange(nSample)*Ts 
    V = np.zeros(nSample) 
    V_dot = np.zeros(nSample)
    V_input_dot = np.zeros(nSample)
    Ndist = int(nSample/10)
    
    A = A_droop
    for i in range(nSample):
        
        if i == Ndist+1:
            print("Change A matrix from droop to the other set")
            A = A_mat
    
        tempV = np.dot(X[:,i].T, P)
        V[i] = np.dot(tempV, X[:,i])
        
        temp_AP = np.dot(A.T, P)
        temp_PA = np.dot(P, A)
        temp_APPA = temp_AP + temp_PA
        
        temp_Vdot = np.dot(X[:,i].T, temp_APPA)
        
        V_dot[i] = np.dot(temp_Vdot, X[:,i]) 
        
        temp_PB = np.dot(P, B)
        temp_xPB = np.dot(X[:,i].T, temp_PB)
        V_input_dot[i] = 2*np.dot(temp_xPB, U[:,i])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    print("Draw Lyapunov function")
    ax1.plot(x, V, label='V(x)', color='black', alpha = 0.7)
            
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('V(x)')
    ax1.grid(linewidth = 0.5, alpha = 0.5, linestyle= '--')
    ax1.legend(loc='best', fontsize = 10)     
    ax1.set_title('Lyapunov function')
    
    print("Draw the derivative of lyapunov function")
    if input_flag:
        print("Consider the input matrix")
        ax2.plot(x, V_dot + V_input_dot, label= 'Vdot', color = 'black')

    else:
        ax2.plot(x, V_dot, label= 'Vdot', color = 'black')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Vdot')
    ax2.grid(linewidth = 0.5, alpha = 0.5, linestyle= '--')
    ax2.legend(loc='best', fontsize = 10)
    ax2.set_title('Vdot')
    
    plt.show()
    
    return V, V_dot, V_input_dot

def lyapunov(param):
    
    Ts = 0.5*10**-4
    wpu = 1.0;
    Wmax = 1.02;
    
    # Generator Parameter
    Pgen = np.array([1, 1, 1])
    droop = 0.05
    Tv = np.array([0.5, 1.0, 1.0])
    Te = np.array([0.05, 0.1, 0.1])
    H = np.array([1.587, 2.5, 2.5])
    D = 0.01
    gen_dict = {'Sbase':150, 'Pgen':Pgen, 'droop':droop,
                'Tv':Tv, 'Te':Te, 'H':H, 'D':D}
    
    if len(param) == 1:
        param = [param]
        param.append(2.4)
        
    # Control Parameter
    con_dict = {'kp1': 8,
                'ki1': 2.5,
                'ki2': 5,
                'kc1' : param[0],
                'Tl1' : param[1]
               }

    Ts = 0.5*10**-4
    error_mat = np.array([0.0005, 0, -0.0005])
    
    scenario = 'nudge'
    #scenario = 'droop'
    load_pattern = [90,110,110, 110]
    case_dict = {'Ngen':3, 'Nval': cal_Nvariable(scenario), 'Ts': 0.5*10**-4, 'Ttotal': 200,
                'wpu':1.0, 'Wmax': 1.02, 'error': error_mat, 'input_flag': True, 'load_pattern': load_pattern}
    
    A = generate_Amat(gen_dict, con_dict, case_dict, 'nudge')
    B = generate_Bmat(gen_dict, con_dict, case_dict)
    
    # Optimize
    n = A.shape[0]

    # Define the LMI variables
    P = cp.Variable((n,n), symmetric = True)
    gamma = 1e-6
    G = np.eye(n)* gamma
    # Define the constraints for the LMI
    
    constraints = [P >> G, A.T @ P + P@A << 0 ]
    
    # Define the objective for the LMI optimization problem
    obj = cp.Minimize(0)
    #obj = cp.Minimize(cp.trace(P))
    
    # Solve the LMI optimization problem
    prob = cp.Problem(obj, constraints)
    
    # Create a SolverOptions object and set the tolerance to 1e-6
    # Solve the problem with the specified options
    #prob.solve(solver=cp.MOSEK, verbose=True) #, options=options)
    prob.solve(solver=cp.MOSEK)
    
    cost = np.trace(P.value)
    
    return cost

def common_lyapunov(param):
    
    Ts = 0.5*10**-4
    wpu = 1.0;
    Wmax = 1.02;
    
    # Generator Parameter
    Pgen = np.array([1, 1, 1])
    droop = 0.05
    Tv = np.array([0.5, 1.0, 1.0])
    Te = np.array([0.05, 0.1, 0.1])
    H = np.array([1.587, 2.5, 2.5])
    D = 0.01
    gen_dict = {'Sbase':150, 'Pgen':Pgen, 'droop':droop,
                'Tv':Tv, 'Te':Te, 'H':H, 'D':D}
    
    if len(param) == 1:
        param = [param]
        param.append(1.76)
    # Control Parameter
    con_dict = {'kp1': 8,
                'ki1': 2.5,
                'ki2': 5,
                'kc1' : param[0],
                'Tl1' : param[1]
               }

    Ts = 0.5*10**-4
    error_mat = np.array([0.0005, 0, -0.0005])
    
    scenario = 'nudge'
    #scenario = 'droop'
    load_pattern = [90,110,110, 110]
    case_dict = {'Ngen':3, 'Nval': cal_Nvariable(scenario), 'Ts': 0.5*10**-4, 'Ttotal': 200,
                'wpu':1.0, 'Wmax': 1.02, 'error': error_mat, 'input_flag': True, 'load_pattern': load_pattern}
    
    A = generate_Amat(gen_dict, con_dict, case_dict, 'nudge')
    A_secPI = generate_Amat(gen_dict, con_dict, case_dict, 'secondPI') 
    B = generate_Bmat(gen_dict, con_dict, case_dict)
    
    # Optimize
    n = A.shape[0]

    # Define the LMI variables
    P = cp.Variable((n,n), symmetric = True)
    gamma = 1e-6
    G = np.eye(n)* gamma
    # Define the constraints for the LMI
    
    constraints = [P >> G, A.T @ P + P@A << 0,
                   A_secPI.T @ P + P@A_secPI << 0]
    
    # Define the objective for the LMI optimization problem
    obj = cp.Minimize(cp.trace(P))
    #obj = cp.Minimize(cp.trace(P))
    
    # Solve the LMI optimization problem
    prob = cp.Problem(obj, constraints)
    
    # Create a SolverOptions object and set the tolerance to 1e-6
    # Solve the problem with the specified options
    #prob.solve(solver=cp.MOSEK, verbose=True) #, options=options)
    
    try:
        prob.solve(solver=cp.MOSEK)
        cost = np.trace(P.value)
        
    except BaseException as e:
        logging.exception("Optimize Failed")
        cost = 10000000
        #print(f"Problem Cannot solved, check with {param}")
    return cost


def run_time_minimize(param):
    
    Ts = 2.0*10**-4 #0.5*10**-4
    wpu = 1.0;
    Wmax = 1.02;
    
    # Generator Parameter
    Pgen = np.array([1, 1, 1])
    droop = 0.05
    Tv = np.array([0.5, 1.0, 1.0])
    Te = np.array([0.05, 0.1, 0.1])
    H = np.array([1.587, 2.5, 2.5])
    D = 0.01
    gen_dict = {'Sbase':150, 'Pgen':Pgen, 'droop':droop,
                'Tv':Tv, 'Te':Te, 'H':H, 'D':D}
    
    if len(param) == 1:
        param = [param]
        param.append(1.76)
    # Control Parameter
    con_dict = {'kp1': 6.7,
                'ki1': 3.2,
                'ki2': 5,
                'kc1' : param[0],
                'Tl1' : param[1]
               }

    Ts = 0.5*10**-4
    error_mat = np.array([0.0005, 0, -0.0005])
    
    scenario = 'nudge'
    #scenario = 'droop'
    load_pattern = [100,80,80, 90]
    case_dict = {'Ngen':3, 'Nval': cal_Nvariable(scenario), 'Ts': 0.5*10**-4, 'Ttotal': 100, 'Tsamp':1/1000,
                'wpu':1.0, 'Wmax': 1.02, 'error': error_mat, 'input_flag': True, 'load_pattern': load_pattern}
    

    X_nudge, U = run_timeseries(gen_dict, con_dict, case_dict, 'nudge')
    load_pattern = case_dict['load_pattern']
    
    Ngen = case_dict['Ngen']
    Nval = case_dict['Nval']
    
    Ttotal, Ts = case_dict['Ttotal'], case_dict['Ts']

    load_pattern = case_dict['load_pattern']
    
    Nsample = int(Ttotal/Ts)
    Ndist = int(20*1/Ts) #int(0.1 * Nsample)
    sum_cost = 0
    try:
        # for i in range(Ngen):
        #     idx = 1+i*Nval
        #     sum_cost += np.sum(abs(X_nudge[idx, Ndist+2:]))*10
        
        if Ngen == 2:
            idx =  1+0*Nval
            idy =  1+1*Nval
            
            dif_gen = np.sum(abs(X_nudge[idx, Ndist+2:] - X_nudge[idy, Ndist+2:]))
            sum_cost += dif_gen
            
        elif Ngen == 3:
            idx =  1+0*Nval
            idy =  1+1*Nval
            idz =  1+2*Nval
            
            dif_gen1 = np.sum(abs(X_nudge[idx, Ndist+2:] - X_nudge[idy, Ndist+2:]))
            dif_gen2 = np.sum(abs(X_nudge[idx, Ndist+2:] - X_nudge[idy, Ndist+2:]))
            sum_cost += (dif_gen1 + dif_gen2)/2
            
        sum_cost += np.sum(abs(X_nudge[0,Ndist+2:])) * 20

    except BaseException as e:
        logging.exception("Optimize Failed")
        cost = 10000000
    print('.', end='')
    
    return sum_cost 

# CODE START
if __name__ == '__main__':
    
    param = np.array([0.1, 2.4])
    Ts = 0.5*10**-4
    wpu = 1.0;
    Wmax = 1.02;
    
    # Generator Parameter
    Pgen = np.array([1, 1, 1])
    droop = 0.05
    Tv = np.array([0.5, 1.0, 1.0])
    Te = np.array([0.05, 0.1, 0.1])
    H = np.array([1.587, 2.5, 2.5])
    D = 0.01
    gen_dict = {'Sbase':150, 'Pgen':Pgen, 'droop':droop,
                'Tv':Tv, 'Te':Te, 'H':H, 'D':D}
    
    if len(param) == 1:
        param = [param]
        param.append(1.76)
    # Control Parameter
    con_dict = {'kp1': 6.7,
                'ki1': 3.2,
                'ki2': 5.0,
                'kc1' : param[0],
                'Tl1' : param[1]
               }

    Ts = 0.5*10**-4
    error_mat = np.array([0.0005, 0, -0.0005])
    
    scenario = 'nudge'
    #scenario = 'droop'
    load_pattern = [90,110,110, 110]
    case_dict = {'Ngen':3, 'Nval': cal_Nvariable(scenario), 'Ts': 0.5*10**-4, 'Ttotal': 80,
                'wpu':1.0, 'Wmax': 1.02, 'error': error_mat, 'input_flag': True, 'load_pattern': load_pattern,
                'Tsamp':1/1000, 
                }
    
    A = generate_Amat(gen_dict, con_dict, case_dict, 'nudge')
    A_secPI = generate_Amat(gen_dict, con_dict, case_dict, 'secondPI') 
    B = generate_Bmat(gen_dict, con_dict, case_dict)
    
    # Optimize
    n = A.shape[0]

    # Define the LMI variables
    P = cp.Variable((n,n), symmetric = True)
    gamma = 1e-4
    G = np.eye(n)* gamma
    # Define the constraints for the LMI
    
    constraints = [P >> G, A.T @ P + P@A << 0,
                   A_secPI.T @ P + P@A_secPI << 0]
    
    # Define the objective for the LMI optimization problem
    obj = cp.Minimize(0)
    #obj = cp.Minimize(cp.trace(P))
    
    # Solve the LMI optimization problem
    prob = cp.Problem(obj, constraints)
    
    # Create a SolverOptions object and set the tolerance to 1e-6
    # Solve the problem with the specified options
    #prob.solve(solver=cp.MOSEK, verbose=True) #, options=options)
    try:
        prob.solve(solver=cp.MOSEK)
        cost = np.trace(P.value)
    except:
        cost = 100000
        #print(f"Problem Cannot solved, check with {param}")


    
    
    