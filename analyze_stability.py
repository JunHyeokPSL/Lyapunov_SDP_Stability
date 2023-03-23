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
    
    Nmat = 1+Ngen*Nval
    Nsample = int(Ttotal/Ts)
    Ndist = int(0.1 * Nsample) 
    
    X_dot = np.zeros([Nmat,1])
    del_X = np.zeros([Nmat, Nsample+1])
    U1 = np.zeros([1, Nsample+1]) # Load Disturbances
    U2 = np.zeros([Ngen, Nsample+1]) # Measurement Noises
    
    del_X[:,0:1] = X_dot*Ts
    
    #initial operating point
    Winit = -0.002
    del_X[0,0] = Winit
    sumPg = sum(Pgen)
    
    PLinit = Winit*D - Winit/droop*sumPg
    
    A = generate_Amat(gen, con, case, 'droop')
    B = generate_Bmat(gen, con, case)
    
    for i in range(Ngen):
        idx = 1 + i*Nval
        del_X[idx,0] = -Winit/droop * Pgen[i]
        del_X[idx+3,0] = Winit

    for i in range(Nsample):
        
        # Switch from Droop to the other control
        if i == Ndist+1:
            A = generate_Amat(gen, con, case, scenario)
            print(f'A matrix change from droop to {scenario}')
        # Disturbance Update
        U2[:,i] = error_mat[:Ngen]
        if i <= 60*1/Ts:
            U1[0,i] = PLinit
            
        elif i>60*1/Ts and i<100*1/Ts:
            U1[0,i] = load_pattern[0] / Sbase /sumPg
            
        elif i>100*1/Ts and i<140*1/Ts:
            U1[0,i] = load_pattern[1] / Sbase /sumPg 
    
    
        elif i>140*1/Ts and i<170*1/Ts:
            U1[0,i] = load_pattern[2] / Sbase /sumPg 
    
        else:
            U1[0,i] = load_pattern[3] / Sbase /sumPg 
    
        X_dot = np.dot(A, del_X[:,i]) + np.dot(B[:,0:1], U1[:,i]) + np.dot(B[:,1:1+Ngen], U2[:,i])
        del_X[:,i+1] = del_X[:,i]+ Ts*X_dot;  
    
    end = time.time()
    print('Simulation Time:', end - start, 'secs')
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
    print('Lyapunov Simulation Time:', end - start, 'secs')
    
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

# CODE START
if __name__ == '__main__':
    
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
    
    # Control Parameter
    con_dict = {'kp1': 8,
                'ki1': 2.5,
                'ki2': 5,
                'kc1' : 0.1,
                'Tl1' : 2.4
               }
 
    Ngen = 3 # number of generator
    Nval = 6 # meaning
    Ts = 0.5*10**-4
    error_mat = np.array([0.001, 0, -0.001])
    scenario = 'droop'
    case_dict = {'Ngen':3, 'Nval': cal_Nvariable(scenario), 'Ts': 0.5*10**-4, 'Ttotal': 200,
                'wpu':1.0, 'Wmax': 1.02, 'error': error_mat}
    
    
    A = generate_Amat(gen_dict, con_dict, case_dict, scenario)
    B = generate_Bmat(gen_dict, con_dict, case_dict)
    
    X_dot, U1, U2 = run_timeseries(gen_dict, con_dict, case_dict, 'droop')