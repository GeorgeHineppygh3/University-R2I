# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:56:00 2023

@author: George Hine
 email:  PPYGH3@nottingham.ac.uk
 
"""

# Resonant coupling fucntions:
    

# Set working directory
import os
os.chdir(r'D:\University\Year 4\University Notes\Modern Applications of Physics - From Research to Industry\Code\biot-savart')

# Imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker

import biot_savart_v4_3 as bs
# i = np.sqrt(-1)
mu0 = 4*np.pi*1e-7
non_coil_length = 1          # units in meters

#%%

def Impedence(resistance,ind_react,cap_react):
    return np.sqrt(resistance**2 + (ind_react+cap_react)**2)


def ind_react(freq,inductance):
    return 2*np.pi*freq*inductance

def cap_react(freq,capacitance):
    return (-1)/(2*np.pi*freq*capacitance)


def length(radius,num_coils):
    return 2*np.pi*radius*num_coils

def total_resitance(rho,coil_rad,num_coil,A_wire):
    coil_length = length(coil_rad,num_coil)
    total_length = coil_length + non_coil_length
    resitance = (rho*total_length)/A_wire
    return resitance

def mag_CTF(omega,L_1,L_2,k,impedence):
    mag = (omega*k*np.sqrt(L_1*L_2))/np.sqrt((impedence**2) + (omega**2)*L_2**2)
    return mag

def Area(radius):
    return np.pi*radius**2

def inductance(num_coil,length,area):
    mu0 = 4*np.pi*1e-7
    ind = (mu0*(num_coil**2)*area)/length
    return ind # this length is apparently the coil length (which means height)

def CTF_magnitude(radius,wire_radius,num_coil,rho,freq,k,length,capacitance):

    omega = 2*np.pi*freq

    area = Area(radius)
    wire_A = Area(wire_radius)

    Res = total_resitance(rho, radius, num_coil, wire_A) # area of the wire

    L1 = inductance(num_coil, length, area) # area of the coil

    L2 = inductance(num_coil, length, area) # area of the coil


    inductance_reactance = ind_react(freq, L2)

    capacitance_reactance = cap_react(freq, capacitance)

    impedence = Impedence(Res, inductance_reactance, capacitance_reactance)

    mag_ctf = mag_CTF(omega, L1, L2, k, impedence)
    
    return mag_ctf

def mag_VTF(omega,L_1,L_2,k,impedence):
    mag = (-k*np.sqrt(L_2))/(np.sqrt(L_1)*(L_2*(k**2 - 1)*omega**2)+impedence)
    return mag

def VTF_magnitude(radius,wire_radius,num_coil,rho,freq,k,length,capacitance):
    
    omega = 2*np.pi*freq

    area = Area(radius)
    wire_A = Area(wire_radius)

    Res = total_resitance(rho, radius, num_coil, wire_A) # area of the wire

    L1 = inductance(num_coil, length, area) # area of the coil

    L2 = inductance(num_coil, length, area) # area of the coil



    capacitance_reactance = cap_react(freq, capacitance)

    impedence = Impedence(Res, inductance_reactance, capacitance_reactance)

    mag_vtf = mag_VTF(omega, L1, L2, k, impedence)
    
    return mag_vtf


#%% After emailing tony he says don't worry abou the impedence

# def mag_ctf_

# def mag_CTF_(omega,L_1,L_2,k):
#     mag = (omega*k*np.sqrt(L_1*L_2))/np.sqrt((impedence**2) + (omega**2)*L_2**2)
#     return mag


#%% Following the maths from `https://iopscience.iop.org/article/10.1088/1742-6596/2125/1/012035/pdf`

def mutual_inductance(k,N_1,N_2,Radius,length):
    mur = 1 # air
    area = Area(Radius)
    M = k*((mu0*mur*N_1*N_2*area)/(length)) # taken from http://rb.gy/z159j9
    return M

def mutual_inductance_kL(k,L1,L2):
    return k*np.sqrt(L1*L2)

def Capacitance(omega,inductance):
    C = 1/(omega**2 * inductance)
    return C
    

# def S_impedence(R_s):
#     # due to the system being in resonance from tuning the capacitance 
#     # the impedence is just the circuit resistance
#     return R_s
    
# def D_impendence(R_d,R_w):
#     # due to the system being in resonance from tuning the capacitance 
#     # the impedence is just the circuit resistance + the resitor 
#     # in the receiver cirucit
#     return R_d + R_w


def S_impedence(R_s, R_d, M, omega, R_w):
    R = R_s + (omega*M)**2/(R_d + R_w)
    return R
    
def D_impendence(R_s, R_d, M, omega, R_w):
    R = R_d + R_w + (omega*M)**2/(R_s)
    return R

def P_in(U, Z_d, Z_s, M, omega):
    P = (U**2 * Z_d)/(Z_s*Z_d + (omega*M)**2) 
    return P

def P_out(U, Z_d, Z_s, M, omega, R_w):
    P = (U**2 * (omega*M)**2 * R_w)/(Z_s*Z_d + (omega*M)**2)**2
    return P

def efficiency(In,Out):
    return Out/In * 100

def I_s(U, Z_d, Z_s, M, omega):
    I = (Z_d*U)/(Z_s*Z_d + (omega*M)**2)
    return I

def I_d(U, Z_d, Z_s, M, omega):
    I = (omega*M*U)/(Z_s*Z_d + (omega*M)**2)
    return I