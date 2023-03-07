# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:27:31 2023

@author: George Hine
 email:  PPYGH3@nottingham.ac.uk
 
"""


# Coil document:

# Set working directory
import os
os.chdir(r'D:\University\Year 4\University Notes\Modern Applications of Physics - From Research to Industry\Code\biot-savart')

#%% Imports


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker

import biot_savart_v4_3 as bs
import Resonant_coupling_functions as res

#%% Test plotting of test data

# bs.plot_coil("coil.txt")

bs.helmholtz_coils('coil1.txt', 'coil2.txt', 100, 100, 100, 1)
# fig = plt.figure()
bs.plot_coil('coil1.txt','coil2.txt')
# bs.plot_coil('coil2.txt')

# div_coil = bs.slice_coil(coil, steplength)
field = bs.calculate_field('coil1.txt', 100, 100, 100)

#%% Creating pair of powered coils

I = 1 # Current
R = 5 # Radius
N = 50 # Number of Segments 
Spacing = 2 # Spacing of segments (cm)

bs.helmholtz_coils("helm1.txt", "helm2.txt", N, R, Spacing, I)

# Plot coils in 3D matplot
bs.plot_coil("helm1.txt", "helm2.txt")


# Create meshgrid for target volume surrounding origin for vector fields of coils
bs.write_target_volume("helm1.txt", "targetvol1", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)
bs.write_target_volume("helm2.txt", "targetvol2", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)

# Create fields from taregt volumes
h1 = bs.read_target_volume("targetvol1")
h2 = bs.read_target_volume("targetvol2")


# use linear superposition of magnetic fields, to get the combined effects of multiple coils
h_total = h1 + h2

# plot magnetic fields in each plane
bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='x', level=0, num_contours=50)

bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='y', level=0, num_contours=50)

bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='z', level=0, num_contours=50)

#%% Creating one powered coil and one unpowered

I = 1 # Current
I_0 = 1e-10
R = 5 # Radius
N = 50 # Number of Segments 

separation = 8

Spacing1 = separation/2 # Spacing of segments (cm)
Spacing2 = -separation/2 # Spacing of segments (cm)
# Create powered coil
bs.create_Bz_circle('coil1.txt', N, R, Spacing1, I, (0,0))


# Create unpowered coil
bs.create_Bz_circle('coil2.txt', N, R, Spacing2, I, (0,0))

# Plot coils in 3D matplot
bs.plot_coil("coil1.txt", "coil2.txt")

# Create meshgrid for target volume surrounding origin for vector fields of coils
bs.write_target_volume("coil1.txt", "targetvol1", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)
bs.write_target_volume("coil2.txt", "targetvol2", (10, 10, 10), (-5, -5, -5), 0.5, 0.5)

# Create fields from taregt volumes
h1 = bs.read_target_volume("targetvol1")
h2 = bs.read_target_volume("targetvol2")

# use linear superposition of magnetic fields, to get the combined effects of multiple coils
h_total = h1 + h2

# plot magnetic fields in each plane
bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='x', level=0, num_contours=50)

bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='y', level=0, num_contours=50)

bs.plot_fields(h_total, (10, 10, 10), (-5, -5, -5), 0.5, which_plane='z', level=0, num_contours=50)

#%% # Now wish to use the level efature to create many slices through field


I = 1 # Current
I_0 = 1e-10
R = 5 # Radius
N = 50 # Number of Segments 

separation = 8

Spacing1 = separation/2 # Spacing of segments (cm)
Spacing2 = -separation/2 # Spacing of segments (cm)

size = 20
box_size = (size,size,size)

start = -10
start_point = (start,start,start)
# Create powered coil
bs.create_Bz_circle('coil1.txt', N, R, Spacing1, I, (0,0))


# Create unpowered coil
bs.create_Bz_circle('coil2.txt', N, R, Spacing2, I, (2,3))

# Plot coils in 3D matplot
bs.plot_coil("coil1.txt", "coil2.txt")

# Create meshgrid for target volume surrounding origin for vector fields of coils
bs.write_target_volume("coil1.txt", "targetvol1", box_size, start_point, 0.5, 0.5)
bs.write_target_volume("coil2.txt", "targetvol2", box_size, start_point, 0.5, 0.5)

# Create fields from taregt volumes
h1 = bs.read_target_volume("targetvol1")
h2 = bs.read_target_volume("targetvol2")

# use linear superposition of magnetic fields, to get the combined effects of multiple coils
h_total = h1 + h2

# plot magnetic fields in each 10 slices vertically:
slices = np.arange(0,11,1)
for i in range(0,3):    #len(slices)
    bs.plot_fields(h_total, box_size, start_point, 0.5, which_plane='x', level= slices[i], num_contours=50)
    plt.title('Slice {} at z = {}'.format(i,slices[i]))
    
    # un-comment for Ignore slice 0
    
    # if i == 0:
    #     plt.close()
    # else:
    #     continue
#%% Testing cell:
import Resonant_coupling_functions as res
    
Rad = 1 # meters

wire_A = 1e-8

N_coil = 10

rho = 5.9e-3

freq = 8e+4
omega = 2*np.pi*freq

k = 0.1

length = 0.05

area = res.area(Rad)

# by assuming the capacitor is a perfect acpacitor we can ignore the affects on the resistance in circuit two

C = 1e-6 # Capacitance

Res = res.total_resitance(rho, Rad, N_coil, wire_A) # area of the wire

L1 = res.inductance(N_coil, length, area) # area of the coil

L2 = res.inductance(N_coil, length, area) # area of the coil


inductance_reactance = res.ind_react(freq, L2)

capacitance_reactance = res.cap_react(freq, C)

impedence = res.impedence(Res, inductance_reactance, capacitance_reactance)

mag_ctf = res.mag_CTF(omega, L1, L2, k, impedence)

#%% First Varying capacitance to explore relationship


# Now want to graph how mag ctf varies against each value

import Resonant_coupling_functions as res
    
Rad = 1 # meters

wire_rad = 5.64e-5
wire_A = 1e-8

N_coil = 10

rho = 5.9e-3

freq = 8e+4

omega = 2*np.pi*freq

k = 0.1

length = 0.05

area = res.Area(Rad)

# by assuming the capacitor is a perfect acpacitor we can ignore the affects on the resistance in circuit two

C = 1e-6 # Capacitance
C_ar = np.arange(1e-8,1e-4,1e-7) # Capacitance


Res = res.total_resitance(rho, Rad, N_coil, wire_A) # area of the wire

L1 = res.inductance(N_coil, length, area) # area of the coil

L2 = res.inductance(N_coil, length, area) # area of the coil


inductance_reactance = res.ind_react(freq, L2)

capacitance_reactance = res.cap_react(freq, C_ar)

impedence = res.Impedence(Res, inductance_reactance, capacitance_reactance)

mag_ctf = res.mag_CTF(omega, L1, L2, k, impedence)

mag_CTF = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k, length, C_ar)

plt.figure()
plt.plot(C_ar,mag_ctf)
plt.xlabel('Capacitance')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(C_ar,mag_CTF)
plt.xlabel('Capacitance')
plt.ylabel('Current Transfer function magnitude')

#%% Varying components


Rad_ar = np.arange(0.01,2.01,0.01)
Rad = np.mean(Rad_ar) # meters


wire_rad_ar = np.arange(1e-5,(2+1e-5)*1e-3,1e-5)
wire_rad = np.mean(wire_rad_ar)


N_coil_ar = np.arange(1,201,1)
N_coil = np.mean(wire_rad_ar)

rho_ar = np.arange(1e-4,(2+1e-4)*1e-2,1e-4)
rho = np.mean(rho_ar)


freq_ar = np.arange(1e+4,(2)*1e+6 + 1e+4,1e+4)
freq = np.mean(freq_ar)


k_ar = np.arange(0.01,2.01,0.01)
k = np.mean(k_ar)


length_ar = np.arange(0.01,2.01,0.01)
length = np.mean(length_ar)

C_ar = np.arange(1e-7,(2+1e-7)*1e-5,1e-7) # Capacitance
C = np.mean(C_ar)


mag_CTF__Capacitance = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k, length, C_ar)


mag_CTF__length = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k, length_ar, C)


mag_CTF__k = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k_ar, length, C)


mag_CTF__frequency = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq_ar, k, length, C)


mag_CTF__resistivity = res.CTF_magnitude(Rad, wire_rad, N_coil, rho_ar, freq, k, length, C)


mag_CTF__Number_of_coils = res.CTF_magnitude(Rad, wire_rad, N_coil_ar, rho, freq, k, length, C)


mag_CTF__wire_radius = res.CTF_magnitude(Rad, wire_rad_ar, N_coil, rho, freq, k, length, C)


mag_CTF__radius = res.CTF_magnitude(Rad_ar, wire_rad, N_coil, rho, freq, k, length, C)


data_for_frame = np.vstack([mag_CTF__wire_radius,mag_CTF__length,mag_CTF__resistivity,mag_CTF__Capacitance]).T

import pandas as pd

df = pd.DataFrame(data=data_for_frame,columns=['Wire radius','Coil height','Resitivity','Capacitance'],dtype=float)
print(df)
df.to_csv('EV_charging.csv')

#%% sort dataframe for pcr

import pandas as pd


df_sorted = df.apply(lambda x: x.sort_values().values)

# print the sorted dataframe
print(df)


#%% What if we vary all of them in a massive arrau amd look for the maximum

mag_CTF_global = res.CTF_magnitude(Rad_ar, wire_rad_ar, N_coil_ar, rho_ar, freq_ar, k_ar, length_ar, C_ar)
maximum = np.max(mag_CTF_global)
arg = np.argwhere(mag_CTF_global == maximum)

# Therefore optimum:
Rad_opt = Rad_ar[arg] # meters


wire_rad_opt = wire_rad_ar[arg]


N_coil_opt = wire_rad_ar[arg]

rho_opt = rho_ar[arg]


freq_opt = freq_ar[arg]


k_opt = k_ar[arg]

length_opt = length_ar[arg]

C_opt = C_ar[arg]

print(Rad_opt,wire_rad_opt,N_coil_opt,rho_opt,freq_opt,k_opt,length_opt,C_opt)

# This suggests the biggest for all of them so feel like this is not working

# Instead want to matmul

# mag_CTF_global_dot = np.cross(np.cross(mag_CTF__wire_radius, mag_CTF__length) , np.cross(mag_CTF__resistivity, mag_CTF__Capacitance))

#%% Plotting cell

plt.figure()
plt.plot(C_ar,mag_CTF__Capacitance)
plt.xlabel('Capacitance')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(length_ar,mag_CTF__length)
plt.xlabel('Length')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(k_ar,mag_CTF__k)
plt.xlabel('k')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(freq_ar,mag_CTF__frequency)
plt.xlabel('Frequency')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(rho_ar,mag_CTF__resistivity)
plt.xlabel('Resistivity')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(N_coil_ar,mag_CTF__Number_of_coils)
plt.xlabel('Number of Coils')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(wire_rad_ar,mag_CTF__wire_radius)
plt.xlabel('Wire Radius')
plt.ylabel('Current Transfer function magnitude')

plt.figure()
plt.plot(Rad_ar,mag_CTF__radius)
plt.xlabel('Radius of coil')
plt.ylabel('Current Transfer function magnitude')


#%% Principle component analsyis to find optimum



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d


# PCA Calculator for Numerical Star Data

df_sorted = pd.read_csv('EV_charging.csv')
array = df.to_numpy()

array = array[:,1:5]

# Now, the data must be standardized:
    
stdarray = np.zeros(np.shape(array))
mean = []
stddev = []
for i in range(0, 3):
    mean.append(np.mean(array[:, i]))
    stddev.append(np.std(array[:, i]))
    for j in range(0, 200):
        stdarray[j, i] = ((array[j, i] - mean[i])/stddev[i])

# Now to find the covariant matrix of the standardized set of data:
    
covmat = np.cov(np.transpose(stdarray))
eigvals, eigvects = np.linalg.eig(covmat)

PCperc = []

for i in range(3):
    PCsum = np.sum(eigvals)
    PCperc.append(eigvals[i]*100/PCsum)
    
# Now, let's plot a scree plot to observe the Principal Components

screeplot = plt.figure(dpi=200)
screeax = plt.axes()
PCs = np.linspace(1, 3, 3)
screeax.bar(PCs, PCperc)
plt.plot(PCs, PCperc, 'k--')
plt.plot(PCs, PCperc, 'k.')
plt.title('Scree Plot of Principal Components')
plt.ylabel('Percentage of Total Variance')

# Now to define the feature vector. I'm going to firstly define a 2D feature vector to test if it is sufficient enough to explain the variance

feature = eigvects[:, [0, 1]]

Final2D = np.matmul(np.transpose(feature), np.transpose(stdarray))

# Now to plot the 2D PCA plot

finalplot2d = plt.figure(dpi=200)
final2dax = plt.axes()
plt.plot(Final2D[0, :], Final2D[1, :], 'r.')

# This plot is good, but does not show the clustering too well, so it might be worth incorporating PC3:
    
feature3d = eigvects[:, [0, 1, 2]]

Final3D = np.matmul(np.transpose(feature3d), np.transpose(stdarray))

finalplot3d = plt.figure(dpi=200)
final3dax = plt.axes(projection='3d')
final3dax.scatter3D(Final3D[0, :], Final3D[1, :], Final3D[2, :], color='red')
plt.title('Principal Component Analysis of the EV wireles charging parameter space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.zlabel('PC3')


#%% Testing functions from http://rb.gy/zywydw

import Resonant_coupling_functions as res

# Global
R_fix_pls = 10
k = 0.45
R_w = 0.5 # ohms
freq = 8e+4 # herts
omega = 2*np.pi*freq # rad per second
U = np.linspace(240,500,260) # volts ?

# Source
wire_radius_s = 2*1e-3 # meters
A_wire_s = res.Area(wire_radius_s) # meters^2

radius_s = 0.5 # meters
N_coils_s = 10
length_s = 0.01 # meters
rho_s = 1.77e-8 # copper resistance per unit length
total_wire_length_s = res.length(radius_s, N_coils_s)
area_s = res.Area(radius_s)
R_s = res.total_resitance(rho_s, radius_s, N_coils_s, A_wire_s)
L_s = res.inductance(N_coils_s, length_s, area_s)

# Reciever
wire_radius_d = 2*1e-3 # meters
A_wire_d = res.Area(wire_radius_d) # meters^2

radius_d = 0.1 # meters
N_coils_d = 5
length_d = 0.01 # meters
rho_d = 1.77e-8 # copper resistance per unit length
total_wire_length_d = res.length(radius_d, N_coils_d)
area_d = res.Area(radius_d)
R_d = res.total_resitance(rho_d, radius_d, N_coils_d, A_wire_d)
L_d = res.inductance(N_coils_d, length_d, area_d)


# Calculating


# Mutual inductance
M = res.mutual_inductance_kL(k, L_s, L_d)


# equation 9 efficiency calculation
efficiency_boi = 100*(((omega*M)**2 * R_w)/((R_d+R_w)*(R_s*(R_d + R_w) + (omega*M)**2))) 

# M1 = res.mutual_inductance(k, N_coils, N_coils, radius, length)

C_s = res.Capacitance(omega, L_s)
C_d = res.Capacitance(omega, L_d) 


# Including reactance terms
# Z_s = res.S_impedence(R_s, R_d, M, omega, R_w)
# Z_d = res.D_impendence(R_s, R_d, M, omega, R_w)

# Excluding reactance terms 
Z_s = R_s + R_fix_pls
Z_d = R_d + R_w 


# Source Voltage and current
I_S = res.I_s(U, Z_d, Z_s, M, omega)
V_S = I_S/Z_s


# Reciever Voltage and current
I_D = res.I_d(U, Z_d, Z_s, M, omega)
V_D = I_D/Z_d


#Power in circuits
# P_S = I_S**2 * Z_s
# P_D = I_D**2 * Z_d


# Power In
P_in = res.P_in(U, Z_d, Z_s, M, omega)
I_in = (P_in/Z_s)**(1/2)
V_in = U


# Power Out
P_out = res.P_out(U, Z_d, Z_s, M, omega, R_w)
I_out =  (P_out/Z_d)**(1/2)
# V_out = I_out*(Z_d)
test = P_out - I_out**2 * Z_d


# Efficiency of power transfer 
efficiency = res.efficiency(P_in, P_out)


V_out = (N_coils_d*V_in)/(k*N_coils_s)


P_in = U**2/Z_s
I_in = P_in/U
P_out = P_in * efficiency/100
# I_out = P_out

battery_capacity =  65*1e+3
charge_time  = battery_capacity/P_out
# max_efficiency = np.max(efficiency_boi)
for_best = np.argwhere(charge_time == 8.601343021521688)

## Decided will have two different charge modes: trickle and speedy boost

# trickle  charging   
# input circuit  V = 277 V          A = 27               charge time 8.60 hours
# output circuit V = 307 V          A = 24.3
#%%

# To do list

# calculate the losses
# calculate the current in each loop (amplitude - take out the j) 
# voltages in each loop (U in and U out)

#%%
import pandas as pd
data_for_frame = np.vstack((R_s,R_d,R_w,Z_s,Z_d,C_s,C_d,I_S,I_D,V_S,V_D,efficiency_boi)).T
df = pd.DataFrame(data=data_for_frame,columns=['R_s','R_d','R_w','Z_s','Z_d','C_s','C_d','I_S','I_D','V_S','V_d','efficiency_boi'],dtype=float)
display(df.to_string())
# df.to_csv('EV_charging.csv')

#%%

# Help me