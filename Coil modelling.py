# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:27:31 2023

@author: George Hine
 email:  PPYGH3@nottingham.ac.uk
 
"""
S

# Coil document:

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
import Resonant_coupling_functions as res

#%% Test plotting of test data
# Test plotting of test data
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
bs.create_Bz_circle('coil2.txt', N, R, Spacing2, I, (0,0))

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
for i in range(0,len(slices)):    
    bs.plot_fields(h_total, box_size, start_point, 0.5, which_plane='x', level= slices[i], num_contours=50)
    plt.title('Slice {} at z = {}'.format(i,slices[i]))
    
    # un-comment for Ignore slice 0
    
    # if i == 0:
    #     plt.close()
    # else:
    #     continue
#%% Testing:
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

#%% 
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

#%%

Rad = 1 # meters
Rad_ar = np.arange(0.1,1.5,0.1)

wire_rad = 5.64e-5
wire_rad_ar = np.arange(1e-5,1e-3,1e-5)


N_coil = 10
N_coil_ar = np.arange(1,100,1)

rho = 5.9e-3
rho_ar = np.arange(1e-4,1e-2,1e-4)

freq = 8e+4
freq_ar = np.arange(1e+4,1e+6,1e+4)

k = 0.1
k_ar = np.arange(0.01,1,0.01)

length = 0.05
length_ar = np.arange(0.01,1,0.01)

C = 1e-6 # Capacitance
C_ar = np.arange(1e-7,1e-5,1e-7) # Capacitance

mag_CTF__Capacitance = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k, length, C_ar)


mag_CTF__length = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k, length_ar, C)


mag_CTF__k = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq, k_ar, length, C)


mag_CTF__frequency = res.CTF_magnitude(Rad, wire_rad, N_coil, rho, freq_ar, k, length, C)


mag_CTF__resitivity = res.CTF_magnitude(Rad, wire_rad, N_coil, rho_ar, freq, k, length, C)


mag_CTF__Number_of_coils = res.CTF_magnitude(Rad, wire_rad, N_coil_ar, rho, freq, k, length, C)


mag_CTF__wire_radius = res.CTF_magnitude(Rad, wire_rad_ar, N_coil, rho, freq, k, length, C)


mag_CTF__radius = res.CTF_magnitude(Rad_ar, wire_rad, N_coil, rho, freq, k, length, C)


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
plt.plot(rho_ar,mag_CTF__resitivity)
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



