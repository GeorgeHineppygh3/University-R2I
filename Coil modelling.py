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

# Imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as ticker

import biot_savart_v4_3 as bs


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
