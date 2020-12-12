# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:20:06 2020

@author: Glogta
"""
# Returns minimum difference between any pair 
def findMinDiff(arr):
    arr = np.unique(arr)
    n = len(arr)
    # Sort array in non-decreasing order 
    arr = sorted(arr) 
  
    # Initialize difference as infinite 
    diff = 10**20
  
    # Find the min diff by comparing adjacent 
    # pairs in sorted array 
    for i in range(n-1): 
        if arr[i+1] - arr[i] < diff: 
            diff = arr[i+1] - arr[i] 
  
    # Return min diff 
    return diff 


import gdal
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Parser
with open('./output/NDVI_example_TVDI_line_equations.txt') as f:
    test = f.readlines()
T_NDVI = test[1].split()
edges_param = [float(x) for x in T_NDVI[1:]] 

#--------------------------------- Load data -------------------------------
outputfile = './input/NDVI_example.tif'
fid=gdal.Open(outputfile,gdal.GA_ReadOnly)
rows_NDVI=fid.RasterYSize
cols_NDVI=fid.RasterXSize
# read each band and store the arrays
NDVI=fid.GetRasterBand(1).ReadAsArray()

outputfile = './input/LST_example.tif'
fid=gdal.Open(outputfile,gdal.GA_ReadOnly)
rows_LST=fid.RasterYSize
cols_LST=fid.RasterXSize
# read each band and store the arrays
LST = fid.GetRasterBand(1).ReadAsArray()

x = np.arange(np.min(NDVI), np.max(NDVI))
y = np.arange(np.min(LST), np.max(LST))

NDVI_max = (edges_param[2]-edges_param[0])/edges_param[1]
NDVI_min = edges_param[0]
LST_min = edges_param[2]+NDVI_max*edges_param[3]
LST_max = edges_param[2]

#-------------------------- reshape data----------------------------------------

LST_array = LST.reshape(1,rows_LST*cols_LST)
NDVI_array = NDVI.reshape(1,rows_NDVI*cols_NDVI)


#----------------Sort the NDVI and do a paired sort on the LST ---------------
NDVI_sorted, LST_sorted = zip(*sorted(zip(NDVI_array.T, LST_array.T)))

NDVI_sorted = np.asarray(NDVI_sorted)
LST_sorted = np.asarray(LST_sorted)

NDVI_mask = (NDVI_sorted < NDVI_max) & (NDVI_sorted > 0)
NDVI_sorted = NDVI_sorted[NDVI_mask]
LST_sorted = LST_sorted[NDVI_mask]

#the positions of the phi_min & max
phi_min = [min(NDVI_sorted), max(LST_sorted)]
phi_max = [max(NDVI_sorted),min(LST_sorted)]

#quick test where phi_min_i and max lies
dry_edge_plot = np.array([[0, NDVI_min],[NDVI_max, LST_min]])
wet_edge_plot = np.array([[0, edges_param[2]],[NDVI_max, LST_min]])
plt.figure(1)
plt.plot(NDVI_sorted, LST_sorted, 'k.', markersize=1)
plt.plot(dry_edge_plot[:,0],dry_edge_plot[:,1], 'r')
plt.plot(wet_edge_plot[:,0],wet_edge_plot[:,1], 'r')
plt.plot(max(NDVI_sorted),min(LST_sorted), 'o', label="$\phi_{max}$")
plt.plot(min(NDVI_sorted), max(LST_sorted), 'o', label="$\phi_{min}$")
plt.grid()
plt.xlabel('NDVI')
plt.ylabel('LST')
plt.title('T-NDVI')
plt.legend()
plt.show()


#LST_i = LST.reshape(1,rows_LST*cols_LST)

#phi min is defined as = 0
phi_min = 0
#phi max is defined as = (Delta+gamma/Delta)
phi_max = ...

#phi is defined as
#Create a linear regression for the dry edge in indexes for the NDVI
model_dry = LinearRegression()
x_dry = np.array([0, NDVI_max])
y_dry = np.array([NDVI_min, LST_min])
model_dry.fit(x_dry.reshape(-1, 1), y_dry)

#find T_i_max (T values for all NDVI values on the dry edge)
LST_i_max = model_dry.predict(NDVI_sorted.reshape(-1,1))

#create a linear regression for the wet edge in endexes for NDVI
model_wet = LinearRegression()
x_wet = np.array([0, NDVI_max])
y_wet = np.array([LST_max, LST_min])
model_wet.fit(x_wet.reshape(-1, 1), y_wet)

#find T_i_max (T values for all NDVI values on the dry edge)
LST_i_min = model_wet.predict(NDVI_sorted.reshape(-1,1))



#now i need to find the parameterized values for phi





#vapor pressure curve ligning 5.26 side 240 (an introduction to planetery atmospheres)
#vapor pressure curve: 0.6112*exp(17.67*T/(T+243.5)): https://glossary.ametsoc.org/wiki/Clausius-clapeyron_equation
#the curve is: diff(vapor pressure) = 0.6112*(17.67/(T+243.5)-17.67*T/(T+243.5)**2)*exp(17.67*T/(T+243.5))
T = np.mean(LST_array)
delta = 0.6112*(17.67/(T+243.5)-17.67*T/(T+243.5)**2)*np.exp(17.67*T/(T+243.5))

#Psychromatic constant - temporary kenan pdf = 0.667
gamma = 0.667
phi_max = (delta+gamma)/delta

#phi_min_i
phi_min = phi_max*((NDVI_sorted-0)/(NDVI_max-0))**2


#calculate phi
phi = np.zeros(len(NDVI_sorted))
for i in range(1,len(NDVI_sorted)):
    phi[i] = (LST_i_max[i] - LST_sorted[i])/(LST_i_max[i]-LST_i_min[i])*(phi_max-phi_min[i])+phi_min[i]
    
    
plt.figure()
plt.plot(NDVI_sorted, phi_min)
plt.plot(NDVI_max,phi_max, 'ks')
plt.xlabel('NDVI')
plt.ylabel('$\phi_{i,min}$')
plt.title('$\phi_{i,min}$ as a function of NDVI')
plt.grid()

