# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:18:16 2020

@author: Anders Slotsbo
"""
#=============================== Run T-NDVI script ================================================
from T_NDVI import *
import gdal
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from data_load import load_data

#========================== Create the seperate data files =======================
date = '07_11_20'
load_data(date)


NDVI_path = 'Data/NDVI_' + date + '.tif'
LST_path = 'Data/LST_' + date + '.tif'
CLM_path = 'Data/CLM_' + date + '.tif'
output_path = "Output"

T_NDVI(NDVI_path, LST_path, CLM_path, output_path)

#==================================== Get T-NDVI plot) ======================================
#======================================== Parser ============================================
with open(output_path + '/NDVI_'+ date +'_TVDI_line_equations.txt') as f:
    test = f.readlines()
T_NDVI = test[1].split()
edges_param = [float(x) for x in T_NDVI[1:]]

#====================================== Load data ===========================================
# NDVI
fid=gdal.Open(NDVI_path, gdal.GA_ReadOnly)
rows_NDVI=fid.RasterYSize
cols_NDVI=fid.RasterXSize
# read the band and store the arrays
NDVI=fid.GetRasterBand(1).ReadAsArray()

# LST
fid=gdal.Open(LST_path, gdal.GA_ReadOnly)
rows_LST=fid.RasterYSize
cols_LST=fid.RasterXSize
# read the band and store the arrays
LST = fid.GetRasterBand(1).ReadAsArray()

x = np.arange(np.min(NDVI), np.max(NDVI))
y = np.arange(np.min(LST), np.max(LST))

NDVI_max = (edges_param[2]-edges_param[0])/edges_param[1]
NDVI_min = edges_param[0]
LST_min = edges_param[2]+NDVI_max*edges_param[3]
LST_max = edges_param[2]

# mask
fid=gdal.Open(CLM_path, gdal.GA_ReadOnly)
rows_mask=fid.RasterYSize
cols_mask=fid.RasterXSize
# read each band and store the arrays
mask = fid.GetRasterBand(1).ReadAsArray()

#==================================== reshape data ==========================================

mask_array = mask.reshape(1,rows_mask*cols_mask)
LST_array = LST.reshape(1,rows_LST*cols_LST)
NDVI_array = NDVI.reshape(1,rows_NDVI*cols_NDVI)

NDVI_array = np.squeeze(NDVI_array)
LST_array = np.squeeze(LST_array)

#===================== Sort the NDVI and do a paired sort on the LST ========================
index = np.arange(0,len(NDVI_array))

NDVI_sorted, index_sorted = (np.array(x) for x in zip(*sorted(zip(NDVI_array, index))))

LST_sorted = LST_array[index_sorted]

#============================= create dry and wet edges ==================================

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

#find T_i_min (T values for all NDVI values on the dry edge)
LST_i_min = model_wet.predict(NDVI_sorted.reshape(-1,1))

#============================= plot before trimming ===========================
dry_edge_plot = np.array([[0, NDVI_min],[NDVI_max, LST_min]])
wet_edge_plot = np.array([[0, edges_param[2]],[NDVI_max, LST_min]])

plt.figure()
plt.plot(NDVI_sorted[NDVI_sorted > -1], LST_sorted[LST_sorted > np.min(LST_sorted)], 'k.', markersize=1)
plt.plot(dry_edge_plot[:,0],dry_edge_plot[:,1], 'r')
plt.plot(wet_edge_plot[:,0],wet_edge_plot[:,1], 'r')
plt.grid()
plt.xlabel('NDVI')
plt.ylabel('$\Delta T$ [°C]')
plt.title('T/NDVI')
plt.legend()
plt.show()
#========================== mask away unwated data =====================================
    
NDVI_mask = (NDVI_sorted < NDVI_max) & (NDVI_sorted > 0)
NDVI_sorted[NDVI_mask == 0] = -1
LST_sorted[NDVI_mask == 0] = np.min(LST_sorted)

LST_mask = (LST_sorted < LST_i_max) & (LST_sorted > LST_i_min)
NDVI_sorted[LST_mask == 0] = -1
LST_sorted[LST_mask == 0] = np.min(LST_sorted)

#================================= plot after trimming ======================================


#quick test where phi_min_i and max lies
dry_edge_plot = np.array([[0, NDVI_min],[NDVI_max, LST_min]])
wet_edge_plot = np.array([[0, edges_param[2]],[NDVI_max, LST_min]])
plt.figure()
plt.plot(NDVI_sorted[NDVI_sorted > -1], LST_sorted[LST_sorted > np.min(LST_sorted)], 'k.', markersize=1)
plt.plot(dry_edge_plot[:,0],dry_edge_plot[:,1], 'r')
plt.plot(wet_edge_plot[:,0],wet_edge_plot[:,1], 'r')
#plt.plot(max(NDVI_sorted),min(LST_sorted), 'o', label="$\phi_{max}$")
#plt.plot(min(NDVI_sorted), max(LST_sorted), 'o', label="$\phi_{min}$")
plt.grid()
plt.xlabel('NDVI')
plt.ylabel('$\Delta T$ [°C]')
plt.title('T/NDVI')
plt.legend()
plt.show()

#Define phi_max
phi_max = 1.26

#calculate phi_min_i
phi_min = phi_max*((NDVI_sorted-np.min(NDVI_sorted[NDVI_sorted != -1]))/(NDVI_max-np.min(NDVI_sorted[NDVI_sorted != -1])))**2

#calculate phi
phi = np.zeros(len(NDVI_sorted))
for i in range(1,len(NDVI_sorted)):
    phi[i] = (LST_i_max[i] - LST_sorted[i])/(LST_i_max[i]-LST_i_min[i])*(phi_max-phi_min[i])+phi_min[i]

#remove the masked out data
phi[NDVI_sorted == -1] = np.NaN   
plt.figure()
plt.plot(NDVI_sorted[NDVI_sorted > -1], phi_min[NDVI_sorted > -1],'.')
plt.xlabel('NDVI')
plt.ylabel('$\phi_{i,min}$')
plt.title('$\phi_{i,min}$ as a function of NDVI')
plt.grid()
plt.show()

#=================================== EF plotting =====================================
EF = phi

#reverse the sorting
index_return, EF_return = (np.array(x) for x in zip(*sorted(zip(index_sorted, EF))))

#reshape the EF into an image
EF_im = EF_return.reshape(rows_mask, cols_mask)

#mask out values over 1.26
EF_im[EF_im > 1.26] = np.NaN


#Calculate the latitude and longitude for plotting & get the projection for exporting
[cols, rows] = NDVI.shape
gt = fid.GetGeoTransform()
lat_max = gt[3]
lat_min = gt[3] + rows*gt[4] + cols * gt[5]
lon_max = gt[0] + rows*gt[1] + cols * gt[2]
lon_min = gt[0]

lon = np.linspace(lon_min,lon_max, 5)
lat = np.linspace(lat_min,lat_max, 5)

y_label_pos = np.linspace(rows,0, 5, dtype=int)
x_label_pos = np.linspace(0,cols, 5, dtype=int)

#plot EF
plt.figure()
plt.pcolor(EF_im, shading='flat')
plt.gca().invert_yaxis()
plt.xticks(x_label_pos,np.round(lon,2))
plt.yticks(y_label_pos,np.round(lat,2))
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Evaporation fraction')
plt.show()

#plot NDVI histogram
plt.figure()
n, bins, patches = plt.hist(x=NDVI_sorted[NDVI_sorted > -1], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('NDVI')
plt.ylabel('Frequency')
plt.title('NDVI array histogram')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

#plot the LST histogram
plt.figure()
n, bins, patches = plt.hist(x=LST_sorted[LST_sorted > -80], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('$\Delta$ T')
plt.ylabel('Frequency')
plt.title('LST array histogram')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

#==================================== Save the EF ===============================
[cols, rows] = NDVI.shape
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create('Output/EF_'+ date + '.tif', rows, cols, 1, gdal.GDT_Float32)
outdata.SetGeoTransform(fid.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(fid.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(EF_im)
outdata.FlushCache()
