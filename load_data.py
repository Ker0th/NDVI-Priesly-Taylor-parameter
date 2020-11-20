# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:50:04 2020

@author: Glogta
"""

# import tarfile
# my_tar = tarfile.open('order_121522.tar')
# my_tar.extractall('./MDFVC') # specify which folder to extract to
# my_tar.close()


# with open('./LSA/HDF5_LSASAF_MSG_LST_MSG-Disk_201409110000') as f:
#     read_data = f.read()


from PIL import Image
from PIL.TiffTags import TAGS

NDVI_input = 'C:/Uni/9._Semester_Speciale_course_data/Coordinated_NVDI'
with Image.open(NDVI_input + "/HDF5_LSASAF_MSG_FVC_MSG-Disk_201409110000_FVC_warped.tif") as img:
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}

im = Image.open(NDVI_input + "/HDF5_LSASAF_MSG_FVC_MSG-Disk_201409110000_FVC_warped.tif")
img.show()



im_np = np.array(im)
# %%
#=============================== Test how to load the bands ===========================================
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('C:/Users/Glogta/OneDrive/Uni_current/Special_project/functions')
from ReadGeoTif import *
from scope_tiff import *
from osgeo import gdal,osr
#filename = 'C:/Uni/9._Semester_Speciale_course_data/subset_collocat/'
#collocate= gdal.Open('Data/LST_collocate_20201012.tif')

#band = collocate.GetRasterBand(1)
#LST = band.ReadAsArray()

data_dir = 'Data/NDVI_LST_07_11.tif'

#=============================== LST ==============================================

LST_full = ReadGeoTif(data_dir, 2)
LST_lat = LST_full[0]
LST_long = LST_full[1]
LST = LST_full[2]

plt.figure()
plt.imshow(LST, extent = [LST_long.min(), LST_long.max(), LST_lat.min(), LST_lat.max()])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show()

#=============================== NDVI =============================================

NDVI_full = ReadGeoTif(data_dir, 1)
NDVI_lat = NDVI_full[0]
NDVI_long = NDVI_full[1]
NDVI = NDVI_full[2]

plt.figure()
plt.imshow(NDVI, extent = [NDVI_long.min(), NDVI_long.max(), NDVI_lat.min(), NDVI_lat.max()])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show()

#=============================== CLM =============================================
CLM = ((NDVI > 0) & (NDVI != np.max(NDVI)) & (LST != -80))*1

plt.figure()
plt.imshow(CLM, extent = [NDVI_long.min(), NDVI_long.max(), NDVI_lat.min(), NDVI_lat.max()])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#============================== Save the files ==================================
from PIL import Image
im = Image.fromarray(NDVI)
im.save('Data/NDVI.tif')

im = Image.fromarray(LST)
im.save('Data/LST.tif')

im = Image.fromarray(CLM)
im.save('Data/CLM.tif')

# %%

lat = np.array([LST_lat.min(), LST_lat.max()])
long = np.array([LST_long.min(), LST_long.max()])
filename = "C:/Uni/9._Semester_Speciale_course_data/MSG_07_11/Corrected/HDF5_LSASAF_MSG_LST_MSG-Disk_202011071200_LST_warped.tif"


NDVI_scope = scope_tiff(filename, lat, long)
plt.figure()
plt.imshow(NDVI_scope, extent=[long[0], long[1], lat[0], lat[1]])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.show()
#================================================================================================
# %%
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('C:/Users/Glogta/OneDrive/Uni_current/Special_project/functions')
from ReadGeoTif import *
from scope_tiff import *

NDVI_input = 'C:/Uni/9._Semester_Speciale_course_data/Coordinated_NVDI'
LST_input = 'C:/Uni/9._Semester_Speciale_course_data/Coordinated_LST'

lat = np.array([54, 57])
long = np.array([8, 12])

lat = np.array([LST_lat.min(), LST_lat.max()])
long = np.array([LST_long.min(), LST_long.max()])
#=============================== NDVI =============================================
filename = "Data/NDVI_collocate_20200711.tif"



NDVI_scope = scope_tiff(filename, lat, long)

plt.figure()
plt.imshow(NDVI_scope, extent=[long[0], long[1], lat[0], lat[1]])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#=============================== LST ==============================================
filename_LST = "Data/LSTdiff_collocate_20200711.tif"

LST_scope = scope_tiff(filename_LST, lat, long)

plt.figure()
plt.imshow(LST_scope, extent=[long[0], long[1], lat[0], lat[1]])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
#=============================== CLM =============================================
CLM_scope = ((NDVI_scope > 0) & (LST_scope != -80))*1

plt.figure()
plt.imshow(CLM_scope, extent=[long[0], long[1], lat[0], lat[1]])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
#============================== Save the files ==================================
from PIL import Image
im = Image.fromarray(NDVI_scope)
im.save('Data/NDVI_scope.tif')

im = Image.fromarray(LST_scope)
im.save('Data/LST_scope.tif')

im = Image.fromarray(CLM_scope)
im.save('Data/CLM_scope.tif')

#=================================== check we got good data ======================
