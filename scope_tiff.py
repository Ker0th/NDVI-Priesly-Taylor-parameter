# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:18:33 2020

@author: Anders Slotsbo
"""
from ReadGeoTif import *
import numpy as np

def scope_tiff(filename, lat=[-90, 90], long=[0, 360]):
    """
    

    Parameters
    ----------
    filename : str
        The path for the file.
    lat : 1x2 array, optional
        The latitude bound for the scope. The default is [-90, 90].
    long : 1x2 array, optional
        The longitude bound for the scope. The default is [0, 360].

    Returns
    -------
    NDVI_scope : NxM numpy matrix
        The focused scope of the image.

    """
    
    #Load the data
    NDVI = ReadGeoTif(filename)
    
    #create the mask for the scope
    mask_lat = (NDVI[0] >= lat[0]) & (NDVI[0] <= lat[1])
    mask_lat = mask_lat.astype(int)
    mask_long = (NDVI[1] >= long[0]) & (NDVI[1] <= long[1])
    mask_long = mask_long.astype(int)
    mask = mask_long + mask_lat == 2
    
    
    #find the index for the scope
    idx_y = np.where(mask)[0]
    idx_x = np.where(mask)[1]
    
    #isolate the scope
    NDVI_scope = NDVI[2][min(idx_y):max(idx_y),min(idx_x):max(idx_x)]

    return NDVI_scope

