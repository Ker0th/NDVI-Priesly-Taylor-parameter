# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:47:01 2020

@author: Glogta
"""

from osgeo import gdal,osr
import numpy as np

def load_data(date):
    """
    Loads the data for T_NDVI, seperate and create masks for the data, furthermore it also saves the files as seperate images
    and exports the projection so that they can be plotted.

    Parameters
    ----------
    date : string
        the date of the the data exported via SNAP in the format DD_MM_YY

    Returns
    -------
    4 Geotiff images if the input file has 3 bands it fills in the ERA5 data as well

    """
    # ======================== Load data =====================================
    data_dir = 'Data/NDVI_LST_'+ date +'.tif'
    ds = gdal.Open(data_dir)
    if ds.RasterCount == 3:
        ERA5 = ds.GetRasterBand(1).ReadAsArray()
        NDVI = ds.GetRasterBand(2).ReadAsArray()
        LST = ds.GetRasterBand(3).ReadAsArray()
    else:
        NDVI = ds.GetRasterBand(1).ReadAsArray()
        LST = ds.GetRasterBand(2).ReadAsArray()
    
    [cols, rows] = NDVI.shape
    driver = gdal.GetDriverByName("GTiff")
    
    
    #=========================== mask data ==================================
    CLM = ((NDVI > 0) & (NDVI != np.max(NDVI)) & (LST != -80))*1
    LST[CLM == 0] = -80
    NDVI[CLM == 0] = -1
    
    
    #==================== Save as seperate Geotiff ==========================
    outdata = driver.Create('Data/NDVI_'+ date + '.tif', rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(NDVI)
    #outdata.GetRasterBand(1).SetNoDataValue(-1)
    outdata.FlushCache()

    outdata = driver.Create('Data/LST_'+ date + '.tif', rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(LST)
    #outdata.GetRasterBand(1).SetNoDataValue(-80)
    outdata.FlushCache()
    
    outdata = driver.Create('Data/ERA5_'+ date + '.tif', rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    #outdata.GetRasterBand(1).WriteArray(ERA5)
    outdata.FlushCache()
    
    outdata = driver.Create('Data/CLM_'+ date + '.tif', rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(CLM)
    #outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache()
    