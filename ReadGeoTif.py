# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 09:08:08 2015

@author: ssim
"""



def ReadGeoTif(filename):
    from osgeo import gdal,osr
    import numpy as np
    
    # get the existing coordinate system
    ds = gdal.Open(filename)
    old_cs= osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())
    Driver = ds.GetDriver().ShortName,'/', ds.GetDriver().LongName
    Size   = ds.RasterXSize,'x',ds.RasterYSize, 'x',ds.RasterCount
    Proj   = ds.GetProjection()
    inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
    inSRS_converter.ImportFromWkt(Proj)  # populates the spatial ref object with our WKT SRS
    inSRS_forPyProj = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
     
    print('Driver: ', Driver)
    print( 'Size is ', Size)
    print( 'Projection is ', Proj)
        
    geotransform = ds.GetGeoTransform()
    if not geotransform is None:
        Origin = (geotransform[0], geotransform[3])
        PxSize = (geotransform[1], geotransform[5])
        print( 'Origin = ', Origin)
        print( 'Pixel Size', PxSize)
    
    #Nx = ds.RasterXSize
    #Ny = ds.RasterYSize
    band = ds.GetRasterBand(1)
    
    #print 'Band Type=',gdal.GetDataTypeName(band.DataType)
    
    min = band.GetMinimum()
    max = band.GetMaximum()
    if min is None or max is None:
        (min,max) = band.ComputeRasterMinMax(1)
    print( 'Min=%.3f, Max=%.3f' % (min,max))
    
    if band.GetOverviewCount() > 0:
        print( 'Band has ', band.GetOverviewCount(), ' overviews.')
    
    if not band.GetRasterColorTable() is None:
        print( 'Band has a color table with ', band.GetRasterColorTable().GetCount(), ' entries.')
    
    
    elevation = band.ReadAsArray()
    nrows, ncols = elevation.shape
    lat = np.zeros(elevation.shape)
    lon = np.zeros(elevation.shape)
    
    #shift the origin by adding geotransform[0] to u and geotransform[3] to v.
    #By the way, the values of the constants a, b, c, d are given by the 1, 2, 4, and 5 entries in the geotransform array.
    x, a, b, y, c, d = ds.GetGeoTransform()
    for ii in np.arange(0,ncols):
        for jj in np.arange(0,nrows):
            u = a*ii + b*jj
            v = c*ii + d*jj
            lon[jj,ii] = geotransform[0] + u
            lat[jj,ii] = geotransform[3] + v
    return  lat, lon, elevation, inSRS_forPyProj 
