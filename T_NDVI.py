# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:45:58 2020

@author: Anders Slotsbo
"""
import sys
sys.path.append('C:/Users/Glogta/OneDrive/Uni_current/Special project/pyTVDI-master')
import pyTVDI as tvdi

def T_NDVI(NDVI_path, LST_path, CLM_path, output_path):
    """
    DESCRIPTION

    Parameters
    ----------
    NDVI_path : TYPE
        DESCRIPTION.
    LST_path : TYPE
        DESCRIPTION.
    CLM_path : TYPE
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    roi_inf = {'geoflag': tvdi.cWHOLE_IMG,
               'x_pix': 0, 'y_pix': 0,
               'dim_cols': 0, 'dim_rows': 0,
               'moving_window': tvdi.cNO_WINDOW,
               'window_size_x': 0,
               'window_size_y': 0}
    
    alg_inf = {'dry_edge': tvdi.cTANG,
               'ts_min': tvdi.cMEAN,
               'output': tvdi.cTVDI,
               'dry_edge_params': [float(0.01), float(0.1)],
               # [ndvi_step, ndvi_lower_limit]
               'ts_min_params': 20,  # see constants file for explanation
               'ts_min_file': '',
               'output_params': [0.1, 0.9, 1.26, tvdi.cLINEAR]}
                                # [min_ndvi, max_ndvi, max_fi, interpolation]
    # default io options
    io_inf = {'ndvi_file': NDVI_path,
              'ts_file': LST_path,
              'CLM_file': CLM_path,
              'delta_file': '',
              'output_dir': output_path,
              'ndvi_mult': 1,
              'ts_mult': 1,
              'delta_mult': 1}
    
    
    output=tvdi.tvdi(io_inf, roi_inf, alg_inf)
    return output
