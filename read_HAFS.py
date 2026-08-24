# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 22:49:58 2026

@author: melka
"""
"""Example of how to read the 'HAFS data'(NetCFD files); and of listing all the fields """
from netCDF4 import Dataset
PATH_D01 = r"C:\Users\melka\Downloads\WRF\fiona21h\wrfout_d01_2022-09-20_15-00-00"
nc = Dataset(PATH_D01)
print(nc.dimensions)
print(nc.getncattr("DX"), nc.getncattr("MAP_PROJ"))

for name, var in nc.variables.items():
    print(f"{name:12s} {str(var.dimensions):55s} {getattr(var, 'units', '')}")

times = nc.variables["Times"][:]
print(b"".join(times[0].data).decode())