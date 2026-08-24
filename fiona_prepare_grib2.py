
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:08:21 2026

@author: melka

GRIB2 version of the Fiona preprocessing.
Reads HAFS storm-domain GRIB2 (moving nest, already on isobaric levels)
and writes a clean NetCDF on pressure levels, drop-in compatible with the
previous RESTART-derived file. No log-p interpolation needed like the previous file
(RESTART): the GRIB2 is already on fixed pressure levels.
"""

import numpy as np
import xarray as xr
import cfgrib
import time as timer

# Configuration

path00 = 'C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/HAFS_data/hfsa_retro/20220920/00/grib2/'
path06 = 'C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/HAFS_data/hfsa_retro/20220920/06/grib2/'
save_path = 'C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/fiona_data_clean_3D_grib2.nc'
save_dir = 'C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/'

# Each entry: (file path, valid time in hours since 2022-09-20 00Z, label)
# The time axis is UTC, not "hours since model init", because we mix two
# init cycles. 03/06Z come from the 00Z cycle, 09/12/15Z from the 06Z cycle.
FILES = [
    (path00 + '07l.2022092000.hfsa.storm.atm.f003.grb2',  3.0, '20Z00 f003 -> 03 UTC'),
    (path00 + '07l.2022092000.hfsa.storm.atm.f006.grb2',  6.0, '20Z00 f006 -> 06 UTC'),
    (path06 + '07l.2022092006.hfsa.storm.atm.f003.grb2',  9.0, '20Z06 f003 -> 09 UTC'),
    (path06 + '07l.2022092006.hfsa.storm.atm.f006.grb2', 12.0, '20Z06 f006 -> 12 UTC'),
    (path06 + '07l.2022092006.hfsa.storm.atm.f009.grb2', 15.0, '20Z06 f009 -> 15 UTC'),
]

# Pressure range to keep. The GRIB2 provides 150 to 900 hPa at 25 hPa spacing,
# i.e. 31 native levels. We keep them as-is, no interpolation.
P_LOW = 150.0
P_HIGH = 900.0

g = 9.80665  # m/s2
R_earth = 6371.0  # km

# Mapping from cfgrib short names to the names used in the output file.
# u, v wind components, t temperature, q specific humidity,
# gh geopotential height, w vertical velocity in pressure coords (omega, Pa/s).
WANTED = {'u', 'v', 't', 'q', 'gh', 'w'}

import cfgrib
datasets = cfgrib.open_datasets(path00 + '07l.2022092000.hfsa.storm.atm.f003.grb2')
for ds in datasets:
    print(sorted(ds.data_vars), ds.coords.get('typeOfLevel', None))
    
    
#%%
import xarray as xr

ds = xr.open_dataset(
    path00 + '07l.2022092000.hfsa.storm.atm.f003.grb2',
    engine='cfgrib',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}},
)
print(ds['isobaricInhPa'].values)
print('min', float(ds['isobaricInhPa'].min()), 'max', float(ds['isobaricInhPa'].max()))
print('n levels', ds['isobaricInhPa'].size)

#%%
import numpy as np

# ds: the isobaric dataset with 45 levels, reference var is the temperature t
for p in ds['isobaricInhPa'].values:
    field = ds['t'].sel(isobaricInhPa=p).values
    frac = np.isnan(field).mean()
    print(f"{p:7.1f} hPa   NaN {100*frac:5.1f} %")
    
import numpy as np
import matplotlib.pyplot as plt

field = ds['t'].sel(isobaricInhPa=500).values  # one level in the middle of the atmosphere
plt.imshow(np.isnan(field), origin='lower')
plt.title('white = NaN')
plt.colorbar()
plt.show()

print('shape', field.shape)
print('dims', ds['t'].dims)
print('coords', list(ds.coords))

import numpy as np
import xarray as xr

def load_psfc_hPa(path):
    ds_sfc = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': 'sp'}},
    )
    return ds_sfc['sp'].values / 100.0  # Pa -> hPa, on (y, x)

# stack the PSFC of the 5 files -> (5, y, x)
psfc_all = np.stack([load_psfc_hPa(f[0]) for f in FILES], axis=0)

psfc_min = float(np.nanmin(psfc_all))
print(f"min PSFC over the domain and the 5 times : {psfc_min:.1f} hPa\n")

# fraction of underground points per level (level pressure > local PSFC)
levels = ds['isobaricInhPa'].values
for p in levels:
    underground = (p > psfc_all)            # True where the level is below the surface
    frac = np.nanmean(underground)
    flag = '  <-- underground somewhere' if frac > 0 else ''
    print(f"{p:7.1f} hPa   underground {100*frac:5.1f} %{flag}")
    

niveaux_propres = [p for p in levels if p <= psfc_min]
print(f"\nlowest level that is 100% clean : {max(niveaux_propres):.1f} hPa")

#%%
import cfgrib
for d in cfgrib.open_datasets(path00 + '07l.2022092000.hfsa.storm.atm.f003.grb2'):
    tl = d[list(d.data_vars)[0]].attrs.get('GRIB_typeOfLevel', '?')
    if 'depth' in tl.lower() or 'soil' in tl.lower():
        print(tl, sorted(d.data_vars))
print("end of the soil search")
#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def load_lsm(path):
    ds_sfc = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': 'lsm'}},
    )
    return ds_sfc['lsm'].values  # (y, x), 1 = land, 0 = sea

lsm = load_lsm(FILES[0][0])
nan_mask = np.isnan(ds['t'].isel(time=0).sel(isobaricInhPa=500).values)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(lsm, origin='lower')
ax[0].set_title('lsm: light = land')
ax[1].imshow(nan_mask, origin='lower')
ax[1].set_title('light = NaN (corners)')
plt.show()

print('land fraction :', f"{100*np.nanmean(lsm > 0.5):.2f} %")
#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def load_lsm(path):
    ds_sfc = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': 'lsm'}},
    )
    return ds_sfc['lsm'].values

lsm = load_lsm(FILES[0][0])
nan_mask = np.isnan(ds['t'].sel(isobaricInhPa=500).values)

print('shape of the 500hPa field :', nan_mask.shape, '| shape of lsm :', lsm.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(lsm, origin='lower'); ax[0].set_title('lsm: light = land')
ax[1].imshow(nan_mask, origin='lower'); ax[1].set_title('light = NaN')
plt.show()

print('land fraction :', f"{100*np.nanmean(lsm > 0.5):.2f} %")
print('NaN fraction  :', f"{100*np.mean(nan_mask):.2f} %")

#%%
import numpy as np

def largest_inscribed_rectangle(valid):
    # valid : boolean array (ny, nx), True = usable point
    ny, nx = valid.shape
    height = np.zeros(nx, dtype=int)
    best = (0, 0, 0, 0, 0)  # area, y0, y1, x0, x1

    for i in range(ny):
        # height of the full columns up to the line i
        height = np.where(valid[i], height + 1, 0)

        # biggest rectangle under the histogram 'height', keeping the bounds
        stack = []  # (col_start, height)
        j = 0
        while j <= nx:
            cur = height[j] if j < nx else 0
            start = j
            while stack and stack[-1][1] > cur:
                col0, h = stack.pop()
                width = j - col0
                area = width * h
                if area > best[0]:
                    best = (area, i - h + 1, i + 1, col0, j)
                start = col0
            stack.append((start, cur))
            j += 1
    return best  # area, y0, y1, x0, x1

# valid mask common to ALL the levels AND all the times
# (a point is only good if it is non-NaN everywhere)
valid = ~np.isnan(ds['t'].sel(isobaricInhPa=500).values)   # 1 level is enought, the NaN border is the same at all of them

area, y0, y1, x0, x1 = largest_inscribed_rectangle(valid)
print(f"inscribed rectangle : lat[{y0}:{y1}] lon[{x0}:{x1}] -> {y1-y0} x {x1-x0} points, area {area}")

# check zero NaN inside the rectangle, on the whole column
sub = ds['t'].isel(latitude=slice(y0, y1), longitude=slice(x0, x1)).values
print('NaN inside the rectangle (all levels) :', int(np.isnan(sub).sum()))
#%%
import numpy as np
import xarray as xr

def valid_mask(path):
    d = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {
            'typeOfLevel': 'isobaricInhPa',
            'shortName': ['t', 'u', 'v', 'q', 'gh', 'w'],
        }},
    )
    return ~np.isnan(d['t'].sel(isobaricInhPa=500).values)

masks = [valid_mask(f[0]) for f in FILES]
valid_all = np.logical_and.reduce(masks)

# does the current rectangle stay clean on the 5 times ?
y0, y1, x0, x1 = 109, 684, 202, 801
sub = valid_all[y0:y1, x0:x1]
print('invalid points inside the rectangle over the 5 times :', int((~sub).sum()))

# how much does the footprint move between the first and the last one ?
print('mask difference between f003 and f015 :', int((masks[0] != masks[-1]).sum()), 'points')

#%%
import numpy as np
import xarray as xr

def valid_mask(path):
    d = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {
            'typeOfLevel': 'isobaricInhPa',
            'shortName': ['t', 'u', 'v', 'q', 'gh', 'w'],
        }},
    )
    return ~np.isnan(d['t'].sel(isobaricInhPa=500).values)

def centre_fiona(path):
    psfc = load_psfc_hPa(path)                 # my function, in hPa
    iso_nan = ~valid_mask(path)                # True = NaN (border)
    psfc_m = np.where(iso_nan, np.nan, psfc)
    j, i = np.unravel_index(np.nanargmin(psfc_m), psfc_m.shape)
    return j, i, np.nanmin(psfc_m)

# intersection of the 5 times
masks = [valid_mask(f[0]) for f in FILES]
valid_all = np.logical_and.reduce(masks)

# inscribed rectangle on the intersection
area, y0, y1, x0, x1 = largest_inscribed_rectangle(valid_all)
print(f"inscribed rectangle (5 times) : lat[{y0}:{y1}] lon[{x0}:{x1}] -> {y1-y0} x {x1-x0} points")
print('invalid inside the rectangle, 5 times :', int((~valid_all[y0:y1, x0:x1]).sum()))

# check the Fiona center at both ends
for f in [FILES[0], FILES[-1]]:
    j, i, p = centre_fiona(f[0])
    inside = (y0 <= j < y1) and (x0 <= i < x1)
    marge = min(j - y0, y1 - j, i - x0, x1 - i) if inside else -1
    print(f"{f[2]:25s} center ({j},{i}) PSFC {p:.1f}  inside:{inside}  margin:{marge} pts")
    #%%
import numpy as np

psfc = load_psfc_hPa(FILES[0][0])
print('shape   :', psfc.shape)
print('min/max :', np.nanmin(psfc), np.nanmax(psfc), 'hPa')
print('median  :', np.nanmedian(psfc), 'hPa')
#%%


# frozen domain and frozen band
Y0, Y1, X0, X1 = 110, 682, 202, 801
P_LOW, P_HIGH = 150.0, 900.0
PATH = FILES[0][0]            # 03 UTC, single instant
JC, IC = 398, 495            # Fiona center (indices on the full grid)

def load_iso_all(path):
    return xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {
            'typeOfLevel': 'isobaricInhPa',
            'shortName': ['t', 'u', 'v', 'q', 'gh', 'w'],
        }},
    )

d = load_iso_all(PATH)

# selection of the vertical band
levels = d['isobaricInhPa'].values
p_band = levels[(levels >= P_LOW) & (levels <= P_HIGH)]
d = d.sel(isobaricInhPa=p_band).isel(latitude=slice(Y0, Y1), longitude=slice(X0, X1))
print('levels kept :', p_band.min(), 'to', p_band.max(), '|', p_band.size, 'levels')

lat = d['latitude'].values
lon = d['longitude'].values
ny, nx, nlev = lat.size, lon.size, p_band.size

# anchor the frame on the cyclone center, conversion deg -> km (tangent plane)
KM_PER_DEG = 111.32
LAT0 = ds['latitude'].values[JC]
LON0 = ds['longitude'].values[IC]
x_km = (lon - LON0) * KM_PER_DEG * np.cos(np.deg2rad(LAT0))
y_km = (lat - LAT0) * KM_PER_DEG
print(f'anchor : lat0 {LAT0:.2f}  lon0 {LON0:.2f}')
print(f'extent : x [{x_km.min():.0f}, {x_km.max():.0f}] km   y [{y_km.min():.0f}, {y_km.max():.0f}] km')

# target fields, order (nlev, ny, nx)
U  = d['u'].values
V  = d['v'].values
W  = d['w'].values     # omega, Pa/s
GH = d['gh'].values
T  = d['t'].values
Q  = d['q'].values

# coordinate grids, same order
P3 = np.broadcast_to(p_band[:, None, None], (nlev, ny, nx))
Y3 = np.broadcast_to(y_km[None, :, None],  (nlev, ny, nx))
X3 = np.broadcast_to(x_km[None, None, :],  (nlev, ny, nx))

# underground mask : we throw away the points whose pressure exceed the local PSFC
psfc = load_psfc_hPa(PATH)[Y0:Y1, X0:X1]      # (ny, nx) hPa
above = (P3 <= psfc[None, :, :]).ravel()

coords  = np.stack([X3.ravel(), Y3.ravel(), P3.ravel()], axis=1)[above]
targets = np.stack([U.ravel(), V.ravel(), W.ravel(),
                    GH.ravel(), T.ravel(), Q.ravel()], axis=1)[above]

print('total points     :', P3.size)
print('kept points      :', coords.shape[0])
print('underground out  :', P3.size - coords.shape[0],
      f'({100*(1-coords.shape[0]/P3.size):.2f} %)')
print('NaN in coords    :', int(np.isnan(coords).sum()))
print('NaN in targets   :', int(np.isnan(targets).sum()))

#%%
import glob, os

for cycle in ['00', '06']:
    base = f'C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/HAFS_data/hfsa_retro/20220920/{cycle}/grib2/'
    print(f"\n=== cycle {cycle}Z ===")
    if not os.path.isdir(base):
        print('  missing folder :', base)
        continue
    for f in sorted(glob.glob(base + '*.grb2')):
        print('  ', os.path.basename(f))
        
#%%
import numpy as np

# valid mask = non-NaN on the 5 times (already computed : valid_all)
# we add the ocean constraint. lsm barely moves, but to be safe we intersect it on the 5 times too
lsm_masks = [load_lsm(f[0]) <= 0.5 for f in FILES]   # True = ocean
ocean_all = np.logical_and.reduce(lsm_masks)         # ocean at the 5 times
ocean_all = ocean_all & ~np.isnan(load_lsm(FILES[0][0]))  # discards the lsm NaN of the border

# final mask : valid (non-NaN) AND ocean, on the 5 times
good = valid_all & ocean_all

area, y0, y1, x0, x1 = largest_inscribed_rectangle(good)
print(f"ocean rectangle : lat[{y0}:{y1}] lon[{x0}:{x1}] -> {y1-y0} x {x1-x0} points, area {area}")

# checks
print('NaN inside the rectangle :', int((~valid_all[y0:y1, x0:x1]).sum()))
print('land inside the rectangle :', int((~ocean_all[y0:y1, x0:x1]).sum()))

# is the Fiona center (398,495) inside, and with wich margin ?
JC, IC = 398, 495
inside = (y0 <= JC < y1) and (x0 <= IC < x1)
marge = min(JC-y0, y1-JC, IC-x0, x1-IC) if inside else -1
print(f"center ({JC},{IC}) inside:{inside}  margin:{marge} pts")

#%%

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

out_grib = "C:/Users/melka/Downloads/WRF/data/gfs_soil_20220920_00.grib2"
d_soil = cfgrib.open_datasets(out_grib)[0]

# GFS coords : latitude decreasing (90 -> -90), longitude 0..360
gfs_lat = d_soil['latitude'].values            # (721,)
gfs_lon = d_soil['longitude'].values           # (1440,) in 0..360

# 1. fill the NaN (ocean) with the nearest land neighbour, layer by layer
def fill_nan_nearest(a):
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]

st_filled    = np.stack([fill_nan_nearest(d_soil['st'].values[k])    for k in range(4)])
soilw_filled = np.stack([fill_nan_nearest(d_soil['soilw'].values[k]) for k in range(4)])

# 2. target grid : my storm grid on the frozen domain
Y0, Y1, X0, X1 = 110, 682, 202, 801
tgt_lat = ds['latitude'].values[Y0:Y1]         # (571,)
tgt_lon = ds['longitude'].values[X0:X1]        # (599,)

# align the longitude conventions : is the storm one in -180..180 or in 0..360 ?
print('storm lon range :', tgt_lon.min(), tgt_lon.max())
print('gfs   lon range :', gfs_lon.min(), gfs_lon.max())

#%%
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# st_filled, soilw_filled : (4, 721, 1440) already filled (previous step)
# gfs_lat (721, decreasing 90->-90), gfs_lon (1440, 0..360 increasing)
# tgt_lat (572), tgt_lon (599) : my storm grid

# RGI wants increasing axes -> so we flip the GFS latitude
lat_inc = gfs_lat[::-1]                          # increasing -90 -> 90

def interp_layer(field2d):
    f = field2d[::-1, :]                         # follows the latitude flip
    interp = RegularGridInterpolator(
        (lat_inc, gfs_lon), f,
        method='linear', bounds_error=False, fill_value=None,
    )
    LON, LAT = np.meshgrid(tgt_lon, tgt_lat)     # (572, 599)
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=1)
    return interp(pts).reshape(LAT.shape)

st_storm    = np.stack([interp_layer(st_filled[k])    for k in range(4)])  # (4,572,599)
soilw_storm = np.stack([interp_layer(soilw_filled[k]) for k in range(4)])

print('st_storm    :', st_storm.shape,
      'min', f'{st_storm.min():.1f}', 'max', f'{st_storm.max():.1f}', 'K')
print('soilw_storm :', soilw_storm.shape,
      'min', f'{soilw_storm.min():.3f}', 'max', f'{soilw_storm.max():.3f}')
print('NaN st   :', int(np.isnan(st_storm).sum()))
print('NaN soilw:', int(np.isnan(soilw_storm).sum()))
#%%
import numpy as np

Y0, Y1, X0, X1 = 110, 682, 202, 801
lat = ds['latitude'].values[Y0:Y1]
lon = ds['longitude'].values[X0:X1]

# center of the domain
ref_lat = float(lat.mean())
ref_lon = float(lon.mean())
# convert lon 0..360 -> -180..180 for WPS
ref_lon_wps = ref_lon - 360 if ref_lon > 180 else ref_lon

# resolution in degrees then estimation in meters
dlat = float(np.abs(np.diff(lat)).mean())
dlon = float(np.abs(np.diff(lon)).mean())
dx_m = dlon * 111320 * np.cos(np.deg2rad(ref_lat))
dy_m = dlat * 111320

print(f"e_we (lon points) : {lon.size}")
print(f"e_sn (lat points) : {lat.size}")
print(f"ref_lat : {ref_lat:.4f}")
print(f"ref_lon (WPS) : {ref_lon_wps:.4f}")
print(f"dlat {dlat:.4f} deg, dlon {dlon:.4f} deg")
print(f"dx ~ {dx_m:.0f} m, dy ~ {dy_m:.0f} m")
print(f"lat range : {lat.min():.3f} to {lat.max():.3f}")
print(f"lon range (0-360) : {lon.min():.3f} to {lon.max():.3f}")
#%%
def centre_fiona_msl(path):
    d = xr.open_dataset(
        path, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'meanSea', 'shortName': 'prmsl'}},
    )
    mslp = d['prmsl'].values / 100.0
    iso_nan = ~valid_mask(path)
    mslp_m = np.where(iso_nan, np.nan, mslp)
    j, i = np.unravel_index(np.nanargmin(mslp_m), mslp_m.shape)
    return j, i, np.nanmin(mslp_m)

for f in [FILES[0], FILES[-1]]:
    j, i, p = centre_fiona_msl(f[0])
    inside = (110 <= j < 682) and (202 <= i < 801)
    marge = min(j-110, 682-j, i-202, 801-i) if inside else -1
    print(f"{f[2]:25s} center ({j},{i}) MSLP {p:.1f} hPa  inside:{inside}  margin:{marge}")
#%%


def load_grib_isobaric(path, p_low, p_high):
    """
    Open one HAFS storm GRIB2 file, keep the isobaric fields we need,
    restrict to [p_low, p_high] and return them sorted with ascending
    pressure and ascending latitude.

    Returns:
        fields: dict name -> ndarray (nlev, ny, nx)
        levels: ndarray (nlev,) pressure in hPa, ascending
        lat: ndarray (ny,) ascending
        lon: ndarray (nx,)
    """
    datasets = cfgrib.open_datasets(path, backend_kwargs={'indexpath': ''})

    fields = {}
    levels = None
    lat = None
    lon = None

    for ds in datasets:
        if 'isobaricInhPa' not in ds.dims and 'isobaricInhPa' not in ds.coords:
            continue
        for name in list(ds.data_vars):
            if name in WANTED and name not in fields:
                da = ds[name]
                if 'isobaricInhPa' not in da.dims:
                    continue
                da = da.sortby('isobaricInhPa')
                da = da.sel(isobaricInhPa=slice(p_low, p_high))
                if 'latitude' in da.coords:
                    da = da.sortby('latitude')
                fields[name] = da.values.astype(np.float32)
                if levels is None:
                    levels = da['isobaricInhPa'].values.astype(np.float32)
                    lat = da['latitude'].values.astype(np.float64)
                    lon = da['longitude'].values.astype(np.float64)

    missing = WANTED - set(fields)
    if missing:
        raise ValueError(f'Missing variables in {path}: {sorted(missing)}')

    return fields, levels, lat, lon


def main():
    nt = len(FILES)

    all_u = all_v = all_T = all_q = None
    all_z = all_geopot = all_omega = None
    all_lat = all_lon = None

    levels_ref = None
    ny = nx = None
    x_km = y_km = None

    time_hours = np.array([t for (_, t, _) in FILES], dtype=np.float32)

    for t_idx, (fpath, t_hr, label) in enumerate(FILES):
        print()
        print(f'Processing t = {t_hr:.0f}h UTC   ({label})')
        t0 = timer.time()

        fields, levels, lat, lon = load_grib_isobaric(fpath, P_LOW, P_HIGH)
        n_lev = len(levels)

        # First pass: fix grid and allocate
        if all_u is None:
            ny, nx = lat.size, lon.size
            levels_ref = levels

            all_u = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_v = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_T = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_q = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_z = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_geopot = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_omega = np.full((nt, n_lev, ny, nx), np.nan, dtype=np.float32)
            all_lat = np.full((nt, ny), np.nan, dtype=np.float32)
            all_lon = np.full((nt, nx), np.nan, dtype=np.float32)

            # Storm-relative km grid, centered on the domain center.
            # Spacing is constant across the nest and across time (the nest
            # config is fixed, only its center moves), so we derive it once.
            dlat = float(np.abs(np.mean(np.diff(lat))))
            dlon = float(np.abs(np.mean(np.diff(lon))))
            lat0 = float(np.mean(lat))
            deg2km = np.pi / 180.0 * R_earth
            y_km = (np.arange(ny) - (ny - 1) / 2.0) * dlat * deg2km
            x_km = (np.arange(nx) - (nx - 1) / 2.0) * dlon * deg2km * np.cos(np.deg2rad(lat0))
            print(f'  Grid {ny} x {nx}, {n_lev} levels {levels[0]:.0f} to {levels[-1]:.0f} hPa')
            print(f'  Spacing ~ {dlat * deg2km:.2f} km (y), {dlon * deg2km * np.cos(np.deg2rad(lat0)):.2f} km (x) at lat0={lat0:.2f}')
        else:
            if (lat.size, lon.size) != (ny, nx):
                raise ValueError(
                    f'Inconsistent grid at t={t_hr}h: {(lat.size, lon.size)} '
                    f'vs {(ny, nx)} expected. Maybe the nest has a different size.'
                )
            if not np.allclose(levels, levels_ref):
                raise ValueError(f'Inconsistent pressure levels at t={t_hr}h.')

        all_u[t_idx] = fields['u']
        all_v[t_idx] = fields['v']
        all_T[t_idx] = fields['t']
        all_q[t_idx] = fields['q']
        all_z[t_idx] = fields['gh']
        all_omega[t_idx] = fields['w']
        all_geopot[t_idx] = g * fields['gh']

        # store absolute georeference (lon mapped to -180..180 for readability)
        all_lat[t_idx] = lat.astype(np.float32)
        all_lon[t_idx] = (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)

        ws = np.sqrt(all_u[t_idx] ** 2 + all_v[t_idx] ** 2)
        idx_850 = int(np.argmin(np.abs(levels - 850.0)))
        vmax_850 = np.nanmax(ws[idx_850])
        print(f'  Done in {timer.time() - t0:.0f}s. '
              f'Vmax at 850 hPa = {vmax_850:.1f} m/s (~{vmax_850 * 1.94:.0f} kt)')

    wind_speed = np.sqrt(all_u ** 2 + all_v ** 2)
    valid_time = (np.datetime64('2022-09-20T00:00:00')
                  + (time_hours.astype('timedelta64[h]')))

    # Save to NetCDF
    print()
    print(f'Saving to {save_path} ...')

    ds_out = xr.Dataset(
        {
            'u': (['time', 'pressure', 'y', 'x'], all_u,
                  {'units': 'm/s', 'long_name': 'Zonal wind'}),
            'v': (['time', 'pressure', 'y', 'x'], all_v,
                  {'units': 'm/s', 'long_name': 'Meridional wind'}),
            'T': (['time', 'pressure', 'y', 'x'], all_T,
                  {'units': 'K', 'long_name': 'Temperature'}),
            'q': (['time', 'pressure', 'y', 'x'], all_q,
                  {'units': 'kg/kg', 'long_name': 'Specific humidity'}),
            'z': (['time', 'pressure', 'y', 'x'], all_z,
                  {'units': 'm', 'long_name': 'Geopotential height'}),
            'geopotential': (['time', 'pressure', 'y', 'x'], all_geopot,
                             {'units': 'm2/s2', 'long_name': 'Geopotential (g*z)'}),
            'omega': (['time', 'pressure', 'y', 'x'], all_omega,
                      {'units': 'Pa/s',
                       'long_name': 'Vertical velocity in pressure coords (model VVEL)'}),
            'wind_speed': (['time', 'pressure', 'y', 'x'], wind_speed,
                           {'units': 'm/s', 'long_name': 'Horizontal wind speed'}),
            'latitude': (['time', 'y'], all_lat,
                         {'units': 'degrees_north', 'long_name': 'Latitude per time'}),
            'longitude': (['time', 'x'], all_lon,
                          {'units': 'degrees_east', 'long_name': 'Longitude per time'}),
        },
        coords={
            'time_hours': ('time', time_hours,
                           {'units': 'hours since 2022-09-20 00:00 UTC',
                            'long_name': 'Valid time'}),
            'valid_time': ('time', valid_time,
                           {'long_name': 'Valid time (UTC)'}),
            'pressure_hPa': ('pressure', levels_ref,
                             {'units': 'hPa', 'long_name': 'Pressure level',
                              'positive': 'down'}),
            'x_km': ('x', x_km.astype(np.float32),
                     {'units': 'km', 'long_name': 'Storm-relative X distance'}),
            'y_km': ('y', y_km.astype(np.float32),
                     {'units': 'km', 'long_name': 'Storm-relative Y distance'}),
        },
        attrs={
            'title': 'Hurricane Fiona - HAFS storm-domain GRIB2 on pressure levels',
            'source': 'HAFS hfsa_retro storm.atm GRIB2, cycles 20220920 00Z and 06Z',
            'time_axis': 'UTC. 03/06Z from 00Z cycle, 09/12/15Z from 06Z cycle',
            'pressure_levels': f'{len(levels_ref)} native levels from {P_LOW:.0f} to {P_HIGH:.0f} hPa',
            'note_omega': 'omega is the model VVEL, usable as reference for the omega_c scheme',
            'note_grid': 'x_km/y_km are storm-relative; latitude/longitude give the absolute georeference per time',
            'reference': 'Eusebi et al. (2024), Comm. Earth Environ., doi:10.1038/s43247-023-01144-2',
        },
    )

    encoding = {}
    for var in ['u', 'v', 'T', 'q', 'z', 'geopotential', 'omega', 'wind_speed']:
        encoding[var] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}

    ds_out.to_netcdf(save_path, encoding=encoding)

    raw_mb = (all_u.nbytes + all_v.nbytes + all_T.nbytes + all_q.nbytes
              + all_z.nbytes + all_geopot.nbytes + all_omega.nbytes
              + wind_speed.nbytes) / 1e6
    print()
    print(f'Done. File saved: {save_path}')
    print(f'  Shape (time={nt}, pressure={len(levels_ref)}, y={ny}, x={nx})')
    print(f'  Raw data size {raw_mb:.0f} MB (compressed in file)')
    print(f'  Times (UTC hours since 20Z00): {list(time_hours)}')

    # Quick sanity check plot
    try:
        import matplotlib.pyplot as plt

        # show 12 UTC (case 115 passage at 12:13) if available, else first time
        t_show = 3 if nt > 3 else 0
        t_label = f'{time_hours[t_show]:.0f} UTC'

        idx_850 = int(np.argmin(np.abs(levels_ref - 850.0)))
        idx_500 = int(np.argmin(np.abs(levels_ref - 500.0)))
        idx_200 = int(np.argmin(np.abs(levels_ref - 200.0)))

        X, Y = np.meshgrid(x_km, y_km)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, p_idx, p_val in zip(axes, [idx_850, idx_500, idx_200], [850, 500, 200]):
            ws_slice = wind_speed[t_show, p_idx]
            cf = ax.contourf(X, Y, ws_slice, levels=20, cmap='jet', vmin=0, vmax=70)
            ax.set_title(f't={t_label}, {p_val} hPa\nVmax = {np.nanmax(ws_slice):.1f} m/s')
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_aspect('equal')
            plt.colorbar(cf, ax=ax, shrink=0.8)

        fig.suptitle('Hurricane Fiona - Sanity Check (GRIB2, 3 levels)', fontweight='bold')
        fig.tight_layout()
        fig.savefig(save_dir + 'fiona_3d_sanity_check_grib2.png', dpi=150, bbox_inches='tight')
        print('  Sanity-check figure saved.')
        plt.show()
    except Exception as e:
        print(f'  (Skipped sanity plot: {e})')


if __name__ == '__main__':
    main()