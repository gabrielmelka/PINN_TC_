"""Build the WPS input npz for the WRF PINN run (13 lead times, f006 to f042).
The initial time (12 UTC, f006) has its core replaced by the PINN/HAFS blend
on u, v, gh, T, q. All other lead times (for the boundary conditions) remain 
100% HAFS parent.

I'm running it on my PC usually on the PC (cfgrib + torch), then transfer the npz to Tiger3.
"""

import numpy as np
import xarray as xr
import cfgrib
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

## Configuration [TO BE REPLACED WITH YOUR OWN PATHS]

PARENT = "C:/Users/melka/Downloads/Stage_M1/code_PINN_like_eusebi/HAFS_data/hfsa_retro/20220920/06/parent/"
SOIL_GRIB = "C:/Users/melka/Downloads/WRF/data/gfs_soil_20220920_00.grib2"
MODEL_PATH = 'C:/Users/melka/Downloads/pinn_fiona_3D_Tq_seq_20260813_021603_end_phase1.pt'
OUT_NPZ    = "C:/Users/melka/Downloads/WRF/data/wps_input_hybrid15_g.npz"
FILES = [
    (PARENT + "07l.2022092006.hfsa.parent.atm.f009.grb2", "2022-09-20_15:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f012.grb2", "2022-09-20_18:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f015.grb2", "2022-09-20_21:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f018.grb2", "2022-09-21_00:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f021.grb2", "2022-09-21_03:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f024.grb2", "2022-09-21_06:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f027.grb2", "2022-09-21_09:00:00"),
    (PARENT + "07l.2022092006.hfsa.parent.atm.f030.grb2", "2022-09-21_12:00:00"),
]


##
INJECT_DATE = "2022-09-20_15:00:00"     # time at which the PINN is injected
INJECT_HOUR = 15.0                      # PINN time_hours
EYE_LAT = 22.0
EYE_LON = -71.5
assert INJECT_DATE in [d for _, d in FILES], "INJECT_DATE absent de FILES"
# wide crop (covers a d01 extended northward up to ~36N)
LON_MIN, LON_MAX = -84.0, -58.0
LAT_MIN, LAT_MAX = 12.0, 38.0

# PINN window (storm-relative km) and ramps
X_MIN, X_MAX = -300.0, 300.0
Y_MIN, Y_MAX = -300.0, 300.0
L_EDGE = 100.0
P_TOP_HI, P_TOP_LO = 200.0, 150.0
P_BOT_HI, P_BOT_LO = 850.0, 900.0

G = 9.80665
KM_DEG = 111.32
device = torch.device("cpu")
# PINN architecture

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, m, sigma):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, m) * sigma)

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN3D(nn.Module):
    def __init__(self, n_hidden=100, n_layers=4, sigma_ffe=0.0, m_ffe=64):
        super().__init__()
        self.use_ffe = sigma_ffe > 0
        if self.use_ffe:
            self.ffe = FourierFeatures(4, m_ffe, sigma_ffe)
            in_dim = 2 * m_ffe + 4
        else:
            in_dim = 4
        layers = [nn.Linear(in_dim, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x, y, p):
        inp = torch.stack([t, x, y, p], dim=-1)
        if self.use_ffe:
            inp = torch.cat([self.ffe(inp), inp], dim=-1)
        out = self.net(inp)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


# weights

def w_spatial(x_km, y_km):
    dx = np.minimum(x_km - X_MIN, X_MAX - x_km)
    dy = np.minimum(y_km - Y_MIN, Y_MAX - y_km)
    return np.clip(np.minimum(dx, dy) / L_EDGE, 0.0, 1.0)


def w_pressure(p_hpa):
    wt = np.clip((p_hpa - P_TOP_LO) / (P_TOP_HI - P_TOP_LO), 0.0, 1.0)
    wb = np.clip((P_BOT_LO - p_hpa) / (P_BOT_LO - P_BOT_HI), 0.0, 1.0)
    return np.minimum(wt, wb)


# PINN loading

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
cfg = ckpt["config"]
nrm = ckpt["normalization"]
phys = ckpt["physics_constants"]
sigma_ffe = cfg.get("sigma_ffe", 0.0)
m_ffe = cfg.get("m_ffe") or 64
model = PINN3D(cfg["n_hidden"], cfg["n_layers"], sigma_ffe, m_ffe).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"PINN charge : {cfg['n_layers']}x{cfg['n_hidden']}, FFE sigma={sigma_ffe}")


def normalise(t, x, y, p):
    tn = 2 * (t - nrm["t_min"]) / (nrm["t_max"] - nrm["t_min"] + 1e-12) - 1
    xn = 2 * (x - nrm["x_min"]) /(nrm["x_max"] - nrm["x_min"] + 1e-12) - 1
    yn = 2 * (y - nrm["y_min"]) / (nrm["y_max"] - nrm["y_min"] + 1e-12) - 1
    pn = 2 * (p - nrm["p_min"]) / (nrm["p_max"] - nrm["p_min"] + 1e-12) - 1
    return tn, xn, yn, pn


def pinn_eval(x_m, y_m, p_pa):
    tn, xn, yn, pn = normalise(np.full_like(x_m, INJECT_HOUR * 3600.0),
                               x_m, y_m, p_pa)
    with torch.no_grad():
        un, vn, wn, hn, Tn, qn = model(
            torch.tensor(tn, dtype=torch.float32),
            torch.tensor(xn, dtype=torch.float32),
            torch.tensor(yn, dtype=torch.float32),
            torch.tensor(pn, dtype=torch.float32))
    u = un.numpy() * nrm["u_0"] + nrm["u_mean"]
    v = vn.numpy() * nrm["u_0"] + nrm["v_mean"]
    gh = (hn.numpy() * nrm["phi_0"] + nrm["phi_mean"]) /  G     # m2/s2 -> m
    T = Tn.numpy() * nrm["T_0"] + nrm["T_mean"]
    q = qn.numpy() * nrm["q_0"] + nrm["q_mean"]
    return u, v, gh, T, q


# crop indices (grid shared by all lead times)
d0 = xr.open_dataset(FILES[0][0], engine="cfgrib",
                     backend_kwargs={"indexpath": "",
                                     "filter_by_keys": {"typeOfLevel": "meanSea"}})
lat_full = d0.latitude.values
lon_full = d0.longitude.values
d0.close()
lon180 = np.where(lon_full > 180.0, lon_full - 360.0, lon_full)
jlat = np.where((lat_full >= LAT_MIN) & (lat_full <= LAT_MAX))[0]
ilon = np.where((lon180 >= LON_MIN) & (lon180 <= LON_MAX))[0]
lat_c = lat_full[jlat]
if lat_c[0] > lat_c[-1]:
    jlat = jlat[::-1]; lat_c = lat_full[jlat]
lon_c = lon180[ilon]
if lon_c[0] > lon_c[-1]:
    ilon = ilon[::-1]; lon_c = lon180[ilon]
print(f"crop : {jlat.size} lat x {ilon.size} lon")


def read3d(path, short):
    d = xr.open_dataset(path, engine="cfgrib",
                        backend_kwargs={"indexpath": "",
                                        "filter_by_keys": {"typeOfLevel": "isobaricInhPa",
                                                           "shortName": short}})
    a = d[short].values
    lev = d.isobaricInhPa.values
    d.close()
    return a[:, jlat][:, :, ilon], lev


def read_sfc(path, tl, short):
    d = xr.open_dataset(path, engine="cfgrib",
                        backend_kwargs={"indexpath": "",
                                        "filter_by_keys": {"typeOfLevel": tl,
                                                           "stepType": "instant",
                                                           "shortName": short}})
    a = d[short].values
    d.close()
    return a[jlat][:, ilon]


def read_meansea(path, short):
    d = xr.open_dataset(path, engine="cfgrib",
                        backend_kwargs={"indexpath": "",
                                        "filter_by_keys": {"typeOfLevel": "meanSea",
                                                           "shortName": short}})
    a = d[short].values
    d.close()
    return a[jlat][:, ilon]


def build_soil():
    ds = cfgrib.open_datasets(SOIL_GRIB)[0]
    glat = ds["latitude"].values
    glon = ds["longitude"].values
    st = ds["st"].values
    sw = ds["soilw"].values
    ds.close()

    def fill(a):
        m = np.isnan(a)
        if not m.any():
            return a
        idx = distance_transform_edt(m, return_distances=False, return_indices=True)
        return a[tuple(idx)]

    st = np.stack([fill(st[k]) for k in range(4)])
    sw = np.stack([fill(sw[k]) for k in range(4)])
    lat_inc = glat[::-1]
    tgt_lon360 = np.where(lon_c < 0, lon_c + 360.0, lon_c)

    def interp(field):
        rgi = RegularGridInterpolator((lat_inc, glon), field[::-1, :],
                                      method="linear", bounds_error=False, fill_value=None)
        LON, LAT = np.meshgrid(tgt_lon360, lat_c)
        return rgi(np.stack([LAT.ravel(), LON.ravel()], 1)).reshape(LAT.shape)

    return (np.stack([interp(st[k]) for k in range(4)]),
            np.stack([interp(sw[k]) for k in range(4)]))


print("sol GFS...")
ST, SM = build_soil()


def inject_pinn(tt, uu, vv, gh, qv, levels):
    """Replace the core by the PINN/HAFS blend on u, v, gh, T, q (sub-box)."""
    ii = np.where(np.abs(lat_c - EYE_LAT) < 3.3)[0]
    jj = np.where(np.abs(lon_c - EYE_LON) < 3.6)[0]
    kk = np.where((levels >= 150.0) & (levels <= 900.0))[0]
    LAT = lat_c[ii]; LON = lon_c[jj]; LEV = levels[kk]

    P3, Y3, X3 = np.meshgrid(LEV, LAT, LON, indexing="ij")
    x_km = (X3 - EYE_LON) * KM_DEG * np.cos(np.deg2rad(EYE_LAT))
    y_km = (Y3 - EYE_LAT) * KM_DEG
    up, vp, ghp, Tp, qp = pinn_eval((x_km * 1e3).ravel(),
                                    (y_km * 1e3).ravel(),
                                    (P3 * 100.0).ravel())
    shp = P3.shape
    up = up.reshape(shp); vp = vp.reshape(shp); ghp = ghp.reshape(shp)
    Tp = Tp.reshape(shp); qp = qp.reshape(shp)

    W = w_spatial(x_km, y_km) * w_pressure(P3)     # (nk, ni, nj)

    sel = np.ix_(kk, ii, jj)
    tt[sel] = W * Tp + (1 - W) * tt[sel]
    uu[sel] = W * up + (1 - W) * uu[sel]
    vv[sel] = W * vp + (1 - W) * vv[sel]
    gh[sel] = W * ghp + (1 - W) * gh[sel]
    qv[sel] = W * qp + (1 - W) * qv[sel]
    print(f"  PINN injecte : sous-boite {shp}, w_max={W.max():.2f}")
    return tt, uu, vv, gh, qv


records = []
levels = None
for path, date in FILES:
    tt, levels = read3d(path, "t")
    uu, _ = read3d(path, "u")
    vv, _ = read3d(path, "v")
    gh, _ = read3d(path, "gh")
    qv, _ = read3d(path, "q")

    if date == INJECT_DATE:
        tt, uu, vv, gh, qv = inject_pinn(tt, uu, vv, gh, qv, levels)

    rec = {
        "date": date, "p_levels": levels,
        "TT": tt, "UU": uu, "VV": vv, "GHT": gh, "QV": qv,
        "PSFC": read_sfc(path, "surface", "sp"),
        "PMSL": read_meansea(path, "prmsl"),
        "SKINTEMP": read_sfc(path, "surface", "t"),
        "SST": read_sfc(path, "surface", "sst"),
        "LANDSEA": read_sfc(path, "surface", "lsm"),
        "SOILHGT": read_sfc(path, "surface", "orog"),
        "ST": ST, "SM": SM,
    }
    records.append(rec)
    print(f"{date} : 3D {tt.shape}")

np.savez(OUT_NPZ, records=records, lat=lat_c, lon_wps=lon_c,
         soil_top=np.array([0., 10., 40., 100.]),
         soil_bot=np.array([10., 40., 100., 200. ]),
         allow_pickle=True)
print("\nsauve :", OUT_NPZ)
print("t_min/t_max :", nrm["t_min"], nrm["t_max"])
print("x_min/x_max :", nrm["x_min"], nrm["x_max"])
print("p_min/p_max :", nrm["p_min"], nrm["p_max"])
tn_check = 2 * (INJECT_HOUR * 3600.0 - nrm["t_min"]) / (nrm["t_max"] - nrm["t_min"]) - 1
print("t normalise a l'injection :", tn_check)

#%% checkpoints
ck = torch.load(MODEL_PATH, map_location='cpu')
print(ck.keys())
print(ck.get('hyperparams', ck.get('args', 'pas de hyperparams')))
nrm = ck['normalization']
print("t:", nrm["t_min"], nrm["t_max"], "| p:", nrm["p_min"], nrm["p_max"])
tn = 2 * (15.0 * 3600.0 - nrm["t_min"]) / (nrm["t_max"] - nrm["t_min"]) - 1
print("t normalise a l'injection :", tn)