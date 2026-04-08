import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# --- Setup (given) ---
rng   = np.random.default_rng(42)
lats_f = np.arange(25, 50, 0.25);  lons_f = np.arange(-120, -70, 0.25)
LON, LAT = np.meshgrid(lons_f, lats_f)
storm  = 30 * np.exp(-((LAT - 37)**2 / 3 + (LON + 100)**2 / 3))
precip = storm + 0.5 + rng.exponential(0.3, storm.shape)
ds_fine = xr.Dataset({'precip': (['lat','lon'], precip, {'units':'mm/day'})},
                      coords={'lat': lats_f, 'lon': lons_f})

# Bilinear regrid to 1°
lats_c = np.arange(25.5, 50, 1.0);  lons_c = np.arange(-119.5, -70, 1.0)
ds_bil = ds_fine.interp(lat=lats_c, lon=lons_c)

# --- Your turn ---
# Area weights:  cos(lat) * dlat_deg * dlon_deg
w_f = np.cos(np.radians(ds_fine.lat)) * 0.25 * 0.25   # shape (lat,)
w_c = np.cos(np.radians(ds_bil.lat))  * 1.0  * 1.0

mass_fine = float((ds_fine['precip'] * w_f).sum())
mass_bil  = float((ds_bil['precip'] * w_c).sum())

print(f"Mass error: {(mass_bil - mass_fine) / mass_fine * 100:.3f}%")
# Also print: mean, min/max, NaN count for both grids
print("-----GRID FINE-----")
print(f"Mean: {float(ds_fine['precip'].mean())} mm / day ")
print(f"Min: {float(ds_fine['precip'].min())} mm / day ")
print(f"Max: {float(ds_fine['precip'].max())} mm / day  \n")


print("-----GRID bil-----")
print(f"Mean: {float(ds_bil['precip'].mean()):.3f} mm / day")
print(f"Min: {float(ds_bil['precip'].min()):.3f} mm / day")
print(f"Max: {float(ds_bil['precip'].max()):.3f} mm / day")

print(f"# of Nans: {int(ds_bil['precip'].isnull().sum())}")










def conservative_regrid_1d(f_src, src_edges, tgt_edges):
    """
    f_src:     source values  (n_src,)
    src_edges: source cell boundaries  (n_src + 1,)
    tgt_edges: target cell boundaries  (n_tgt + 1,)
    Returns:   target values  (n_tgt,)
    """
    n_tgt = len(tgt_edges) - 1
    out   = np.zeros(n_tgt)
    for i in range(n_tgt):
        t0, t1 = tgt_edges[i], tgt_edges[i + 1]
        # loop over source cells, accumulate weighted contributions
        for k in range(len(f_src)):
            s0, s1 = src_edges[k], src_edges[k + 1]
            overlap = max(0.0, min(s1, t1) - max(s0,t0))          # length of [s0,s1] ∩ [t0,t1]
            out[i] += f_src[k] * overlap
        out[i] /= (t1 - t0)       # normalize by target cell width
    return out




# --- Test scaffold (given) ---
lon_idx   = np.argmin(np.abs(lons_f - (-100)))
f_src     = ds_fine['precip'].values[:, lon_idx]

src_dlat  = float(lats_f[1] - lats_f[0])
src_edges = np.append(lats_f - src_dlat / 2, lats_f[-1] + src_dlat / 2)
tgt_edges = np.arange(src_edges[0], src_edges[-1] + 0.001, 1.0)

f_tgt = conservative_regrid_1d(f_src, src_edges, tgt_edges)

# Mass check — should be 0.000000%
mass_before = np.sum(f_src * np.diff(src_edges))
mass_after  = np.sum(f_tgt * np.diff(tgt_edges))
print(f"1D mass error: {(mass_after - mass_before) / mass_before * 100:.6f}%")

# Plot: overlay fine (0.25°) and conservative (1°) profiles
tgt_ctrs = 0.5 * (tgt_edges[:-1] + tgt_edges[1:])
fig, ax  = plt.subplots(figsize=(8, 3.5))
ax.plot(lats_f, f_src, label='Fine (0.25°)')      # fine grid line
ax.step(np.append(tgt_edges[:-1], tgt_edges[-1]),
        np.append(f_tgt, f_tgt[-1]), label='Conservative (1°)') # coarse step plot
ax.set_xlabel('Latitude (°N)');  ax.set_ylabel('Precip (mm/day)')
ax.legend();  plt.tight_layout();  plt.show()