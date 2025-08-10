"""
Doppler-based sky-map toy model from rotating Earth transmitters.

This script:
  1) Loads city positions and populations.
  2) Rotates Earth about an arbitrary axis and computes line-of-sight Doppler shifts
     for each city as it traverses the visible limb (narrow beam selection).
  3) Builds a time–frequency "power" spectrogram P(t, f) proportional to population.
  4) Adds Gaussian noise, smooths in frequency, and FFTs over time to obtain harmonics.
  5) Integrates against associated Legendre functions to get spherical-harmonic-like
     coefficients a_sh (and, with noise, a_sh_obs).
  6) Reconstructs a simple sky map from those coefficients and exports all plots
     to a single multi-page PDF file (no EPS output).

Notes:
  * Input file "city_10.csv" must contain: "Latitude", "Longitude", "Population".
  * This script uses Cartopy for geographic plots. On some systems you may need to
    configure SSL certificates (handled below via certifi).
  * SciPy API: this code uses `scipy.special.sph_harm_y` as in the original source.
    If your SciPy provides only `sph_harm(m, l, theta, phi)`, adapt `compute_map_chunk`
    accordingly without changing the math.
"""

from __future__ import annotations

import os
from math import factorial, sqrt, pi

import certifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.special import lpmv
from scipy.special import sph_harm_y  # Keep original API usage; see note above.

import cartopy.crs as ccrs

# Ensure Cartopy can fetch resources over HTTPS on environments with strict SSL.
os.environ["SSL_CERT_FILE"] = certifi.where()


# ----------------------------
# Parameters (unchanged logic)
# ----------------------------
theta_deg, phi_deg = 0, 0            # Rotation axis in degrees: colatitude theta, longitude phi.
width, smooth = 0.05, 1              # Beam half-width (radians) and Gaussian smoothing (freq-axis).
noise = 5                            # Std. dev. of Gaussian noise added to P(t,f).
l_max = 20                           # Maximum spherical-harmonic degree used in reconstruction.
days = 1                             # Simulation duration in sidereal days.
observer_dir = np.array([1, 0, 0])   # Observer LOS unit vector (x-axis).
chunk_size = 300                     # Number of cities per processing chunk (no multiprocessing here).

earth_radius = 6371e3                # [m]
omega = 2 * np.pi / 86164            # Earth's sidereal rotation rate [rad/s]
c = 3e8                              # Speed of light [m/s]
B = earth_radius * omega / c         # Max fractional Doppler magnitude.
f_max = 1.5 * B                      # Frequency window half-span for the spectrogram axis.

num_freq = 1200
num_times = 1200 * days
times = np.linspace(0, 86164 * days, num_times)            # [s]
frequencies = np.linspace(-f_max, f_max, num_freq)         # fractional frequency offset (dimensionless)


# ----------------------------
# Utilities
# ----------------------------
def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 rotation matrix that rotates unit vector a to unit vector b.
    Stable Rodrigues formula; identity or negative identity on near-parallel/antiparallel.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c_ab = float(np.dot(a, b))
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return np.eye(3) if c_ab > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c_ab) / (v_norm ** 2))


def process_city(row: pd.Series,
                 city_vec: np.ndarray,
                 axis: np.ndarray,
                 omega: float,
                 times: np.ndarray,
                 observer_dir: np.ndarray,
                 earth_radius: float,
                 c: float,
                 f_max: float,
                 num_freq: int):
    """
    For a single city:
      - Rotate its position over time about 'axis' with rate omega.
      - Select 'visible' times when the city's projected x-component is within 'width' of the limb.
      - Compute LOS velocity and fractional frequency offset df = v_los / c.
      - Accumulate power at (t, f_index) proportional to population (arbitrary units).
    Returns:
      P_tf_local: (num_times, num_freq) contribution,
      visible_latitudes: list of apparent latitudes along axis (diagnostics),
      visible_dfs: list of df at visible times (diagnostics).
    """
    pop = row["Population"]
    angles = omega * times

    # Skew-symmetric matrix for cross-product with rotation axis k.
    k = axis
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    I = np.eye(3)

    # Time-dependent rotation matrices R(t) via Rodrigues' formula.
    Rts = I + np.sin(angles)[:, None, None] * K + (1 - np.cos(angles))[:, None, None] * (K @ K)

    # Rotate the fixed city position over time.
    city_vec_t = np.einsum('ijk,k->ij', Rts, city_vec)

    # Visibility: near-limb selection via projected x-component (observer looks along +x).
    x_comp = city_vec_t[:, 0]
    visible = np.abs(x_comp / earth_radius) < width

    # Rigid rotation velocity and LOS projection.
    v_t = np.cross(omega * axis, city_vec_t)   # [m/s]
    v_los = np.einsum('ij,j->i', v_t, observer_dir)
    df = v_los / c                              # fractional offset (dimensionless)

    # Bin fractional frequency into spectrogram indices.
    f_indices = np.clip(
        np.floor((df + f_max) / (2 * f_max) * num_freq).astype(int),
        0, num_freq - 1
    )

    # Diagnostics (unused downstream but kept as in original).
    lat_along_axis = np.degrees(np.arcsin(np.einsum('ij,j->i', city_vec_t, axis) / earth_radius))
    visible_latitudes = lat_along_axis[visible]
    visible_dfs = df[visible]

    # Accumulate power, scaled by population.
    P_tf_local = np.zeros((len(times), num_freq))
    np.add.at(P_tf_local, (np.where(visible)[0], f_indices[visible]), pop / 1e6)

    return P_tf_local, visible_latitudes.tolist(), visible_dfs.tolist()


def process_chunk(start_idx: int,
                  end_idx: int,
                  cities: pd.DataFrame,
                  city_vecs: np.ndarray,
                  axis: np.ndarray,
                  omega: float,
                  times: np.ndarray,
                  observer_dir: np.ndarray,
                  earth_radius: float,
                  c: float,
                  f_max: float,
                  num_freq: int):
    """
    Process a contiguous chunk of cities to keep memory use bounded.
    """
    try:
        cities_chunk = cities.iloc[start_idx:end_idx].copy()
        city_vecs_chunk = city_vecs[start_idx:end_idx]

        P_tf_chunk = np.zeros((len(times), num_freq))
        chunk_visible_latitudes, chunk_visible_dfs = [], []

        for i, row in cities_chunk.iterrows():
            city_vec = city_vecs_chunk[i - start_idx]
            P_loc, vis_lat, vis_df = process_city(
                row, city_vec, axis, omega, times, observer_dir,
                earth_radius, c, f_max, num_freq
            )
            P_tf_chunk += P_loc
            chunk_visible_latitudes.extend(vis_lat)
            chunk_visible_dfs.extend(vis_df)

        return P_tf_chunk, chunk_visible_latitudes, chunk_visible_dfs
    except Exception as e:
        print(f"Error in process_chunk: {e}")
        raise


def integrate_over_frequency(P_of: np.ndarray,
                             frequencies: np.ndarray,
                             B: float,
                             l_max: int):
    """
    Perform frequency-domain integration for each (l, m) using the time-FFT rows indexed by m.
    Returns a list-of-lists P_of_int[l][m] matching the original structure.
    """
    P_of_int = [[0 for _ in range(l + 1)] for l in range(l_max + 1)]
    f_mask = (frequencies >= 0) & (frequencies <= B)
    f_values = frequencies[f_mask]
    f_indices = np.where(f_mask)[0]

    for l in range(l_max + 1):
        for m in range(l + 1):
            row_segment = P_of[m, f_indices]
            leg = lpmv(m, l, np.sqrt(1 - (f_values / B) ** 2))
            integrand = f_values * leg * row_segment
            P_of_int[l][m] += np.trapezoid(integrand, f_values)

    return P_of_int


def compute_a_sh(P_of_int, B: float, l_max: int):
    """
    Convert integrated quantities to a_sh coefficients with the same parity condition as original.
    Returns list-of-lists a_sh[l][m].
    """
    a_sh = [[0 for _ in range(l + 1)] for l in range(l_max + 1)]
    for l in range(l_max + 1):
        for m in range(l + 1):
            if (l + m) % 2 != 0:
                a_sh[l][m] = 0
            else:
                factor = (1j ** m) * B * sqrt(pi * (2 * l + 1)) * sqrt(factorial(l - m) / factorial(l + m))
                a_sh[l][m] = factor * P_of_int[l][m]
    return a_sh


def compute_map_chunk(l: int, a_sh, alpha_grid: np.ndarray, delta_grid: np.ndarray):
    """
    Sum Y_lm(alpha, delta) * a_lm for a fixed degree l over m=-l..l using the original sph_harm_y API.
    """
    Y_map_chunk = np.zeros_like(alpha_grid, dtype=complex)
    for m in range(-l, l + 1):
        a_lm = (-1) ** m * np.conj(a_sh[l][abs(m)]) if m < 0 else a_sh[l][m]
        Y_lm = sph_harm_y(l, m, delta_grid, alpha_grid)  # (theta=delta, phi=alpha) per original.
        Y_map_chunk += a_lm * Y_lm
    return Y_map_chunk


def save_page(pdf: PdfPages):
    """
    Append the current Matplotlib figure to the multi-page PDF and close it.
    """
    pdf.savefig()
    plt.close()


# ----------------------------
# Main script
# ----------------------------
if __name__ == '__main__':
    # Load city catalog (expects Latitude [deg], Longitude [deg], Population).
    cities = pd.read_csv("city_10.csv")

    # Rotation axis unit vector from (theta, phi) in degrees.
    theta, phi = np.radians([theta_deg, phi_deg])
    axis = np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])
    axis /= np.linalg.norm(axis)

    # City ECEF-like unit vectors scaled by Earth radius.
    lat_rad, lon_rad = np.radians(cities[["Latitude", "Longitude"]].values.T)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    city_vecs_orig = np.stack([x, y, z], axis=1) * earth_radius

    # Align Earth's z-axis with chosen rotation axis.
    R_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)
    city_vecs = city_vecs_orig @ R_align.T

    # Accumulate spectrogram over city chunks.
    P_tf = np.zeros((num_times, num_freq))
    all_visible_latitudes, all_visible_dfs = [], []

    for start_idx in range(0, len(cities), chunk_size):
        end_idx = min(start_idx + chunk_size, len(cities))
        P_chunk, vis_lat, vis_df = process_chunk(
            start_idx, end_idx, cities, city_vecs,
            axis, omega, times, observer_dir,
            earth_radius, c, f_max, num_freq
        )
        P_tf += P_chunk
        all_visible_latitudes.extend(vis_lat)
        all_visible_dfs.extend(vis_df)

    # Preserve diagnostics exactly as before.
    visible_latitudes, visible_dfs = all_visible_latitudes, all_visible_dfs

    # Additive Gaussian noise and frequency smoothing (identical sigma).
    P_tf_obs = P_tf + np.random.normal(0, noise, P_tf.shape)
    P_tf_smooth = gaussian_filter1d(P_tf, sigma=smooth, axis=1)
    P_tf_obs_smooth = gaussian_filter1d(P_tf_obs, sigma=smooth, axis=1)

    # FFT along time to obtain harmonics (rows index m).
    P_of = np.fft.fft(P_tf_smooth, axis=0)
    P_of_obs = np.fft.fft(P_tf_obs_smooth, axis=0)

    # Frequency-domain integrations for clean and noisy cases (structure preserved).
    P_of_int = integrate_over_frequency(P_of, frequencies, B, l_max)
    P_of_obs_int = integrate_over_frequency(P_of_obs, frequencies, B, l_max)

    # a_sh coefficients (and with noise) with same parity selection and prefactor.
    a_sh = compute_a_sh(P_of_int, B, l_max)
    a_sh_obs = compute_a_sh(P_of_obs_int, B, l_max)

    # ----------------------------
    # Plotting & single-PDF export
    # ----------------------------
    pdf_filename = "reconstruct.pdf"
    with PdfPages(pdf_filename) as pdf:

        # City distribution on a Plate Carree map.
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.scatter(
            cities["Longitude"],
            cities["Latitude"],
            s=cities["Population"] / 1e5,
            c='blue',
            transform=ccrs.PlateCarree()
        )
        ax.coastlines(color='gray', linewidth=1)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Spectrogram (linear scale).
        plt.figure(figsize=(10, 6), dpi=600)
        plt.imshow(
            P_tf_smooth,
            extent=[-f_max, f_max, 24 * days, 0],
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=np.max(P_tf_smooth)
        )
        plt.xlabel("Fractional frequency offset")
        plt.ylabel("Time [hours]")
        plt.colorbar(label="Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Spectrogram (log scale).
        plt.figure(figsize=(10, 6), dpi=600)
        plt.imshow(
            np.log10(P_tf_smooth + 1e-10),
            extent=[-f_max, f_max, 24 * days, 0],
            aspect='auto',
            cmap='viridis',
            vmin=-2,
            vmax=np.log10(np.max(P_tf_smooth) + 1e-10)
        )
        plt.xlabel("Fractional frequency offset")
        plt.ylabel("Time [hours]")
        plt.colorbar(label="Log10 Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Spectrogram with noise (linear).
        plt.figure(figsize=(10, 6), dpi=600)
        plt.imshow(
            P_tf_obs_smooth,
            extent=[-f_max, f_max, 24 * days, 0],
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=np.max(P_tf_obs_smooth)
        )
        plt.xlabel("Fractional frequency offset")
        plt.ylabel("Time [hours]")
        plt.colorbar(label="Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Spectrogram with noise (log).
        plt.figure(figsize=(10, 6), dpi=600)
        P_tf_obs_smooth_clipped = np.clip(P_tf_obs_smooth, 1e-10, None)
        plt.imshow(
            np.log10(P_tf_obs_smooth_clipped),
            extent=[-f_max, f_max, 24 * days, 0],
            aspect='auto',
            cmap='viridis',
            vmin=-2,
            vmax=np.log10(np.max(P_tf_obs_smooth_clipped))
        )
        plt.xlabel("Fractional frequency offset")
        plt.ylabel("Time [hours]")
        plt.colorbar(label="Log10 Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Fourier transform (P_of, log scale).
        plt.figure(figsize=(10, 6), dpi=600)
        plt.imshow(
            np.log10(np.abs(P_of) ** 2 + 1e-10),
            extent=[frequencies[0], frequencies[-1], 0, num_times // 2],
            aspect='auto',
            cmap='viridis',
            vmin=-2,
            vmax=np.log10(np.max(np.abs(P_of) ** 2))
        )
        plt.ylim(0, 50)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("harmonics of Earth's rotation frequency m")
        plt.colorbar(label="Log10 Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Fourier transform (P_of_obs, log scale).
        plt.figure(figsize=(10, 6), dpi=600)
        plt.imshow(
            np.log10(np.abs(P_of_obs) ** 2 + 1e-10),
            extent=[frequencies[0], frequencies[-1], 0, num_times // 2],
            aspect='auto',
            cmap='viridis',
            vmin=2,  # Keep original vmin choice.
            vmax=np.log10(np.max(np.abs(P_of_obs) ** 2))
        )
        plt.ylim(0, 50)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("harmonics of Earth's rotation frequency m")
        plt.colorbar(label="Log10 Power (arb. units)")
        plt.grid(True)
        plt.tight_layout()
        save_page(pdf)

        # Plot column 900 (index 899) of P_of and P_of_obs.
        col_idx = 899
        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(range(len(P_of)), np.abs(P_of[:, col_idx]) ** 2, label="Column 900 (|P_of|^2)")
        plt.xlim(1, 100)
        plt.ylim(1e1, 1e6)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel("Power (|P_of|^2)")
        plt.grid(True)
        plt.legend()
        save_page(pdf)

        plt.figure(figsize=(10, 6), dpi=600)
        plt.plot(range(len(P_of_obs)), np.abs(P_of_obs[:, col_idx]) ** 2, label="Column 900 (|P_of|^2)")
        plt.xlim(1, 100)
        plt.ylim(1e1, 1e6)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel("Power (|P_of|^2)")
        plt.grid(True)
        plt.legend()
        save_page(pdf)

        # Average power per l for a_sh and a_sh_obs.
        plt.figure(figsize=(10, 6))
        a_sh_avg = [np.mean([np.abs(a_sh[l][m]) ** 2 for m in range(l + 1)]) for l in range(1, l_max + 1)]
        a_sh_obs_avg = [np.mean([np.abs(a_sh_obs[l][m]) ** 2 for m in range(l + 1)]) for l in range(1, l_max + 1)]
        plt.loglog(range(1, l_max + 1), a_sh_avg, marker='o', label="Average |a_sh[l][m]|^2")
        plt.loglog(range(1, l_max + 1), a_sh_obs_avg, marker='x', label="Average |a_sh_obs[l][m]|^2")
        plt.xlabel("l (log scale)")
        plt.ylabel("Average |a_sh[l][m]|^2 (log scale)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        save_page(pdf)

        # Map reconstruction from a_sh (clean).
        alpha = np.linspace(0, 2 * np.pi, 360)
        delta = np.linspace(0, np.pi, 180)
        alpha_grid, delta_grid = np.meshgrid(alpha, delta)

        Y_map = np.zeros_like(alpha_grid, dtype=complex)
        for l in range(l_max + 1):
            Y_map += compute_map_chunk(l, a_sh, alpha_grid, delta_grid)
        Y_map_real = np.real(Y_map)
        Y_map_real = np.clip(Y_map_real, 0, 0.8 * np.max(Y_map_real))

        # --- ここからCartopy地図・海岸線・サイズ調整 ---
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        cf = ax.contourf(
            np.degrees(np.pi - alpha_grid),
            np.degrees(np.pi / 2 - delta_grid),
            Y_map_real,
            levels=100,
            vmin=0,
            vmax=np.max(Y_map_real),
            cmap="viridis",
            transform=ccrs.PlateCarree()
        )
        cbar = plt.colorbar(
            cf,
            label="Radio intensity [arb. units]",
            ax=ax,
            orientation='vertical',
            fraction=0.0414,
            pad=0.04
        )
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.grid(True)
        ax.coastlines(color='gray', linewidth=1)
        ax.set_aspect('auto')
        plt.subplots_adjust(right=0.88, top=0.91, bottom=0.09)
        save_page(pdf)
        # --- ここまで変更 ---

        # Map reconstruction from a_sh_obs (with noise) over a geographic base.
        Y_map_obs = np.zeros_like(alpha_grid, dtype=complex)
        for l in range(l_max + 1):
            Y_map_obs += compute_map_chunk(l, a_sh_obs, alpha_grid, delta_grid)
        Y_map_obs_real = np.real(Y_map_obs)
        Y_map_obs_real = np.clip(Y_map_obs_real, 0, 0.8 * np.max(Y_map_obs_real))

        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        cf = ax.contourf(
            np.degrees(np.pi - alpha_grid),
            np.degrees(np.pi / 2 - delta_grid),
            Y_map_obs_real,
            levels=100,
            vmin=0,
            vmax=np.max(Y_map_obs_real),
            cmap="viridis",
            transform=ccrs.PlateCarree()
        )
        cbar = plt.colorbar(
            cf,
            label="Radio intensity [arb. units]",
            ax=ax,
            orientation='vertical',
            fraction=0.0414,
            pad=0.04
        )
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.grid(True)
        ax.coastlines(color='gray', linewidth=1)
        ax.set_aspect('auto')
        plt.subplots_adjust(right=0.88, top=0.91, bottom=0.09)
        save_page(pdf)

