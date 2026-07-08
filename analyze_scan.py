import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
 
# ============================================================
# Thresholding inside the circular aperture
# ============================================================
USE_THRESHOLD_MASK = True
PEAK_FRACTION_THRESHOLD = 0.0
NOISE_SIGMA_THRESHOLD = 3.0
 
# ============================================================
# Settings
# ============================================================
PIXEL_SIZE_MM = 0.017
BORDER_WIDTH = 20
NOISE_SIGMA_MULTIPLIER = 3.0
 
# Analyze only frames with Z_RANGE[0] <= z <= Z_RANGE[1]
Z_RANGE = (-8,7)   # set to None to use all frames
 
APERTURE_RADIUS_SCALE = 3
MIN_APERTURE_RADIUS_PX = 6.0
 
LAMBDA_UM = 3.0
LAMBDA_MM = LAMBDA_UM * 1e-3
 
MANUAL_SCAN_DIR = None
 
SAVE_FRAME_PLOTS = False
 
GALLERY_NCOLS = 4
GALLERY_PERCENTILE_VMAX = 99.5
 
# ============================================================
# Choose scan folder
# ============================================================
SCAN_ROOT = os.path.join(os.getcwd(), "all_scans")
 
def get_latest_scan_folder(scan_root=SCAN_ROOT):
    folders = [
        os.path.join(scan_root, d)
        for d in os.listdir(scan_root)
        if os.path.isdir(os.path.join(scan_root, d)) and d.startswith("scan_")
    ]
    if not folders:
        raise FileNotFoundError(f"No scan_* folders found in: {scan_root}")
    return max(folders, key=os.path.getmtime)
 
scan_dir = MANUAL_SCAN_DIR if MANUAL_SCAN_DIR is not None else get_latest_scan_folder()
print("Using scan folder:", scan_dir)
 
# ============================================================
# Helpers
# ============================================================
def extract_position_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"z_([+-]?\d+\.\d+)", name)
    return float(m.group(1)) if m else np.nan
 
def connected_component_from_seed(mask, seed_y, seed_x):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
 
    sy = int(round(seed_y))
    sx = int(round(seed_x))
 
    if sy < 0 or sy >= h or sx < 0 or sx >= w:
        return out
 
    if not mask[sy, sx]:
        return out
 
    stack = [(sy, sx)]
    visited[sy, sx] = True
 
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
 
    while stack:
        y, x = stack.pop()
        out[y, x] = True
        for dy, dx in neighbors:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                if mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
    return out
 
def beam_radius_model(z, w0, z0, M2):
    return np.sqrt(w0**2 + (M2 * LAMBDA_MM / (np.pi * w0))**2 * (z - z0)**2)
 
def fit_m2(z_mm, d4_mm):
    z = np.asarray(z_mm, dtype=float)
    d = np.asarray(d4_mm, dtype=float)
    good = np.isfinite(z) & np.isfinite(d) & (d > 0)
 
    z = z[good]
    d = d[good]
 
    if len(z) < 5:
        raise RuntimeError("Need at least 5 valid points for M² fit.")
 
    w = d / 2.0
    i0 = np.argmin(w)
    w0_guess = max(1e-6, w[i0])
    z0_guess = z[i0]
    M2_guess = 1.5
    z_span = np.ptp(z)
 
    popt, _ = curve_fit(
        beam_radius_model,
        z, w,
        p0=[w0_guess, z0_guess, M2_guess],
        bounds=([1e-6, z.min() - abs(z_span) - 10, 0.2],
                [10 * max(w), z.max() + abs(z_span) + 10, 100.0]),
        maxfev=20000
    )
 
    w0, z0, M2 = popt
    w_fit = beam_radius_model(z, *popt)
    d_fit = 2.0 * w_fit
 
    ss_res = np.sum((d - d_fit)**2)
    ss_tot = np.sum((d - np.mean(d))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
 
    return {
        "w0_mm": w0,
        "d0_mm": 2.0 * w0,
        "z0_mm": z0,
        "M2": M2,
        "r2": r2,
        "z_data_mm": z,
        "d_data_mm": d,
        "z_fit_mm": np.linspace(z.min(), z.max(), 400),
    }
 
# Cache indices per image shape to avoid recomputing
_INDICES_CACHE = {}
 
def get_indices(shape):
    if shape not in _INDICES_CACHE:
        _INDICES_CACHE[shape] = np.indices(shape)
    return _INDICES_CACHE[shape]
 
def analyze_frame_fast(path):
    frame = np.load(path).astype(np.float64)
    h, w = frame.shape
 
    border = np.concatenate([
        frame[:BORDER_WIDTH, :].ravel(),
        frame[-BORDER_WIDTH:, :].ravel(),
        frame[:, :BORDER_WIDTH].ravel(),
        frame[:, -BORDER_WIDTH:].ravel()
    ])
 
    bg = np.median(border)
    noise_sigma = np.std(border - bg)
 
    img = frame - bg
    img[img < 0] = 0.0
 
    if img.sum() <= 0:
        return {
            "file": os.path.basename(path),
            "position_mm": extract_position_from_name(path),
            "status": "failed",
            "reason": "No signal after background subtraction",
        }
 
    y_idx, x_idx = get_indices(img.shape)
 
    # Use peak only to identify the main beam support
    peak_y, peak_x = np.unravel_index(np.argmax(img), img.shape)
 
    threshold = NOISE_SIGMA_MULTIPLIER * noise_sigma
    mask0 = img > threshold
    beam_mask = connected_component_from_seed(mask0, peak_y, peak_x)
 
    if beam_mask.sum() == 0 or img[beam_mask].sum() <= 0:
        return {
            "file": os.path.basename(path),
            "position_mm": extract_position_from_name(path),
            "status": "failed",
            "reason": "No beam survived thresholding",
        }
 
    img_thr = np.where(beam_mask, img, 0.0)
    total0 = img_thr.sum()
 
    # First-pass centroid from the connected beam component
    xc0_px = (img_thr * x_idx).sum() / total0
    yc0_px = (img_thr * y_idx).sum() / total0
 
    sigma_x0_px = np.sqrt((img_thr * (x_idx - xc0_px)**2).sum() / total0)
    sigma_y0_px = np.sqrt((img_thr * (y_idx - yc0_px)**2).sum() / total0)
 
    r_ap_px = max(
        MIN_APERTURE_RADIUS_PX,
        APERTURE_RADIUS_SCALE * max(sigma_x0_px, sigma_y0_px)
    )
 
    # Aperture centered on centroid, not peak pixel
    rr2 = (x_idx - xc0_px)**2 + (y_idx - yc0_px)**2
    ap_mask = rr2 <= r_ap_px**2
 
    if USE_THRESHOLD_MASK:
        peak_val = img[peak_y, peak_x]
        thresh_peak = PEAK_FRACTION_THRESHOLD * peak_val
        thresh_noise = NOISE_SIGMA_THRESHOLD * noise_sigma
        intensity_threshold = max(thresh_peak, thresh_noise)
 
        thresh_mask = img >= intensity_threshold
 
        # Keep only thresholded pixels connected to the centroid region
        seed_y = int(round(yc0_px))
        seed_x = int(round(xc0_px))
 
        if (
            0 <= seed_y < h and 0 <= seed_x < w and
            thresh_mask[seed_y, seed_x]
        ):
            thresh_component = connected_component_from_seed(thresh_mask, seed_y, seed_x)
        else:
            # fall back to peak-connected component if centroid lands on a false pixel
            thresh_component = connected_component_from_seed(thresh_mask, peak_y, peak_x)
 
        final_mask = ap_mask & thresh_component
    else:
        intensity_threshold = 0.0
        final_mask = ap_mask
 
    img_ap = np.where(final_mask, img, 0.0)
    total = img_ap.sum()
 
    if total <= 0:
        return {
            "file": os.path.basename(path),
            "position_mm": extract_position_from_name(path),
            "status": "failed",
            "reason": "No signal left after aperture + threshold mask",
        }
 
    # Final centroid inside the final mask: use this as D4σ center
    xc_px = (img_ap * x_idx).sum() / total
    yc_px = (img_ap * y_idx).sum() / total
 
    sigma_x_px = np.sqrt((img_ap * (x_idx - xc_px)**2).sum() / total)
    sigma_y_px = np.sqrt((img_ap * (y_idx - yc_px)**2).sum() / total)
 
    d4x_px = 4.0 * sigma_x_px
    d4y_px = 4.0 * sigma_y_px
 
    peak_centroid_distance_px = np.sqrt((peak_x - xc_px)**2 + (peak_y - yc_px)**2)
 
    return {
        "file": os.path.basename(path),
        "position_mm": extract_position_from_name(path),
        "status": "ok",
        "peak_x_px": peak_x,
        "peak_y_px": peak_y,
        "peak_x_mm": peak_x * PIXEL_SIZE_MM,
        "peak_y_mm": peak_y * PIXEL_SIZE_MM,
        "xc_px": xc_px,
        "yc_px": yc_px,
        "xc_mm": xc_px * PIXEL_SIZE_MM,
        "yc_mm": yc_px * PIXEL_SIZE_MM,
        "d4x_px": d4x_px,
        "d4y_px": d4y_px,
        "d4x_mm": d4x_px * PIXEL_SIZE_MM,
        "d4y_mm": d4y_px * PIXEL_SIZE_MM,
        "aperture_radius_px": r_ap_px,
        "aperture_radius_mm": r_ap_px * PIXEL_SIZE_MM,
        "peak_centroid_distance_px": peak_centroid_distance_px,
        "peak_centroid_distance_mm": peak_centroid_distance_px * PIXEL_SIZE_MM,
        "_img": img,
        "_img_ap": img_ap,
        "_threshold": intensity_threshold,
    }
 
def make_beam_gallery_all_fast(df_ok, out_dir, ncols=4):
    if df_ok.empty:
        print("No successful frames to plot in gallery.")
        return None
 
    df_ok = df_ok.sort_values("position_mm").reset_index(drop=True)
    n = len(df_ok)
    nrows = int(math.ceil(n / ncols))
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
 
    for ax in axes:
        ax.set_visible(False)
 
    for ax, (_, row) in zip(axes, df_ok.iterrows()):
        ax.set_visible(True)
 
        img = row["_img"]
        h, w = img.shape
        extent_full = [0, w * PIXEL_SIZE_MM, h * PIXEL_SIZE_MM, 0]
 
        xc_mm = row["xc_mm"]
        yc_mm = row["yc_mm"]
        peak_x_mm = row["peak_x_mm"]
        peak_y_mm = row["peak_y_mm"]
        r_ap_mm = row["aperture_radius_mm"]
 
        xc_px = row["xc_px"]
        yc_px = row["yc_px"]
        r_ap_px = row["aperture_radius_px"]
 
        x_min = max(0, int(np.floor(xc_px - 1.4 * r_ap_px)))
        x_max = min(w, int(np.ceil(xc_px + 1.4 * r_ap_px)))
        y_min = max(0, int(np.floor(yc_px - 1.4 * r_ap_px)))
        y_max = min(h, int(np.ceil(yc_px + 1.4 * r_ap_px)))
 
        ax.imshow(
            img,
            cmap="inferno",
            origin="upper",
            extent=extent_full,
            aspect="equal",
            interpolation="nearest",
        )
 
        # Centroid and peak
        ax.scatter([xc_mm], [yc_mm], c="cyan", s=22, marker="+", linewidths=1.2)
        ax.scatter([peak_x_mm], [peak_y_mm], c="lime", s=18, marker="x", linewidths=1.0)
 
        circ = patches.Circle(
            (xc_mm, yc_mm),
            radius=r_ap_mm,
            fill=False,
            edgecolor="white",
            linewidth=1.0
        )
        ax.add_patch(circ)
 
        ax.set_xlim(x_min * PIXEL_SIZE_MM, x_max * PIXEL_SIZE_MM)
        ax.set_ylim(y_max * PIXEL_SIZE_MM, y_min * PIXEL_SIZE_MM)
        ax.set_xticks([])
        ax.set_yticks([])
 
        ax.set_title(
            f"z = {row['position_mm']:.2f} mm\n"
            f"D4σx = {row['d4x_mm']:.3f} mm\n"
            f"D4σy = {row['d4y_mm']:.3f} mm",
            fontsize=8
        )
 
    for ax in axes[n:]:
        ax.set_visible(False)
 
    plt.tight_layout()
    out_path = os.path.join(out_dir, "beam_gallery_all_fast.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()
    return out_path
 
# ============================================================
# Gather files
# ============================================================
frame_files = sorted(glob.glob(os.path.join(scan_dir, "*_frame_raw.npy")))
if not frame_files:
    raise FileNotFoundError(f"No *_frame_raw.npy files found in: {scan_dir}")
 
if Z_RANGE is not None:
    z_min, z_max = Z_RANGE
    filtered_files = []
    for f in frame_files:
        z = extract_position_from_name(f)
        if np.isfinite(z) and z_min <= z <= z_max:
            filtered_files.append(f)
    frame_files = filtered_files
    print(f"Using z range: {z_min:.3f} mm to {z_max:.3f} mm")
 
if not frame_files:
    raise FileNotFoundError(
        f"No *_frame_raw.npy files found in the requested z range: {Z_RANGE}"
    )
 
analysis_dir = os.path.join(scan_dir, "analysis_fast_centroid_centered")
os.makedirs(analysis_dir, exist_ok=True)
 
print(f"Found {len(frame_files)} raw frames")
 
# ============================================================
# Analyze all frames
# ============================================================
results = []
for i, path in enumerate(frame_files, 1):
    print(f"[{i}/{len(frame_files)}] {os.path.basename(path)}")
    results.append(analyze_frame_fast(path))
 
df = pd.DataFrame(results)
csv_path = os.path.join(analysis_dir, "d4sigma_summary.csv")
 
df_to_save = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
df_to_save.to_csv(csv_path, index=False)
print("\nSaved summary:", csv_path)
 
# ============================================================
# Keep successful frames
# ============================================================
df_ok = df[df["status"] == "ok"].copy().sort_values("position_mm")
if df_ok.empty:
    raise RuntimeError("No valid frames analyzed successfully.")
 
# ============================================================
# D4sigma plots
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_ok["position_mm"], df_ok["d4x_mm"], "o-", label="D4σx (mm)")
ax.plot(df_ok["position_mm"], df_ok["d4y_mm"], "s-", label="D4σy (mm)")
ax.set_xlabel("Stage position (mm)")
ax.set_ylabel("D4σ (mm)")
ax.set_title("Beam size vs stage position")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
 
summary_plot = os.path.join(analysis_dir, "d4sigma_vs_position.png")
fig.savefig(summary_plot, dpi=180, bbox_inches="tight")
plt.show()
 
# ============================================================
# Peak-centroid offset diagnostic
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_ok["position_mm"], df_ok["peak_centroid_distance_mm"], "o-")
ax.set_xlabel("Stage position (mm)")
ax.set_ylabel("Peak-centroid offset (mm)")
ax.set_title("Peak-centroid separation vs stage position")
ax.grid(alpha=0.3)
plt.tight_layout()
 
offset_plot = os.path.join(analysis_dir, "peak_centroid_offset_vs_position.png")
fig.savefig(offset_plot, dpi=180, bbox_inches="tight")
plt.show()
 
# ============================================================
# Gallery
# ============================================================
gallery_path = make_beam_gallery_all_fast(df_ok, analysis_dir, ncols=GALLERY_NCOLS)
if gallery_path is not None:
    print("Saved gallery:", gallery_path)
 
# ============================================================
# M² fits
# ============================================================
fit_x = fit_m2(df_ok["position_mm"].values, df_ok["d4x_mm"].values)
fit_y = fit_m2(df_ok["position_mm"].values, df_ok["d4y_mm"].values)
 
d_fit_x = 2 * beam_radius_model(fit_x["z_fit_mm"], fit_x["w0_mm"], fit_x["z0_mm"], fit_x["M2"])
d_fit_y = 2 * beam_radius_model(fit_y["z_fit_mm"], fit_y["w0_mm"], fit_y["z0_mm"], fit_y["M2"])
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_ok["position_mm"], df_ok["d4x_mm"], "o", label="D4σx data")
ax.plot(fit_x["z_fit_mm"], d_fit_x, "-", label=f"M²x fit = {fit_x['M2']:.3f}")
ax.plot(df_ok["position_mm"], df_ok["d4y_mm"], "s", label="D4σy data")
ax.plot(fit_y["z_fit_mm"], d_fit_y, "-", label=f"M²y fit = {fit_y['M2']:.3f}")
ax.set_xlabel("Stage position (mm)")
ax.set_ylabel("D4σ diameter (mm)")
ax.set_title(f"M² fit (λ = {LAMBDA_UM:.3f} µm)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
 
m2_plot = os.path.join(analysis_dir, "M2_fit.png")
fig.savefig(m2_plot, dpi=180, bbox_inches="tight")
plt.show()
 
m2_summary = pd.DataFrame([
    {
        "axis": "x",
        "lambda_um": LAMBDA_UM,
        "w0_mm": fit_x["w0_mm"],
        "d0_mm": fit_x["d0_mm"],
        "z0_mm": fit_x["z0_mm"],
        "M2": fit_x["M2"],
        "R2": fit_x["r2"],
    },
    {
        "axis": "y",
        "lambda_um": LAMBDA_UM,
        "w0_mm": fit_y["w0_mm"],
        "d0_mm": fit_y["d0_mm"],
        "z0_mm": fit_y["z0_mm"],
        "M2": fit_y["M2"],
        "R2": fit_y["r2"],
    }
])
 
m2_csv = os.path.join(analysis_dir, "M2_summary.csv")
m2_summary.to_csv(m2_csv, index=False)
 
print("\nAnalysis complete")
print("Analysis folder:", analysis_dir)