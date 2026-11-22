import os
import shutil
import datetime
import glob
import json
import math
import sys
import re
import time
import tempfile
import threading
import contextlib
import functools
import concurrent.futures
import numpy as np
import xarray as xr
import eccodes
import cv2
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3
import contourpy
import gzip
import struct

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ.setdefault("TQDM_DISABLE", "1")   # silence download progress bars (best effort)
sys.stdout.reconfigure(line_buffering=True)  # so log timestamps match reality

# 🌟 NORTH AMERICA BOUNDING BOX (0.25° grid crop)
NA_LAT_MIN = 5.0
NA_LAT_MAX = 80.0
NA_LON_MIN = -175.0
NA_LON_MAX = -40.0
NA_GRID_H = int(round((NA_LAT_MAX - NA_LAT_MIN) / 0.25)) + 1
NA_GRID_W = int(round((NA_LON_MAX - NA_LON_MIN) / 0.25)) + 1

# 🌟 DYNAMIC MEMBER RANGE CONFIGURATION (Supports GHA Matrix)
MEMBER_RANGE = os.environ.get("MEMBER_RANGE", "1-10")

for arg in list(sys.argv):
    if arg.startswith("--members="):
        MEMBER_RANGE = arg.split("=", 1)[1]
        sys.argv.remove(arg)
    elif arg == "--members":
        idx = sys.argv.index(arg)
        if idx + 1 < len(sys.argv):
            MEMBER_RANGE = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
        sys.argv.pop(idx)

def parse_member_range(range_str):
    m = re.match(r"^(\d+)-(\d+)$", range_str.strip())
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    if "," in range_str:
        return [int(x.strip()) for x in range_str.split(",")]
    return list(range(1, 11))

MEMBER_LIST = parse_member_range(MEMBER_RANGE)
NUM_MEMBERS = len(MEMBER_LIST)
MEMBERS_STR = f"m{MEMBER_LIST[0]:02d}-m{MEMBER_LIST[-1]:02d}"
MEMBER_IDS = [f"m{i:02d}" for i in MEMBER_LIST]

# 🌟 MANUAL RUN OVERRIDE (Set to None to automatically fetch the latest run)
MANUAL_RUN_DATE = 20261122
MANUAL_RUN_CYCLE = 12

# 🌟 AIFS Ensemble: 6-hourly to F024
FORECAST_STEPS = list(range(0, 361, 6))

MAX_TEXTURE_SIZE = 4096
MAX_CONCURRENT_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))
GZIP_LEVEL = 6
GZIP_THREADS = 4
MAX_CONCURRENT_PARAMS = 2

# 🌟 Point to the AIFS ENS config file
CONFIG_FILE_PATH = os.path.join("config", "parametersAIFSENS.json")
HF_REPO_ID = "PhillyWeatherGuy/baroclinic-aifsens-data"

CONTOUR_COORD_SCALE = 1000
CONTOUR_FORMAT = "CTV2"
INT16_MAX = 32767

CLIMATOLOGY_CACHE = {}
_ECCODES_LOCK = threading.Lock()
_NA_CROP_CACHE = {}
_MEMBER_TO_IDX = {m: idx for idx, m in enumerate(sorted(MEMBER_LIST))}
SHARED_2T_CACHE_DIR = os.path.join(tempfile.gettempdir(), f"aifs_2t_cache_{MEMBERS_STR}")

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
TIMINGS = {}
_T_LOCK = threading.Lock()

@contextlib.contextmanager
def tm(key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _T_LOCK:
            t = TIMINGS.setdefault(key, [0.0, 0])
            t[0] += dt
            t[1] += 1

def timed(key):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with tm(key):
                return fn(*a, **k)
        return wrapper
    return deco

def _gzip_to_file(path, raw_bytes):
    with open(path, "wb") as f:
        f.write(gzip.compress(raw_bytes, compresslevel=GZIP_LEVEL))

def _print_timing(pid, phases, total):
    ph = " | ".join(f"{k} {v:.1f}s" for k, v in phases.items())
    work = " | ".join(f"{k} {v[0]:.1f}s/{v[1]}x" for k, v in sorted(TIMINGS.items()))
    print(f"⏱️ [{pid}] total {total:.1f}s || phases: {ph} || thread-summed: {work}", flush=True)

def load_parameter_config(param_key="2t"):
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            if param_key in params:
                return params[param_key]
    raise FileNotFoundError(f"Parameter '{param_key}' not found in {CONFIG_FILE_PATH}")

# ---------------------------------------------------------------------------
# Fast direct ecCodes GRIB2 decoder + North America crop
# ---------------------------------------------------------------------------

def _to_slice_or_idx(idx):
    if len(idx) >= 2:
        diffs = np.diff(idx)
        if np.all(diffs == 1):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        if np.all(diffs == -1):
            stop = None if idx[-1] == 0 else int(idx[-1]) - 1
            return slice(int(idx[0]), stop, -1)
    return idx

def _get_na_crop(gid, ni, nj, lat_first, lat_last, lon_first, lon_last):
    key = (ni, nj, round(lat_first, 6), round(lat_last, 6), round(lon_first, 6), round(lon_last, 6))
    cached = _NA_CROP_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        lats = np.asarray(eccodes.codes_get_array(gid, "distinctLatitudes"), dtype=np.float64)
        lons = np.asarray(eccodes.codes_get_array(gid, "distinctLongitudes"), dtype=np.float64)
        if len(lats) != nj or len(lons) != ni:
            raise ValueError("grid axis length mismatch")
    except Exception:
        lats = np.linspace(lat_first, lat_last, nj)
        dlon = (lon_last - lon_first) % 360.0
        step_lon = dlon / (ni - 1) if ni > 1 else 0.0
        lons = lon_first + np.arange(ni, dtype=np.float64) * step_lon

    lat_idx = np.flatnonzero((lats >= NA_LAT_MIN - 1e-6) & (lats <= NA_LAT_MAX + 1e-6))
    if len(lat_idx) > 1 and lats[lat_idx[0]] < lats[lat_idx[-1]]:
        lat_idx = lat_idx[::-1]

    lons_180 = ((lons + 180.0) % 360.0) - 180.0
    lon_idx = np.flatnonzero((lons_180 >= NA_LON_MIN - 1e-6) & (lons_180 <= NA_LON_MAX + 1e-6))
    lon_idx = lon_idx[np.argsort(lons_180[lon_idx], kind="stable")]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"Empty NA crop: lat_idx={len(lat_idx)}, lon_idx={len(lon_idx)}")

    lats_na = lats[lat_idx]
    lons_na = lons_180[lon_idx]
    lat_sel = _to_slice_or_idx(lat_idx)
    lon_sel = _to_slice_or_idx(lon_idx)
    use_slices = isinstance(lat_sel, slice) and isinstance(lon_sel, slice)
    ix = None if use_slices else np.ix_(lat_idx, lon_idx)

    entry = (lat_sel, lon_sel, ix, lats_na, lons_na, len(lat_idx), len(lon_idx))
    _NA_CROP_CACHE[key] = entry
    return entry

def read_grib_na(grib_path):
    by_param = {}
    lats_na = None
    lons_na = None

    with _ECCODES_LOCK:
        with open(grib_path, "rb") as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    short_name = eccodes.codes_get(gid, "shortName")
                    level = int(eccodes.codes_get(gid, "level"))
                    try:
                        m_num = int(eccodes.codes_get(gid, "perturbationNumber"))
                    except Exception:
                        m_num = 0

                    m_idx = _MEMBER_TO_IDX.get(m_num)
                    if m_idx is None:
                        if NUM_MEMBERS == 1:
                            m_idx = 0
                        else:
                            continue

                    ni = int(eccodes.codes_get(gid, "Ni"))
                    nj = int(eccodes.codes_get(gid, "Nj"))
                    lat_first = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
                    lat_last = float(eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees"))
                    lon_first = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
                    lon_last = float(eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees"))

                    lat_sel, lon_sel, ix, lats_na, lons_na, h_na, w_na = _get_na_crop(
                        gid, ni, nj, lat_first, lat_last, lon_first, lon_last
                    )

                    vals = eccodes.codes_get_values(gid).reshape(nj, ni)
                    cropped = vals[lat_sel, lon_sel] if ix is None else vals[ix]
                    cropped = cropped.astype(np.float32, copy=True)

                    if eccodes.codes_get(gid, "bitmapPresent"):
                        miss = float(eccodes.codes_get(gid, "missingValue"))
                        cropped[cropped == miss] = np.nan

                    lvl_dict = by_param.setdefault(short_name, {})
                    arr = lvl_dict.get(level)
                    if arr is None:
                        arr = np.full((NUM_MEMBERS, h_na, w_na), np.nan, dtype=np.float32)
                        lvl_dict[level] = arr
                    arr[m_idx] = cropped
                finally:
                    eccodes.codes_release(gid)

    return by_param, lats_na, lons_na

def _get_2t_cache_path(target_date, chosen_run, step):
    return os.path.join(SHARED_2T_CACHE_DIR, f"2t_{target_date}_{chosen_run}_{step:03d}.npy")

# ---------------------------------------------------------------------------
# Contours: vectorized extraction + CTV2 binary packing
# ---------------------------------------------------------------------------

def _pack_level(cont_gen, level):
    pts_list, off_list = cont_gen.lines(level)
    pts, off = pts_list[0], off_list[0]
    if pts is None or off is None or len(off) < 2:
        return None

    off = off.astype(np.intp)
    lens = np.diff(off)
    starts = off[:-1]

    x = np.minimum(pts[:, 0], 180.0)
    seam = (np.maximum.reduceat(x, starts) < -179.99) | (np.minimum.reduceat(x, starts) > 179.99)

    q = np.empty((len(pts), 2), dtype=np.int32)
    q[:, 0] = np.rint(x * CONTOUR_COORD_SCALE)
    q[:, 1] = np.rint(pts[:, 1] * CONTOUR_COORD_SCALE)

    keep_line = ~seam
    if not keep_line.all():
        q = q[np.repeat(keep_line, lens)]
        lens = lens[keep_line]
    if len(lens) == 0:
        return None

    n = len(q)
    is_start = np.zeros(n, dtype=bool)
    is_start[np.cumsum(lens) - lens] = True

    if n > 1:
        is_start[1:] |= (np.abs(np.diff(q, axis=0)) > INT16_MAX).any(axis=1)

    seg_starts = np.flatnonzero(is_start)
    seg_lens = np.diff(np.append(seg_starts, n))

    keep_seg = seg_lens >= 2
    if not keep_seg.all():
        q = q[np.repeat(keep_seg, seg_lens)]
        seg_lens = seg_lens[keep_seg]
        seg_starts = np.cumsum(seg_lens) - seg_lens
        if len(seg_lens) == 0:
            return None
        is_start = np.zeros(len(q), dtype=bool)
        is_start[seg_starts] = True

    deltas = np.diff(q, axis=0)
    inner = ~is_start[1:]
    dx = deltas[inner, 0].astype("<i2")
    dy = deltas[inner, 1].astype("<i2")

    return b"".join((
        struct.pack("<I", len(seg_lens)),
        seg_lens.astype("<u4").tobytes(),
        q[seg_starts, 0].astype("<i4").tobytes(),
        q[seg_starts, 1].astype("<i4").tobytes(),
        dx.tobytes(),
        dy.tobytes(),
    ))

def _level_style(level, contours_config):
    name = str(level)
    unit = None
    for c_def in contours_config:
        if c_def.get("dynamic") and c_def.get("unit") in {"dam", "mb", "hPa"}:
            unit = c_def.get("unit")
            name = str(int(level)) if float(level).is_integer() else str(level)
            break

    match = next(
        (c for c in contours_config if c.get("dynamic") or c.get("target") == level),
        {},
    )
    return {
        "name": name,
        "unit": unit or match.get("unit", ""),
        "color": match.get("color", "#000000"),
        "width": match.get("width", 1.6),
        "opacity": match.get("opacity", 0.9),
    }

@timed("contour_math")
def extract_contour_levels(raw_arr_k, contours_config=None, contour_settings=None):
    if not contours_config:
        return []

    try:
        contour_settings = contour_settings or {}
        coarsen_stride = max(1, int(contour_settings.get("coarsen_stride", 1)))
        blur_kernel = tuple(contour_settings.get("blur_kernel", (5, 5)))
        blur_sigma = float(contour_settings.get("blur_sigma", 1.2))

        working_arr = raw_arr_k.astype(np.float32)
        if coarsen_stride > 1:
            working_arr = cv2.resize(
                working_arr,
                (max(1, working_arr.shape[1] // coarsen_stride), max(1, working_arr.shape[0] // coarsen_stride)),
                interpolation=cv2.INTER_AREA,
            )

        frame_h, frame_w = working_arr.shape
        smoothed = cv2.GaussianBlur(working_arr, blur_kernel, blur_sigma)
        smoothed_flipped = np.flipud(smoothed)

        lons = np.linspace(NA_LON_MIN, NA_LON_MAX, frame_w)
        lats = np.linspace(NA_LAT_MIN, NA_LAT_MAX, frame_h)
        cont_gen = contourpy.contour_generator(
            x=lons, y=lats, z=smoothed_flipped,
            line_type=contourpy.LineType.ChunkCombinedOffset,
        )

        interval = 6.0
        explicit_levels = []

        for c_def in contours_config:
            if c_def.get("dynamic"):
                interval = float(c_def.get("interval", 6.0))
                valid_mask = np.isfinite(working_arr) & (working_arr > 0)
                if not np.any(valid_mask):
                    explicit_levels = []
                    break

                valid_values = working_arr[valid_mask]
                data_min = float(np.nanmin(valid_values))
                data_max = float(np.nanmax(valid_values))
                if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                    start = math.ceil(data_min / interval) * interval
                    end = math.floor(data_max / interval) * interval
                    explicit_levels = [float(v) for v in np.arange(start, end + (interval * 0.5), interval)]
                break
            if "target" in c_def:
                explicit_levels.append(float(c_def["target"]))

        if not explicit_levels:
            return []

        levels = []
        for target_val in sorted(set(explicit_levels)):
            payload = _pack_level(cont_gen, target_val)
            if payload is not None:
                levels.append({"props": _level_style(target_val, contours_config), "payload": payload})

        return levels
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return []

@timed("pack")
def pack_step(levels):
    return {
        "blob": b"".join(l["payload"] for l in levels),
        "props": [l["props"] for l in levels],
    }

def assemble_contour_file(steps):
    parts = [CONTOUR_FORMAT.encode("ascii"), struct.pack("<I", len(steps))]
    for step in sorted(steps):
        entry = steps[step]
        parts.append(struct.pack("<III", int(step), len(entry["props"]), len(entry["blob"])))
        parts.append(entry["blob"])
    return b"".join(parts)

def normalize_array(raw_arr, param_config):
    scaling = param_config.get("scaling", {})
    mode = scaling.get("mode", "linear")

    if mode == "piecewise" and "val_points" in scaling and "byte_points" in scaling:
        multiplier = scaling.get("unit_multiplier", 1.0)
        v = np.nan_to_num(raw_arr, nan=0.0) * multiplier
        val_pts = scaling["val_points"]
        byte_pts = scaling["byte_points"]
        return np.interp(v, val_pts, byte_pts).astype(np.uint8)
    else:
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))
        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)

def get_climatology_grid(param_id, target_date=None, target_shape=(NA_GRID_H, NA_GRID_W), unit="m"):
    month = int(target_date[4:6]) if target_date and len(target_date) >= 6 else 10
    cache_key = f"{param_id}_{month}_{unit}_{target_shape[0]}x{target_shape[1]}"
    if cache_key in CLIMATOLOGY_CACHE:
        return CLIMATOLOGY_CACHE[cache_key]

    clim_dir = os.path.join("config", "climatology")
    if not os.path.exists(clim_dir):
        clim_dir = "climatology"

    candidate_files = [
        os.path.join(clim_dir, "z500_clim.nc"),
        os.path.join(clim_dir, "era5_z500_climatology.nc"),
        os.path.join(clim_dir, "z500_climatology.nc"),
        os.path.join(clim_dir, f"z500_clim_{month:02d}.npy"),
        os.path.join(clim_dir, "z500_clim.npy"),
        os.path.join(clim_dir, "z500_clim.npz"),
    ]

    for c_path in candidate_files:
        if os.path.exists(c_path):
            try:
                if c_path.endswith(".nc"):
                    with xr.open_dataset(c_path) as ds:
                        var = next((v for v in ["z", "gh", "z500", "hgt"] if v in ds), list(ds.data_vars)[0])
                        da = ds[var]
                        if "month" in da.coords:
                            da = da.sel(month=month)
                        grid = np.squeeze(da.values).astype(np.float32)
                elif c_path.endswith(".npz"):
                    with np.load(c_path) as data:
                        grid = data[f"month_{month:02d}"] if f"month_{month:02d}" in data else data[list(data.keys())[0]]
                elif c_path.endswith(".npy"):
                    grid = np.load(c_path).astype(np.float32)

                if grid.shape == (721, 1440) and target_shape != (721, 1440):
                    y0 = int(round((90.0 - NA_LAT_MAX) / 0.25))
                    y1 = int(round((90.0 - NA_LAT_MIN) / 0.25)) + 1
                    x0 = int(round((NA_LON_MIN + 180.0) / 0.25))
                    x1 = int(round((NA_LON_MAX + 180.0) / 0.25)) + 1
                    grid = grid[y0:y1, x0:x1]

                if grid.shape != target_shape:
                    grid = cv2.resize(grid, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

                if np.nanmean(grid) > 20000.0:
                    scale = 98.0665 if unit == "dam" else 9.80665
                    grid /= scale
                elif unit == "dam" and np.nanmean(grid) > 1000.0:
                    grid /= 10.0

                CLIMATOLOGY_CACHE[cache_key] = grid
                return grid
            except Exception as e:
                pass

    return None

@timed("decode_all")
def process_thickness_grib(grib_path, contour_config):
    by_param, _, _ = read_grib_na(grib_path)
    z_levels = by_param.get("z") or by_param[next(iter(by_param))]
    if 500 in z_levels and 1000 in z_levels:
        z500 = z_levels[500]
        z1000 = z_levels[1000]
    else:
        sorted_lvls = sorted(z_levels)
        z500, z1000 = z_levels[sorted_lvls[0]], z_levels[sorted_lvls[1]]

    thickness_dam = (z500 - z1000) / 98.0665
    if thickness_dam.ndim == 3:
        return [
            extract_contour_levels(
                thickness_dam[m],
                contour_config.get("contours", []),
                contour_config.get("contours_settings")
            )
            for m in range(thickness_dam.shape[0])
        ]

    return [extract_contour_levels(
        thickness_dam,
        contour_config.get("contours", []),
        contour_config.get("contours_settings")
    )]

@timed("decode_all")
def process_grib_to_array(grib_path, param_config, want_raster=True, target_date=None, want_raw=False, preloaded=None):
    if preloaded is not None:
        by_param, lats_na, lons_na = preloaded
    else:
        by_param, lats_na, lons_na = read_grib_na(grib_path)

    grib_var = param_config["grib_param"]

    if "u" in by_param and "v" in by_param:
        u = next(iter(by_param["u"].values()))
        v = next(iter(by_param["v"].values()))

        if param_config.get("id") == "pva":
            lat_rad = np.deg2rad(lats_na)
            R = 6371000.0
            dlat = np.deg2rad(np.abs(lats_na[0] - lats_na[1]))
            dlon = np.deg2rad(np.abs(lons_na[1] - lons_na[0]))

            dy = R * dlat
            dx = R * np.cos(lat_rad[:, None]) * dlon

            dv_dx = np.gradient(v, axis=-1) / dx
            du_dy = -np.gradient(u, axis=-2) / dy
            raw_arr_k = dv_dx - du_dy
        else:
            raw_arr_k = np.hypot(u, v)
    else:
        target_var = grib_var if (isinstance(grib_var, str) and grib_var in by_param) else next(iter(by_param))
        raw_arr_k = next(iter(by_param[target_var].values()))
        if NUM_MEMBERS == 1 and raw_arr_k.shape[0] == 1:
            raw_arr_k = raw_arr_k[0]

    if param_config.get("category_mode"):
        return np.nan_to_num(raw_arr_k, nan=0.0).clip(0, 255).astype(np.uint8), []

    if "anom" in param_config.get("id", "").lower() or param_config.get("is_anomaly"):
        target_unit = param_config.get("unit", "m").lower()
        scale_div = 98.0665 if target_unit in {"dam"} else 9.80665
        hgt = raw_arr_k / scale_div

        clim = get_climatology_grid(param_config["id"], target_date, hgt.shape[-2:], unit=target_unit)
        if clim is None:
            clim = np.tile(np.nanmean(hgt, axis=-1, keepdims=True), (1, 1, hgt.shape[-1])) if hgt.ndim == 3 else np.tile(np.nanmean(hgt, axis=1, keepdims=True), (1, hgt.shape[1]))

        raw_arr_k = hgt - clim

    elif param_config.get("unit") == "dam" or str(param_config.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
        raw_arr_k = raw_arr_k / 98.0665
    elif param_config.get("unit") in {"mb", "hPa"} or str(param_config.get("grib_param", "")).lower() in {"msl", "mslp"}:
        raw_arr_k = raw_arr_k / 100.0

    # 🌟 Extract contours for EACH member rather than averaging across axis=0
    if raw_arr_k.ndim == 3:
        contour_levels = [
            extract_contour_levels(
                raw_arr_k[m],
                param_config.get("contours", []),
                param_config.get("contours_settings")
            )
            for m in range(raw_arr_k.shape[0])
        ]
    else:
        contour_levels = [extract_contour_levels(
            raw_arr_k,
            param_config.get("contours", []),
            param_config.get("contours_settings")
        )]

    if want_raw:
        return np.nan_to_num(raw_arr_k, nan=0.0).astype(np.float32), contour_levels

    arr_8bit = normalize_array(raw_arr_k, param_config) if want_raster else None
    return arr_8bit, contour_levels

def fetch_and_process_step(client, target_date, chosen_run, step, param_config, model_name):
    patterns = param_config["filename_patterns"]
    grib_file = patterns["grib"].format(model=model_name, param=param_config["id"], step=step)
    grib_p = param_config["grib_param"]
    is_interval = bool(param_config.get("interval_hours") or re.search(r'[-_]?(6|24)h(?:r)?$', param_config.get("id", "").lower()))

    if step == 0 and isinstance(grib_p, str) and grib_p in {"tp", "sf"}:
        zero_frame = np.zeros((NUM_MEMBERS, NA_GRID_H, NA_GRID_W), dtype=np.float32 if is_interval else np.uint8)
        zero_ptype = np.zeros((NUM_MEMBERS, NA_GRID_H, NA_GRID_W), dtype=np.uint8) if param_config.get("ptype_source") else None
        zero_packed = [pack_step([]) for _ in range(NUM_MEMBERS)]
        return step, zero_frame, zero_packed, zero_ptype

    contour_sources = param_config.get("contour_sources")
    if not contour_sources and param_config.get("contour_source"):
        contour_sources = [param_config["contour_source"]]

    has_contours = bool(contour_sources)
    raster_config = {**param_config, "contours": []} if has_contours else param_config

    # 🌟 Reuse cached 2t grid written by prate when available
    if param_config["id"] == "2t" and not contour_sources:
        cache_2t = _get_2t_cache_path(target_date, chosen_run, step)
        if os.path.exists(cache_2t):
            try:
                t2m_cached = np.load(cache_2t)
                preloaded = ({"2t": {2: t2m_cached}}, None, None)
                frame_arr, raw_contour_levels = process_grib_to_array(
                    None, raster_config, target_date=target_date, want_raw=is_interval, preloaded=preloaded
                )
                if raw_contour_levels and not isinstance(raw_contour_levels[0], list):
                    contour_levels = [list(raw_contour_levels)]
                else:
                    contour_levels = [list(m_l) for m_l in raw_contour_levels]
                packed = [pack_step(m_lev) for m_lev in contour_levels]
                print(f"  ⚡ [{param_config['id']}] Processed F{step:03d} (cached 2t)")
                return step, frame_arr, packed, None
            except Exception:
                pass

    base_params = list(grib_p) if isinstance(grib_p, list) else [grib_p]
    levtype = param_config.get("levtype", "sfc")
    ptype_source = param_config.get("ptype_source")
    need_prate_ptype = bool(ptype_source and param_config["id"] == "prate")

    batched_sfc_sources = []
    remaining_contour_sources = []
    if contour_sources:
        for c_src in contour_sources:
            if levtype == "sfc" and c_src.get("levtype", "sfc") == "sfc" and "levelist" not in c_src and c_src.get("type") != "thickness":
                batched_sfc_sources.append(c_src)
            else:
                remaining_contour_sources.append(c_src)

    retrieve_params = list(base_params)
    if levtype == "sfc":
        if need_prate_ptype:
            retrieve_params.append("2t")
        for c_src in batched_sfc_sources:
            retrieve_params.append(c_src["grib_param"])
    retrieve_params = list(dict.fromkeys(retrieve_params))

    retrieve_kwargs = {
        "date": target_date,
        "time": int(chosen_run),
        "step": step,
        "type": "pf",
        "levtype": levtype,
        "param": retrieve_params,
        "target": grib_file,
        "stream": "enfo",
        "number": MEMBER_LIST
    }

    if "levelist" in param_config:
        retrieve_kwargs["levelist"] = param_config["levelist"]

    try:
        with tm("download"):
            client.retrieve(**retrieve_kwargs)
        if os.path.exists(grib_file):
            with tm("decode_all"):
                by_param, lats_na, lons_na = read_grib_na(grib_file)

            try: os.remove(grib_file)
            except Exception: pass

            preloaded = (by_param, lats_na, lons_na)
            frame_arr, raw_contour_levels = process_grib_to_array(
                None, raster_config, target_date=target_date, want_raw=is_interval, preloaded=preloaded
            )

            # Ensure contour_levels is a list of lists (one per member)
            if raw_contour_levels and not isinstance(raw_contour_levels[0], list):
                contour_levels = [list(raw_contour_levels)]
            else:
                contour_levels = [list(m_l) for m_l in raw_contour_levels]

            for c_src in batched_sfc_sources:
                c_cfg = {**param_config, **c_src}
                _, levels = process_grib_to_array(
                    None, c_cfg, want_raster=False, target_date=target_date, preloaded=preloaded
                )
                if levels and not isinstance(levels[0], list):
                    levels = [levels]
                for m_idx, m_lev in enumerate(levels):
                    if m_idx < len(contour_levels):
                        contour_levels[m_idx].extend(m_lev)
                    else:
                        contour_levels.append(m_lev)

            ptype_arr = None

            # 🌟 AIFS-ENS Specific Ptype Derivation (reuses batched 2t from sfc GRIB)
            if need_prate_ptype:
                t_pl_file = f"{model_name}_{param_config['id']}_t_pl_{step:03d}.grib2"
                try:
                    t2m_dict = by_param.get("2t")
                    t2m_arr = next(iter(t2m_dict.values())) if t2m_dict else None

                    if t2m_arr is not None and os.path.isdir(SHARED_2T_CACHE_DIR):
                        cache_2t = _get_2t_cache_path(target_date, chosen_run, step)
                        tmp_2t = os.path.join(SHARED_2T_CACHE_DIR, f".tmp_{step:03d}.npy")
                        np.save(tmp_2t, t2m_arr)
                        os.replace(tmp_2t, cache_2t)

                    with tm("download_ptype"):
                        client.retrieve(
                            date=target_date, time=int(chosen_run), step=step,
                            stream="enfo", type="pf", levtype="pl", levelist=[850, 700, 500], param=["t"],
                            number=MEMBER_LIST, target=t_pl_file
                        )

                    if t2m_arr is not None and os.path.exists(t_pl_file):
                        with tm("decode_all"):
                            t_by_param, _, _ = read_grib_na(t_pl_file)
                        t_pl_dict = t_by_param.get("t") or t_by_param[next(iter(t_by_param))]
                        col_stack = np.stack([t_pl_dict[lvl] for lvl in sorted(t_pl_dict)], axis=0)
                        max_col_t = np.max(np.where((col_stack > 100.0) & (col_stack < 350.0), col_stack, -np.inf), axis=0)

                        ptype_arr = np.zeros(t2m_arr.shape, dtype=np.uint8)
                        rain_mask = t2m_arr > 274.817
                        ptype_arr[rain_mask] = 1

                        surf_freeze = ~rain_mask
                        warm_nose = max_col_t > 273.15
                        ptype_arr[surf_freeze & warm_nose] = 3
                        ptype_arr[surf_freeze & ~warm_nose] = 5

                        if frame_arr is not None:
                            ptype_arr[frame_arr == 0] = 0
                except Exception as e:
                    print(f"  ⚠️ [{param_config['id']}] AIFS Ptype derivation failed F{step:03d}: {e}")
                finally:
                    if os.path.exists(t_pl_file):
                        try: os.remove(t_pl_file)
                        except: pass

            if remaining_contour_sources:
                for c_src in remaining_contour_sources:
                    c_id = c_src.get("id", c_src.get("grib_param", "contour"))
                    c_grib_file = patterns["grib"].format(
                        model=model_name,
                        param=f"{param_config['id']}_{c_id}",
                        step=step
                    )
                    c_kwargs = dict(retrieve_kwargs)
                    c_kwargs.update({
                        "param": [c_src["grib_param"]],
                        "levtype": c_src.get("levtype", "sfc"),
                        "target": c_grib_file
                    })
                    if "levelist" in c_src:
                        c_kwargs["levelist"] = c_src["levelist"]
                    elif "levelist" in c_kwargs:
                        del c_kwargs["levelist"]

                    try:
                        with tm("download_contour"):
                            client.retrieve(**c_kwargs)
                        if os.path.exists(c_grib_file):
                            if c_src.get("type") == "thickness":
                                levels = process_thickness_grib(c_grib_file, c_src)
                            else:
                                c_cfg = {**param_config, **c_src}
                                _, levels = process_grib_to_array(
                                    c_grib_file, c_cfg, want_raster=False, target_date=target_date
                                )

                            if levels and not isinstance(levels[0], list):
                                levels = [levels]

                            for m_idx, m_lev in enumerate(levels):
                                if m_idx < len(contour_levels):
                                    contour_levels[m_idx].extend(m_lev)
                                else:
                                    contour_levels.append(m_lev)
                    except Exception as e:
                        print(f"  ❌ [{param_config['id']}] Contour error ({c_id}) F{step:03d}: {e}")
                    finally:
                        if os.path.exists(c_grib_file):
                            try: os.remove(c_grib_file)
                            except Exception: pass

            packed = [pack_step(m_lev) for m_lev in contour_levels]

            print(f"  ⚡ [{param_config['id']}] Processed F{step:03d}")
            return step, frame_arr, packed, ptype_arr
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None, None, None

@timed("stack")
def build_volume_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run, ptype_arrays=None):
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape[-2:]
    frames_per_volume = 10
    num_members = NUM_MEMBERS
    members_str = MEMBERS_STR

    chunks = []
    patterns = param_config["filename_patterns"]
    pattern_key = "volume" if "volume" in patterns else "spritesheet"

    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_volume)):
        chunk_frames = frame_arrays[i:i + frames_per_volume]
        chunk_steps = steps_written[i:i + frames_per_volume]
        chunk_ptype = ptype_arrays[i:i + frames_per_volume] if ptype_arrays else None

        volume_arr = np.stack(chunk_frames, axis=0).transpose(1, 0, 2, 3).astype(np.uint8)
        
        volume_filename = patterns[pattern_key].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            members=members_str,
            chunk_idx=chunk_idx
        )

        chunk_manifest = {
            "file": volume_filename,
            "forecast_steps": chunk_steps,
            "frame_count": len(chunk_frames)
        }
        chunk_entry = {
            "array": volume_arr,
            "manifest_data": chunk_manifest
        }
        
        if chunk_ptype and all(frame is not None for frame in chunk_ptype):
            ptype_filename = volume_filename.replace("_volume_", "_ptype_volume_")
            ptype_stacked = np.stack(chunk_ptype, axis=0).transpose(1, 0, 2, 3)
            chunk_entry["ptype_array"] = ptype_stacked.astype(np.uint8)
            chunk_manifest["ptype_file"] = ptype_filename
            
        chunks.append(chunk_entry)

    return chunks, frame_w, frame_h

def upload_single_file(s3_client, bucket_name, filepath, filename):
    if os.path.getsize(filepath) < 100:
        return
    content_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    try:
        with open(filepath, 'rb') as f:
            s3_client.put_object(Bucket=bucket_name, Key=filename, Body=f, ContentType=content_type)
        print(f"  ✅ Uploaded to B2: {filename}")
    except Exception as e:
        print(f"  ❌ Failed to upload {filename}: {e}")

def upload_to_huggingface(folder_path, repo_id=HF_REPO_ID):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment. Skipping Hugging Face upload.")
        return

    print(f"\n🤗 Uploading {folder_path} assets to Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload ensemble model grids ({MEMBERS_STR}): {folder_path}"
        )
        print(f"  ✅ Uploaded to Hugging Face: {folder_path}")
    except Exception as e:
        print(f"  ❌ Failed to upload to Hugging Face: {e}")

def prune_old_huggingface_runs(repo_id=HF_REPO_ID, days_to_keep=7):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return

    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))

    print(f"\n🧹 Checking for Hugging Face runs older than {cutoff_date.strftime('%Y-%m-%d')} (>{days_to_keep} days)...")

    try:
        from huggingface_hub import HfApi, CommitOperationDelete
        api = HfApi(token=hf_token)
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        files_to_delete = []

        for fname in all_files:
            match = re.search(r'_(\d{8})_', fname)
            if match:
                file_date_int = int(match.group(1))
                if file_date_int < cutoff_int:
                    files_to_delete.append(fname)

        if files_to_delete:
            print(f"  🗑️ Deleting {len(files_to_delete)} expired files older than {days_to_keep} days...")
            batch_size = 200
            for i in range(0, len(files_to_delete), batch_size):
                batch = files_to_delete[i:i + batch_size]
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Prune runs older than {days_to_keep} days ({i + len(batch)}/{len(files_to_delete)})"
                )
            print("  ✅ Pruning complete!")
    except Exception as e:
        print(f"  ⚠️ Could not prune old Hugging Face files: {e}")

def run_master_pipeline(selected_param_key="2t"):
    MODEL_NAME = "aifs_ens"
    param_config = load_parameter_config(selected_param_key)
    patterns = param_config["filename_patterns"]

    output_dist_dir = f"run_{param_config['id']}_{MEMBERS_STR}"

    TIMINGS.clear()
    t_total = time.perf_counter()
    phases = {}

    if MANUAL_RUN_DATE and MANUAL_RUN_CYCLE:
        CHOSEN_RUN = str(MANUAL_RUN_CYCLE).zfill(2)
        target_date = str(MANUAL_RUN_DATE)
    else:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_hour = now_utc.hour

        if current_hour >= 20:
            CHOSEN_RUN, target_date = "12", now_utc.strftime("%Y%m%d")
        elif current_hour >= 14:
            CHOSEN_RUN, target_date = "06", now_utc.strftime("%Y%m%d")
        elif current_hour >= 8:
            CHOSEN_RUN, target_date = "00", now_utc.strftime("%Y%m%d")
        elif current_hour >= 2:
            CHOSEN_RUN = "18"
            target_date = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")
        else:
            CHOSEN_RUN = "12"
            target_date = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")

    init_time_iso = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}T{CHOSEN_RUN}:00:00Z"

    print(f"🌍 [{param_config['id']}] Model: {MODEL_NAME} | Members: {MEMBERS_STR} | Param: {param_config['name']} | Run: {CHOSEN_RUN}z on {target_date}")

    for f in glob.glob(f"{MODEL_NAME}_{param_config['id']}_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    client = Client(source="azure", model="aifs-ens", resol="0p25")
    os.makedirs(output_dist_dir, exist_ok=True)

    t0 = time.perf_counter()
    results = {}
    contour_steps = {}
    ptype_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, client, target_date, CHOSEN_RUN, step, param_config, MODEL_NAME): step
            for step in FORECAST_STEPS
        }
        for future in concurrent.futures.as_completed(future_to_step):
            step, arr, packed, ptype_arr = future.result()
            if arr is not None:
                results[step] = arr
                if packed and any(m.get("props") for m in packed):
                    contour_steps[step] = packed
                if ptype_arr is not None:
                    ptype_results[step] = ptype_arr

    phases["fetch_phase"] = time.perf_counter() - t0
    print(f"[{param_config['id']}] steps ok: {len(results)}/{len(FORECAST_STEPS)}", flush=True)

    sorted_steps = sorted(results.keys())

    interval_h = param_config.get("interval_hours")
    if not interval_h:
        match = re.search(r'[-_]?(6|24)h(?:r)?$', param_config.get("id", "").lower())
        if match:
            interval_h = int(match.group(1))

    if interval_h:
        diff_results = {}
        for s in sorted_steps:
            target_prev = s - interval_h
            if target_prev <= 0:
                prev_arr = results.get(0, 0.0)
            else:
                candidates = [st for st in sorted_steps if st <= target_prev]
                prev_s = max(candidates) if candidates else None
                prev_arr = results[prev_s] if prev_s is not None else 0.0

            diff = np.maximum(0.0, results[s] - prev_arr)
            
            if diff.max() > 0:
                if diff.ndim == 3:
                    for m_idx in range(diff.shape[0]):
                        if diff[m_idx].max() > 0:
                            diff[m_idx] = cv2.erode(diff[m_idx], np.ones((2, 2), np.uint8))
                else:
                    diff = cv2.erode(diff, np.ones((2, 2), np.uint8))

            diff_results[s] = normalize_array(diff, param_config)
        results = diff_results

    frame_arrays = [results[s] for s in sorted_steps]
    ptype_arrays = [ptype_results.get(s) for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print(f"❌ [{param_config['id']}] No frames processed. Exiting pipeline.")
        return

    t0 = time.perf_counter()
    if contour_steps and "run_contours" in patterns:
        flat_binary_steps = {}
        json_steps = {}

        for step in sorted(contour_steps):
            member_pack_list = contour_steps[step]
            if isinstance(member_pack_list, list):
                for m, m_pack in enumerate(member_pack_list):
                    bin_step_id = (step * 1000) + m
                    flat_binary_steps[bin_step_id] = m_pack
                    json_steps[f"{step}_m{m}"] = {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "MultiLineString", "coordinates": []},
                                "properties": props,
                            }
                            for props in m_pack["props"]
                        ],
                    }
            else:
                flat_binary_steps[step] = member_pack_list
                json_steps[str(step)] = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "MultiLineString", "coordinates": []},
                            "properties": props,
                        }
                        for props in member_pack_list["props"]
                    ],
                }

        contour_binary = assemble_contour_file(flat_binary_steps)
        compressed_contours = gzip.compress(contour_binary, compresslevel=6)

        if len(compressed_contours) >= 100:
            master_contours = {
                "model": MODEL_NAME,
                "parameter": param_config["id"],
                "run": f"{CHOSEN_RUN}z",
                "date": target_date,
                "members": NUM_MEMBERS,
                "member_range": MEMBERS_STR,
                "member_ids": MEMBER_IDS,
                "type": "ensemble",
                "steps": json_steps,
            }

            c_pattern = patterns["run_contours"]
            if "{members}" in c_pattern:
                run_contour_filename = c_pattern.format(
                    model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower(), members=MEMBERS_STR
                )
            elif MEMBERS_STR != "m01-m10":
                run_contour_filename = c_pattern.replace(".json", f"_{MEMBERS_STR}.json").format(
                    model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
                )
            else:
                run_contour_filename = c_pattern.format(
                    model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
                )

            run_binary_filename = run_contour_filename.replace(".json", ".bin.gz")
            master_contours["binary"] = {
                "format": CONTOUR_FORMAT,
                "scale": CONTOUR_COORD_SCALE,
                "file": run_binary_filename,
            }

            print(f"  📦 [{param_config['id']}] Contours ({MEMBERS_STR}): {len(contour_binary)/1e6:.2f} MB raw -> {len(compressed_contours)/1e6:.2f} MB gzip")

            with open(os.path.join(output_dist_dir, run_contour_filename), 'w') as f:
                json.dump(master_contours, f, separators=(",", ":"))
            with open(os.path.join(output_dist_dir, run_binary_filename), "wb") as f:
                f.write(compressed_contours)

            if "latest_contours" in patterns:
                lc_pattern = patterns["latest_contours"]
                if "{members}" in lc_pattern:
                    latest_contour_filename = lc_pattern.format(
                        model=MODEL_NAME, param=param_config["id"], members=MEMBERS_STR
                    )
                elif MEMBERS_STR != "m01-m10":
                    latest_contour_filename = lc_pattern.replace(".json", f"_{MEMBERS_STR}.json").format(
                        model=MODEL_NAME, param=param_config["id"]
                    )
                else:
                    latest_contour_filename = lc_pattern.format(
                        model=MODEL_NAME, param=param_config["id"]
                    )
                with open(os.path.join(output_dist_dir, latest_contour_filename), 'w') as f:
                    json.dump(master_contours, f, separators=(",", ":"))
    phases["contour_file"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks, frame_w, frame_h = build_volume_chunks(
        frame_arrays,
        steps_written,
        model_name=MODEL_NAME,
        param_config=param_config,
        target_date=target_date,
        chosen_run=CHOSEN_RUN,
        ptype_arrays=ptype_arrays if ptype_results else None
    )

    manifest_chunks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=GZIP_THREADS) as gz_pool:
        gz_futures = []
        for chunk in chunks:
            filename = chunk["manifest_data"]["file"]
            filepath = os.path.join(output_dist_dir, filename)

            if filename.endswith(".bin"):
                gz_futures.append(gz_pool.submit(_gzip_to_file, filepath, chunk["array"].tobytes()))
                
                ptype_filename = chunk["manifest_data"].get("ptype_file")
                if ptype_filename and "ptype_array" in chunk:
                    gz_futures.append(gz_pool.submit(
                        _gzip_to_file,
                        os.path.join(output_dist_dir, ptype_filename),
                        chunk["ptype_array"].tobytes(),
                    ))

            manifest_chunks.append(chunk["manifest_data"])

        for fut in gz_futures:
            fut.result()
    phases["volumes+gzip"] = time.perf_counter() - t0

    manifest = {
        "model": MODEL_NAME,
        "parameter": param_config["id"],
        "name": param_config.get("name", param_config["id"]),
        "unit": param_config.get("unit", ""),
        "scaling": param_config.get("scaling", {}),
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "ensemble_chunked",
        "members": NUM_MEMBERS,
        "member_range": MEMBERS_STR,
        "member_ids": MEMBER_IDS,
        "frames_per_member": 10,
        "member_size_bytes": frame_w * frame_h * 10,
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "bounds": {
            "west": NA_LON_MIN,
            "east": NA_LON_MAX,
            "south": NA_LAT_MIN,
            "north": NA_LAT_MAX
        },
        "temp_min_k": param_config.get("min_val", param_config.get("scaling", {}).get("min_val", 0.0)),
        "temp_max_k": param_config.get("max_val", param_config.get("scaling", {}).get("max_val", 255.0)),
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    run_manifest_filename = patterns["run_manifest"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower(), members=MEMBERS_STR
    )

    manifest_files_to_write = [run_manifest_filename]

    if MEMBERS_STR == "m01-m10":
        manifest_files_to_write.append("manifest.json")
    else:
        manifest_files_to_write.append(f"manifest_{MEMBERS_STR}.json")

    if "latest_manifest" in patterns:
        lm_pattern = patterns["latest_manifest"]
        if "{members}" in lm_pattern:
            latest_manifest_filename = lm_pattern.format(
                model=MODEL_NAME, param=param_config["id"], members=MEMBERS_STR
            )
        elif MEMBERS_STR != "m01-m10":
            latest_manifest_filename = lm_pattern.replace(".json", f"_{MEMBERS_STR}.json").format(
                model=MODEL_NAME, param=param_config["id"]
            )
        else:
            latest_manifest_filename = lm_pattern.format(
                model=MODEL_NAME, param=param_config["id"]
            )
        manifest_files_to_write.append(latest_manifest_filename)

    for m_fname in manifest_files_to_write:
        m_path = os.path.join(output_dist_dir, m_fname)
        with open(m_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    print(f"\n🎉 [{param_config['id']}] Assets ready in {output_dist_dir}/")

    t0 = time.perf_counter()
    upload_to_huggingface(output_dist_dir)
    phases["hf_upload"] = time.perf_counter() - t0

    try:
        shutil.rmtree(output_dist_dir)
        print(f"  ✅ [{param_config['id']}] Cleanup complete.")
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Failed to delete temp directory: {e}")

    _print_timing(param_config["id"], phases, time.perf_counter() - t_total)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_params = sys.argv[1:]
    else:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_params = list(data.get("parameters", {}).keys())
        else:
            target_params = ["2t"]

    share_2t = ("prate" in target_params and "2t" in target_params)
    if share_2t:
        target_params = ["prate", "2t"] + [p for p in target_params if p not in ("prate", "2t")]
        os.makedirs(SHARED_2T_CACHE_DIR, exist_ok=True)
        batch_size = 1
    else:
        batch_size = MAX_CONCURRENT_PARAMS

    print(f"🚀 Launching Pipeline for Parameters: {target_params} | Range: {MEMBERS_STR}")

    t_run = time.perf_counter()

    try:
        for i in range(0, len(target_params), batch_size):
            batch = target_params[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = math.ceil(len(target_params) / batch_size)
            t_batch = time.perf_counter()

            print(f"\n📦 [Batch {batch_num}/{total_batches}] Running {len(batch)} parameter(s) concurrently: {batch}")

            with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(run_master_pipeline, param): param for param in batch}
                for future in concurrent.futures.as_completed(futures):
                    param = futures[future]
                    try:
                        future.result()
                        print(f"✅ Finished parameter: {param}")
                    except Exception as e:
                        print(f"❌ Error processing parameter '{param}': {e}")

            print(f"⏱️ Batch {batch_num}/{total_batches} took {time.perf_counter() - t_batch:.1f}s", flush=True)
    finally:
        if os.path.isdir(SHARED_2T_CACHE_DIR):
            shutil.rmtree(SHARED_2T_CACHE_DIR, ignore_errors=True)

    print(f"\n🎉 ALL PARAMETERS COMPLETED FOR {MEMBERS_STR}!")
    
    # 🌟 Only let the m01-m10 runner prune expired runs so matrix jobs don't collide
    if MEMBERS_STR == "m01-m10":
        t_p = time.perf_counter()
        prune_old_huggingface_runs(days_to_keep=7)
        print(f"⏱️ prune {time.perf_counter() - t_p:.1f}s | pipeline total {time.perf_counter() - t_run:.1f}s", flush=True)
