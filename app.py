import math
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import numpy as np
import streamlit as st
from skyfield.api import Loader


# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Orion / Moon Viewer",
    layout="wide"
)

# -------------------------------
# Constants / defaults
# -------------------------------
EARTH_RADIUS_KM = 6378.137
MOON_RADIUS_KM = 1737.4
LOCAL_TZ = ZoneInfo("America/Edmonton")

DEFAULT_START_LOCAL = datetime(2026, 4, 6, 6, 0, 0, tzinfo=LOCAL_TZ)
DEFAULT_END_LOCAL   = datetime(2026, 4, 7, 6, 0, 0, tzinfo=LOCAL_TZ)

DEFAULT_MAIN_XMIN = -145000
DEFAULT_MAIN_XMAX = -120000

DEFAULT_AUTO_UPDATE_INTERVAL_MS = 120000
DEFAULT_PLAY_INTERVAL_MS = 1500

REPO_DIR = Path(__file__).resolve().parent

# -------------------------------
# Session state
# -------------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

if "playing" not in st.session_state:
    st.session_state.playing = False

if "last_data_signature" not in st.session_state:
    st.session_state.last_data_signature = None

if "ephemeris_slider" not in st.session_state:
    st.session_state.ephemeris_slider = 0


# -------------------------------
# Helpers
# -------------------------------
def parse_oem(file_path: Path, local_tz: ZoneInfo):
    times_utc = []
    times_local = []
    x_vals = []
    y_vals = []
    z_vals = []
    vx_vals = []
    vy_vals = []
    vz_vals = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) != 7:
                continue
            if "T" not in parts[0]:
                continue

            try:
                t_utc = datetime.fromisoformat(parts[0]).replace(tzinfo=timezone.utc)

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                vx = float(parts[4])
                vy = float(parts[5])
                vz = float(parts[6])

                times_utc.append(t_utc)
                times_local.append(t_utc.astimezone(local_tz))
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
                vx_vals.append(vx)
                vy_vals.append(vy)
                vz_vals.append(vz)

            except ValueError:
                continue

    if not times_utc:
        raise ValueError("No ephemeris data rows were parsed from the file.")

    return {
        "times_utc": times_utc,
        "times_local": times_local,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "vx": vx_vals,
        "vy": vy_vals,
        "vz": vz_vals,
    }


def filter_window(data, start_local, end_local):
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    times_utc_f = []
    times_local_f = []
    x_f = []
    y_f = []
    z_f = []
    vx_f = []
    vy_f = []
    vz_f = []

    for t_utc, t_local, x, y, z, vx, vy, vz in zip(
        data["times_utc"], data["times_local"], data["x"], data["y"], data["z"],
        data["vx"], data["vy"], data["vz"]
    ):
        if start_utc <= t_utc <= end_utc:
            times_utc_f.append(t_utc)
            times_local_f.append(t_local)
            x_f.append(x)
            y_f.append(y)
            z_f.append(z)
            vx_f.append(vx)
            vy_f.append(vy)
            vz_f.append(vz)

    if not times_utc_f:
        raise ValueError("No data points found in the selected time window.")

    return {
        "times_utc": times_utc_f,
        "times_local": times_local_f,
        "x": x_f,
        "y": y_f,
        "z": z_f,
        "vx": vx_f,
        "vy": vy_f,
        "vz": vz_f,
    }


def compute_velocity_terms(windowed):
    speed_f = []
    radial_f = []
    tangential_f = []

    for x, y, z, vx, vy, vz in zip(
        windowed["x"], windowed["y"], windowed["z"],
        windowed["vx"], windowed["vy"], windowed["vz"]
    ):
        r = math.sqrt(x * x + y * y + z * z)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        radial = (x * vx + y * vy + z * vz) / r
        tangential = math.sqrt(max(0.0, speed * speed - radial * radial))

        speed_f.append(speed)
        radial_f.append(radial)
        tangential_f.append(tangential)

    apogee_idx = speed_f.index(min(speed_f))
    apogee_time = windowed["times_utc"][apogee_idx]

    elapsed_hours_window = [
        (t - apogee_time).total_seconds() / 3600.0
        for t in windowed["times_utc"]
    ]

    return {
        "speed": speed_f,
        "radial": radial_f,
        "tangential": tangential_f,
        "apogee_idx": apogee_idx,
        "apogee_time": apogee_time,
        "elapsed_hours_window": elapsed_hours_window,
    }


@st.cache_resource
def load_skyfield_ephemeris(bsp_path_str: str):
    bsp_path = Path(bsp_path_str)
    load = Loader(str(bsp_path.parent))
    ts = load.timescale()
    eph = load(str(bsp_path.name))
    earth = eph["earth"]
    moon = eph["moon"]
    return ts, earth, moon


def compute_moon_positions(times_utc_f, bsp_path: Path):
    ts, earth, moon = load_skyfield_ephemeris(str(bsp_path))

    years   = [t.year for t in times_utc_f]
    months  = [t.month for t in times_utc_f]
    days    = [t.day for t in times_utc_f]
    hours   = [t.hour for t in times_utc_f]
    minutes = [t.minute for t in times_utc_f]
    seconds = [t.second + t.microsecond / 1e6 for t in times_utc_f]

    t_sf = ts.utc(years, months, days, hours, minutes, seconds)
    moon_geocentric = (moon - earth).at(t_sf)
    moon_xyz_km = moon_geocentric.position.km

    return {
        "x_moon": moon_xyz_km[0],
        "y_moon": moon_xyz_km[1],
        "z_moon": moon_xyz_km[2],
    }


def get_closest_past_idx(times_utc_f):
    now_utc = datetime.now(timezone.utc)
    past_indices = [i for i, t in enumerate(times_utc_f) if t <= now_utc]
    return past_indices[-1] if past_indices else None


def get_heading_angle(x_f, y_f, idx):
    back_steps = min(2, idx)
    forward_steps = min(2, len(x_f) - 1 - idx)

    if back_steps >= 1 and forward_steps >= 1:
        i0 = idx - back_steps
        i1 = idx + forward_steps
        dx = x_f[i1] - x_f[i0]
        dy = y_f[i1] - y_f[i0]
    elif forward_steps >= 1:
        i1 = idx + forward_steps
        dx = x_f[i1] - x_f[idx]
        dy = y_f[i1] - y_f[idx]
    elif back_steps >= 1:
        i0 = idx - back_steps
        dx = x_f[idx] - x_f[i0]
        dy = y_f[idx] - y_f[i0]
    else:
        dx, dy = 1.0, 0.0

    return math.degrees(math.atan2(dy, dx)) - 90


def line_segment_intersects_sphere_diagnostics(p0, p1, c, r):
    d = p1 - p0
    dd = np.dot(d, d)

    if dd == 0.0:
        dist = np.linalg.norm(p0 - c)
        return {
            "intersects": dist <= r,
            "closest_distance_km": dist,
            "radius_km": r,
            "t_unclamped": 0.0,
            "t_clamped": 0.0,
            "closest_point": p0.copy(),
            "segment_length_km": 0.0,
        }

    t = np.dot(c - p0, d) / dd
    t_clamped = max(0.0, min(1.0, t))
    closest = p0 + t_clamped * d
    dist = np.linalg.norm(closest - c)

    return {
        "intersects": dist <= r,
        "closest_distance_km": dist,
        "radius_km": r,
        "t_unclamped": t,
        "t_clamped": t_clamped,
        "closest_point": closest,
        "segment_length_km": math.sqrt(dd),
    }


def tangent_points_from_origin_to_moon_xy(x_moon, y_moon, idx, r):
    cx = float(x_moon[idx])
    cy = float(y_moon[idx])

    d2 = cx * cx + cy * cy
    d = math.sqrt(d2)

    if d <= r:
        return None

    factor1 = (r * r) / d2
    factor2 = r * math.sqrt(d2 - r * r) / d2

    t1x = cx * (1.0 - factor1) + cy * factor2
    t1y = cy * (1.0 - factor1) - cx * factor2

    t2x = cx * (1.0 - factor1) - cy * factor2
    t2y = cy * (1.0 - factor1) + cx * factor2

    t1 = np.array([t1x, t1y], dtype=float)
    t2 = np.array([t2x, t2y], dtype=float)

    return t1, t2


def build_earth_origin_tangent_ray_xy(tangent_xy):
    tx = float(tangent_xy[0])
    ty = float(tangent_xy[1])

    norm = math.hypot(tx, ty)
    if norm < 1e-12:
        return None

    ux = tx / norm
    uy = ty / norm

    L = 10_000_000.0
    return [0.0, L * ux], [0.0, L * uy]


def get_occultation_geometry(idx, x_f, y_f, z_f, x_moon, y_moon, z_moon, moon_radius_km):
    earth_center = np.array([0.0, 0.0, 0.0], dtype=float)
    s = np.array([x_f[idx], y_f[idx], z_f[idx]], dtype=float)
    m = np.array([x_moon[idx], y_moon[idx], z_moon[idx]], dtype=float)

    diag = line_segment_intersects_sphere_diagnostics(s, earth_center, m, moon_radius_km)
    blocked = diag["intersects"]

    earth_to_moon_km = np.linalg.norm(m)
    earth_to_sc_km = np.linalg.norm(s)
    moon_to_sc_km = np.linalg.norm(s - m)

    return {
        "blocked": blocked,
        "diag": diag,
        "earth_to_moon_km": earth_to_moon_km,
        "earth_to_sc_km": earth_to_sc_km,
        "moon_to_sc_km": moon_to_sc_km,
    }


def build_info_text(idx, start_local, end_local, times_local_f, x_f, y_f, z_f,
                    speed_f, radial_f, tangential_f, x_moon, y_moon, z_moon,
                    earth_radius_km, moon_radius_km, local_tz):
    now_local = datetime.now(timezone.utc).astimezone(local_tz)

    r = math.sqrt(x_f[idx] ** 2 + y_f[idx] ** 2 + z_f[idx] ** 2)
    altitude = r - earth_radius_km

    moon_range_km = math.sqrt(
        (x_f[idx] - x_moon[idx]) ** 2 +
        (y_f[idx] - y_moon[idx]) ** 2 +
        (z_f[idx] - z_moon[idx]) ** 2
    )
    moon_surface_distance_km = moon_range_km - moon_radius_km

    return (
        f"Window: {start_local.strftime('%Y-%m-%d %H:%M %Z')} to\n"
        f"{end_local.strftime('%Y-%m-%d %H:%M %Z')}\n"
        f"State local time: {times_local_f[idx].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Last update: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"3D velocity: {speed_f[idx]:,.2f} km/s\n"
        f"Tangential velocity: {tangential_f[idx]:,.2f} km/s\n"
        f"Radial velocity: {radial_f[idx]:,.2f} km/s\n"
        f"Earth altitude: {altitude:,.0f} km\n"
        f"Lunar altitude: {moon_surface_distance_km:,.0f} km"
    )


def build_occultation_diagnostic_text(idx, occ, times_local_f, x_moon, y_moon, moon_radius_km):
    t_local = times_local_f[idx].strftime("%Y-%m-%d %H:%M:%S %Z")
    diag = occ["diag"]
    blocked_txt = "YES" if occ["blocked"] else "NO"

    lines = [
        "Occultation diagnostics",
        f"State time: {t_local}",
        f"Blocked in 3D: {blocked_txt}",
        f"Closest line-to-Moon distance: {diag['closest_distance_km']:,.1f} km",
        f"Moon radius: {diag['radius_km']:,.1f} km",
        f"Intersection margin: {diag['radius_km'] - diag['closest_distance_km']:,.1f} km",
        f"Segment parameter t (raw): {diag['t_unclamped']:.4f}",
        f"Segment parameter t (clamped): {diag['t_clamped']:.4f}",
        f"Earth→Moon distance: {occ['earth_to_moon_km']:,.1f} km",
        f"Earth→SC distance: {occ['earth_to_sc_km']:,.1f} km",
        f"Moon→SC distance: {occ['moon_to_sc_km']:,.1f} km",
    ]

    if occ["blocked"]:
        tangents_xy = tangent_points_from_origin_to_moon_xy(
            x_moon, y_moon, idx, moon_radius_km
        )
        if tangents_xy is not None:
            t1, t2 = tangents_xy
            lines.append(f"Tangent 1 XY: ({t1[0]:,.1f}, {t1[1]:,.1f}) km")
            lines.append(f"Tangent 2 XY: ({t2[0]:,.1f}, {t2[1]:,.1f}) km")

    return "\n".join(lines)


def build_figure(windowed, vel, moon_pos, idx, main_xmin, main_xmax,
                 earth_radius_km, moon_radius_km, start_local, end_local,
                 local_tz, show_diag_box):
    x_f = windowed["x"]
    y_f = windowed["y"]
    z_f = windowed["z"]
    times_local_f = windowed["times_local"]

    speed_f = vel["speed"]
    radial_f = vel["radial"]
    tangential_f = vel["tangential"]
    elapsed_hours_window = vel["elapsed_hours_window"]

    x_moon = moon_pos["x_moon"]
    y_moon = moon_pos["y_moon"]
    z_moon = moon_pos["z_moon"]

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1.05, 2.35],
        height_ratios=[1, 1, 1],
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,
        wspace=0.25,
        hspace=0.30
    )

    ax_tan = fig.add_subplot(gs[0, 0])
    ax_rad = fig.add_subplot(gs[1, 0], sharex=ax_tan)
    ax_spd = fig.add_subplot(gs[2, 0], sharex=ax_tan)
    ax = fig.add_subplot(gs[:, 1])

    if len(x_f) >= 2:
        ax.plot(x_f, y_f, linewidth=2, label="Orion trajectory", zorder=3)
    else:
        ax.scatter(x_f[0], y_f[0], s=20, label="Orion trajectory", zorder=3)

    ax.scatter(x_f[0], y_f[0], s=40, color="orange", label="Window start", zorder=5)
    ax.scatter(x_f[-1], y_f[-1], s=40, color="purple", label="Window end", zorder=5)

    ax.plot(x_moon, y_moon, linestyle="--", linewidth=1.2, label="Moon trajectory", zorder=2)
    ax.scatter([x_moon[idx]], [y_moon[idx]], s=35, label="Moon current position", zorder=5)

    moon_circle = Circle(
        (x_moon[idx], y_moon[idx]),
        radius=moon_radius_km,
        fill=True,
        linewidth=1.5
    )
    ax.add_patch(moon_circle)

    angle = get_heading_angle(x_f, y_f, idx)
    ax.plot(
        [x_f[idx]], [y_f[idx]],
        linestyle="None",
        marker=(3, 0, angle),
        markersize=10,
        color="red",
        label="Current state",
        zorder=6
    )

    occ = get_occultation_geometry(
        idx, x_f, y_f, z_f, x_moon, y_moon, z_moon, moon_radius_km
    )

    if occ["blocked"]:
        tangents_xy = tangent_points_from_origin_to_moon_xy(
            x_moon, y_moon, idx, moon_radius_km
        )
        if tangents_xy is not None:
            t1_xy, t2_xy = tangents_xy
            ray1 = build_earth_origin_tangent_ray_xy(t1_xy)
            ray2 = build_earth_origin_tangent_ray_xy(t2_xy)

            if ray1 is not None and ray2 is not None:
                ax.plot(ray1[0], ray1[1], linestyle="-", linewidth=2, color="grey", zorder=8)
                ax.plot(ray2[0], ray2[1], linestyle="-", linewidth=2, color="grey", zorder=8)

                x_poly = [0.0, ray1[0][1], ray2[0][1]]
                y_poly = [0.0, ray1[1][1], ray2[1][1]]
                ax.fill(x_poly, y_poly, color="grey", alpha=0.2, zorder=7)

    plot_x = list(x_f) + list(x_moon)
    plot_y = list(y_f) + list(y_moon)

    x_min, x_max = min(plot_x), max(plot_x)
    y_min, y_max = min(plot_y), max(plot_y)

    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 1000.0
    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 1000.0

    ax.set_xlim(main_xmin, main_xmax)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("Orion + Moon Trajectories (EME2000 / J2000, X-Y View)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="lower left")

    info_text = build_info_text(
        idx, start_local, end_local, times_local_f, x_f, y_f, z_f,
        speed_f, radial_f, tangential_f, x_moon, y_moon, z_moon,
        earth_radius_km, moon_radius_km, local_tz
    )
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
        zorder=10
    )

    if show_diag_box:
        diag_text = build_occultation_diagnostic_text(
            idx, occ, times_local_f, x_moon, y_moon, moon_radius_km
        )
        ax.text(
            0.98, 0.98,
            diag_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="black"),
            zorder=10
        )

    ax_tan.plot(elapsed_hours_window, tangential_f, linewidth=1.5)
    ax_tan.scatter([elapsed_hours_window[idx]], [tangential_f[idx]], s=40, zorder=5, color="red")
    ax_tan.set_title("Tangential Velocity")
    ax_tan.set_ylabel("km/s")
    ax_tan.grid(True)

    ax_rad.plot(elapsed_hours_window, radial_f, linewidth=1.5)
    ax_rad.scatter([elapsed_hours_window[idx]], [radial_f[idx]], s=40, zorder=5, color="red")
    ax_rad.set_title("Radial Velocity")
    ax_rad.set_ylabel("km/s")
    ax_rad.grid(True)

    ax_spd.plot(elapsed_hours_window, speed_f, linewidth=1.5)
    ax_spd.scatter([elapsed_hours_window[idx]], [speed_f[idx]], s=40, zorder=5, color="red")
    ax_spd.set_title("3D Velocity")
    ax_spd.set_xlabel("Hours relative to apogee")
    ax_spd.set_ylabel("km/s")
    ax_spd.grid(True)

    plt.setp(ax_tan.get_xticklabels(), visible=False)
    plt.setp(ax_rad.get_xticklabels(), visible=False)

    return fig


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Controls")

oem_path_text = st.sidebar.text_input(
    "OEM file path (repo-relative)",
    value="orion_oem.asc"
)
bsp_path_text = st.sidebar.text_input(
    "DE421 path (repo-relative)",
    value="de421.bsp"
)

main_xmin = st.sidebar.number_input("Main X min", value=DEFAULT_MAIN_XMIN)
main_xmax = st.sidebar.number_input("Main X max", value=DEFAULT_MAIN_XMAX)

auto_update_enabled = st.sidebar.toggle("Auto update", value=True)
play_enabled = st.sidebar.toggle("Play", value=st.session_state.playing)
show_diag_box = st.sidebar.toggle("Show occultation diagnostics", value=False)

auto_update_interval_ms = st.sidebar.number_input(
    "Auto update interval (ms)",
    min_value=100,
    value=DEFAULT_AUTO_UPDATE_INTERVAL_MS,
    step=100
)
play_interval_ms = st.sidebar.number_input(
    "Play interval (ms)",
    min_value=20,
    value=DEFAULT_PLAY_INTERVAL_MS,
    step=10
)

st.session_state.playing = play_enabled

oem_abs = REPO_DIR / oem_path_text
bsp_abs = REPO_DIR / bsp_path_text

if not oem_abs.exists():
    st.error(f"OEM file not found: {oem_abs}")
    st.stop()

if not bsp_abs.exists():
    st.error(f"Ephemeris file not found: {bsp_abs}")
    st.stop()

start_local = DEFAULT_START_LOCAL
end_local = DEFAULT_END_LOCAL

# -------------------------------
# Load / preprocess
# -------------------------------
raw = parse_oem(oem_abs, LOCAL_TZ)
windowed = filter_window(raw, start_local, end_local)
vel = compute_velocity_terms(windowed)
moon_pos = compute_moon_positions(windowed["times_utc"], bsp_abs)

n = len(windowed["times_utc"])
closest_past_idx = get_closest_past_idx(windowed["times_utc"])
default_idx = closest_past_idx if closest_past_idx is not None else n - 1

data_signature = (
    str(oem_abs),
    str(bsp_abs),
    n,
    windowed["times_utc"][0].isoformat(),
    windowed["times_utc"][-1].isoformat(),
)

if st.session_state.last_data_signature != data_signature:
    st.session_state.idx = default_idx
    st.session_state.ephemeris_slider = default_idx
    st.session_state.last_data_signature = data_signature

if st.session_state.idx < 0:
    st.session_state.idx = 0
if st.session_state.idx > n - 1:
    st.session_state.idx = n - 1

if st.session_state.ephemeris_slider < 0:
    st.session_state.ephemeris_slider = 0
if st.session_state.ephemeris_slider > n - 1:
    st.session_state.ephemeris_slider = n - 1

st.title("Orion / Moon Viewer")

# -------------------------------
# Timed rerun logic
# -------------------------------
run_every = None
if st.session_state.playing:
    run_every = play_interval_ms / 1000.0
elif auto_update_enabled:
    run_every = auto_update_interval_ms / 1000.0


@st.fragment(run_every=run_every)
def live_panel():
    # Automatic state update
    if st.session_state.playing:
        st.session_state.idx = (int(st.session_state.idx) + 1) % n
        st.session_state.ephemeris_slider = int(st.session_state.idx)

    elif auto_update_enabled:
        new_idx = get_closest_past_idx(windowed["times_utc"])
        if new_idx is not None:
            st.session_state.idx = int(new_idx)
            st.session_state.ephemeris_slider = int(new_idx)

    # Manual slider for scrubbing
    idx_from_slider = st.slider(
        "Ephemeris index",
        min_value=0,
        max_value=n - 1,
        value=int(st.session_state.ephemeris_slider),
        key="ephemeris_slider_widget",
    )

    # Manual scrubbing should control the plot when not playing / auto-updating
    if not st.session_state.playing and not auto_update_enabled:
        st.session_state.idx = int(idx_from_slider)
        st.session_state.ephemeris_slider = int(idx_from_slider)

    idx = int(st.session_state.idx)

    fig = build_figure(
        windowed=windowed,
        vel=vel,
        moon_pos=moon_pos,
        idx=idx,
        main_xmin=main_xmin,
        main_xmax=main_xmax,
        earth_radius_km=EARTH_RADIUS_KM,
        moon_radius_km=MOON_RADIUS_KM,
        start_local=start_local,
        end_local=end_local,
        local_tz=LOCAL_TZ,
        show_diag_box=show_diag_box,
    )

    st.pyplot(fig, clear_figure=True)

    current_time = windowed["times_local"][idx].strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"Current displayed state: {current_time}")

    if st.session_state.playing or auto_update_enabled:
        st.caption("Manual slider scrubbing is active when both Play and Auto update are off.")


live_panel()
