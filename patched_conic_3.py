"""
Preliminary Mars emergency-return (abort) delta-v model  --  3-D / REALISTIC PLANETS
====================================================================================

What this does
--------------
1. Launches a spacecraft from Earth (TMI) on a Sun-centered two-body (conic)
   outbound trajectory.
2. At several "days after TMI" an instantaneous abort burn is applied.
3. A Lambert solution finds the heliocentric velocity needed to intercept
   Earth after a chosen return time; the abort delta-v is the vector
   difference between that velocity and the coasting velocity.
4. The return time of flight is searched for the minimum delta-v.
5. Trajectories are plotted (top-down, side view, and 3-D) and optionally
   animated.

Planet models (set in RUN_CASES)
--------------------------------
"circular"  : the original model. Circular, coplanar orbits.  Outbound state
              is Earth's velocity plus a purely tangential TMI V-infinity.
              (Mars is placed where the outbound conic crosses Mars' orbit,
              purely so it can be drawn.)
"realistic" : Earth and Mars follow JPL/Standish mean Keplerian elements
              (J2000 epoch + linear secular rates, valid ~1800-2050):
              real eccentricity (Mars e = 0.093), Mars inclination
              (1.85 deg to the ecliptic), and the correct relative phasing
              on a real calendar date.  Everything is 3-D.

Outbound options for the "realistic" model
------------------------------------------
"tangential": same idea as the original: Earth velocity + TMI_VINF along
              Earth's velocity direction.  Does NOT aim at Mars, so this
              isolates the pure effect of eccentricity/inclination on the
              abort geometry.
"lambert"   : the outbound leg is a Lambert arc from Earth to Mars' actual
              position DRA_OUTBOUND_DAYS later.  The TMI V-infinity is then
              whatever that requires (compare it with the DRA 5.0 value of
              3.94 km/s).  The launch date is either LAUNCH_DATE or the best
              date found by scanning LAUNCH_SEARCH_* (lowest TMI V-infinity).

Assumptions that remain
-----------------------
- Sun-centered two-body propagation (no planetary gravity, no SRP)
- Instantaneous abort burn; single-revolution prograde Lambert return
- Earth aerobraking / reentry delta-v excluded (arrival V-infinity IS reported)
- Planet elements use mean elements + linear rates (no short-period terms),
  and Earth means the Earth-Moon barycenter
- No parking-orbit / patched-conic departure from LEO, no Mars gravity

Earlier fixes retained (see original header): correct Lambert transfer-angle
sign (long-way transfers), no root-bracketing across invalid regions, and the
arrival V-infinity is reported.  The Lambert routine already works in 3-D
because only the z-component of r1 x r2 is used to pick the direction.
"""

import math
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------
# CONSTANTS AND SETTINGS
# ---------------------------------------------------------

AU = 149597870.7              # km
DAY = 86400.0                 # s
MU_SUN = 1.32712440018e11     # km^3/s^2

R_EARTH = 1.000000 * AU       # circular model only
R_MARS = 1.523679 * AU        # circular model only

# DRA 5.0 Addendum, representative first opportunity
TMI_VINF = 3.940              # km/s (used by "tangential" outbound)
DRA_OUTBOUND_DAYS = 190.0     # days (used by "lambert" outbound)

EMERGENCY_DAYS = [5, 10, 15, 20, 25, 30]

RETURN_MIN_DAYS = 1
RETURN_MAX_DAYS = 300
RETURN_STEP_DAYS = 2

# Optional cap on Earth-arrival V-infinity (km/s). None = no cap.
MAX_ARRIVAL_VINF = None

# ---- Which cases to compute and compare --------------------------------
# (model, outbound).  outbound is ignored for "circular".
RUN_CASES = [
    ("circular", None),
    ("realistic", "tangential"),
    ("realistic", "lambert"),
]

# ---- Launch date for the realistic model --------------------------------
LAUNCH_DATE = None                 # e.g. "2037-03-15", or None to auto-search
LAUNCH_SEARCH_START = "2037-01-01"
LAUNCH_SEARCH_DAYS = 800
LAUNCH_SEARCH_STEP = 10

# ---- Plotting ------------------------------------------------------------
PLOT_CASE = ("realistic", "lambert")   # case used for trajectory plots
PLOT_DAYS = EMERGENCY_DAYS              # abort days to plot; [] to skip
PLOT_COMPARISON = True                 # delta-v / V-inf vs abort day, all cases
ANIMATE_DAY = None                     # e.g. 90 to animate that case
SAVE_ANIMATION = False                  # e.g. "abort.gif"
SAVE_FIGURES = True                   # True -> save PNGs

# ---------------------------------------------------------
# REAL-CALENDAR TIME  (seconds since J2000 = 2000-01-01 12:00)
# ---------------------------------------------------------

J2000 = datetime(2000, 1, 1, 12, 0, 0)

def date_to_t(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d - J2000).total_seconds()

def t_to_date(t):
    from datetime import timedelta
    return (J2000 + timedelta(seconds=float(t))).strftime("%Y-%m-%d")

# ---------------------------------------------------------
# PLANET EPHEMERIS
# ---------------------------------------------------------
# JPL "Keplerian elements for approximate positions" (Standish), 1800-2050.
# Each entry: (value at J2000, rate per Julian century).
# a [AU], e, I [deg], L = mean longitude [deg], varpi = longitude of
# perihelion [deg], node = longitude of ascending node [deg].
ELEMENTS = {
    "earth": {   # Earth-Moon barycenter
        "a":     (1.00000261,    0.00000562),
        "e":     (0.01671123,   -0.00004392),
        "I":     (-0.00001531,  -0.01294668),
        "L":     (100.46457166,  35999.37306),
        "varpi": (102.93768193,  0.32327364),
        "node":  (0.0,           0.0),
    },
    "mars": {
        "a":     (1.52371034,    0.00001847),
        "e":     (0.09339410,    0.00007882),
        "I":     (1.84969142,   -0.00813131),
        "L":     (-4.55343205,   19140.30268),
        "varpi": (-23.94362959,  0.44441088),
        "node":  (49.55953891,  -0.29257343),
    },
}
PERIOD_DAYS = {"earth": 365.256, "mars": 686.98}
CENTURY = 36525.0 * DAY

def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def kepler_state(name, t):
    """Heliocentric ecliptic (J2000) position/velocity (km, km/s) at t (s since J2000)."""
    el = ELEMENTS[name]
    T = t / CENTURY
    val = {k: el[k][0] + el[k][1] * T for k in el}

    a = val["a"] * AU
    e = val["e"]
    inc = math.radians(val["I"])
    node = math.radians(val["node"])
    omega = math.radians(val["varpi"] - val["node"])

    M = math.radians((val["L"] - val["varpi"]) % 360.0)
    if M > math.pi:
        M -= 2.0 * math.pi

    E = M if e < 0.8 else math.pi
    for _ in range(50):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break

    n_eff = math.radians(el["L"][1] - el["varpi"][1]) / CENTURY   # dM/dt
    Edot = n_eff / (1.0 - e * math.cos(E))
    b = a * math.sqrt(1.0 - e * e)

    r_pf = np.array([a * (math.cos(E) - e), b * math.sin(E), 0.0])
    v_pf = np.array([-a * math.sin(E) * Edot, b * math.cos(E) * Edot, 0.0])

    R = _Rz(node) @ _Rx(inc) @ _Rz(omega)
    return R @ r_pf, R @ v_pf

# ---- circular-model planets ----
def circular_speed(radius):
    return math.sqrt(MU_SUN / radius)

def angular_rate(radius):
    return math.sqrt(MU_SUN / radius**3)

V_EARTH = circular_speed(R_EARTH)
N_EARTH = angular_rate(R_EARTH)
N_MARS = angular_rate(R_MARS)

HOHMANN_TOF = math.pi * math.sqrt(((R_EARTH + R_MARS) / 2)**3 / MU_SUN)
HOHMANN_DAYS = HOHMANN_TOF / DAY

# ---- active configuration (set by configure()) ----
MODEL = "circular"
OUTBOUND = "tangential"
T_LAUNCH = 0.0            # absolute time (s since J2000) of TMI
_INIT = None              # (r0, v0) at TMI
_MARS_PHASE = (0.0, 0.0)  # circular model: (angle at t_ref, t_ref)

def earth_position(tau):
    """Earth heliocentric position (km) at mission time tau (s after TMI)."""
    if MODEL == "circular":
        th = N_EARTH * tau
        return np.array([R_EARTH * math.cos(th), R_EARTH * math.sin(th), 0.0])
    return kepler_state("earth", T_LAUNCH + tau)[0]

def earth_velocity(tau):
    if MODEL == "circular":
        th = N_EARTH * tau
        return V_EARTH * np.array([-math.sin(th), math.cos(th), 0.0])
    return kepler_state("earth", T_LAUNCH + tau)[1]

def mars_position(tau):
    if MODEL == "circular":
        th0, tref = _MARS_PHASE
        th = th0 + N_MARS * (tau - tref)
        return np.array([R_MARS * math.cos(th), R_MARS * math.sin(th), 0.0])
    return kepler_state("mars", T_LAUNCH + tau)[0]

def planet_orbit_curve(name, n=720):
    """Sampled path of one full orbit (used only for drawing)."""
    P = PERIOD_DAYS[name] * DAY
    pos = mars_position if name == "mars" else earth_position
    return np.array([pos(tau) for tau in np.linspace(0.0, P, n)])

# ---------------------------------------------------------
# OUTBOUND SPACECRAFT MODEL
# ---------------------------------------------------------

def initial_state():
    """Spacecraft heliocentric (r, v) at TMI for the active configuration."""
    return _INIT

def equations(t, state):
    """Sun-centered two-body equations of motion (3-D)."""
    x, y, z, vx, vy, vz = state
    r = math.sqrt(x*x + y*y + z*z)
    k = -MU_SUN / r**3
    return [vx, vy, vz, k * x, k * y, k * z]

def propagate_outbound(days):
    """Heliocentric position and velocity at `days` after TMI."""
    r0, v0 = initial_state()
    sol = solve_ivp(equations, (0.0, days * DAY),
                    np.concatenate([r0, v0]),
                    method="DOP853", rtol=1e-10, atol=1e-8)
    final = sol.y[:, -1]
    return final[:3], final[3:]

# ---------------------------------------------------------
# LAMBERT SOLVER: UNIVERSAL-VARIABLE FORMULATION (works in 3-D)
# ---------------------------------------------------------

def stumpff_C(z):
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    return 0.5

def stumpff_S(z):
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz**3
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0

def lambert(r1_vec, r2_vec, tof, prograde=True):
    """
    Single-revolution Lambert solution.  "Prograde" = counter-clockwise as
    seen from ecliptic north (same sense as the planets).

    Returns (v1, v2): heliocentric velocities at r1 and r2 (km/s).
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dtheta = float(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0))

    # z-component of r1 x r2 (valid for 3-D vectors) picks the sweep direction.
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]

    sin_mag = math.sqrt(max(0.0, 1.0 - cos_dtheta**2))
    long_way = (cross_z < 0) if prograde else (cross_z >= 0)
    sin_dtheta = -sin_mag if long_way else sin_mag

    if sin_mag < 1e-9:
        raise ValueError("Lambert geometry is nearly collinear.")

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))
    if abs(A) < 1e-12:
        raise ValueError("Lambert parameter A is too close to zero.")

    def y_of_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        return r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)

    def time_residual(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        if C <= 0:
            return float("nan")
        y = y_of_z(z)
        if y < 0:
            return float("nan")
        x = math.sqrt(y / C)
        return (x**3 * S + A * math.sqrt(y)) / math.sqrt(MU_SUN) - tof

    grid = np.linspace(-100.0, 4.0 * math.pi**2 - 1e-5, 3000)
    bracket = None
    previous_z = None
    previous_f = None

    for z in grid:
        try:
            f = time_residual(float(z))
        except (ValueError, OverflowError):
            f = float("nan")

        if not math.isfinite(f):
            previous_z, previous_f = None, None
            continue

        if previous_f is not None and previous_f * f < 0:
            bracket = (previous_z, float(z))
            break

        previous_z, previous_f = float(z), f

    if bracket is None:
        raise ValueError("No valid Lambert root found for this geometry.")

    z = brentq(time_residual, bracket[0], bracket[1], xtol=1e-12)
    y = y_of_z(z)

    f_lagrange = 1.0 - y / r1
    g_lagrange = A * math.sqrt(y / MU_SUN)
    gdot_lagrange = 1.0 - y / r2

    if abs(g_lagrange) < 1e-12:
        raise ValueError("Lambert solution has singular g coefficient.")

    v1 = (r2_vec - f_lagrange * r1_vec) / g_lagrange
    v2 = (gdot_lagrange * r2_vec - r1_vec) / g_lagrange
    return v1, v2

# ---------------------------------------------------------
# CASE CONFIGURATION
# ---------------------------------------------------------

def _setup_circular_mars():
    """
    Circular model: Mars phase is chosen so Mars is at the point where the
    outbound conic crosses Mars' orbit (drawing aid only).
    """
    global _MARS_PHASE
    sol = solve_ivp(equations, (0.0, 450 * DAY), np.concatenate(_INIT),
                    method="DOP853", rtol=1e-10, atol=1e-8,
                    t_eval=np.linspace(0.0, 450 * DAY, 901))
    rad = np.linalg.norm(sol.y[:3], axis=0)
    idx = np.argmax(rad >= R_MARS)
    if rad[idx] < R_MARS:
        _MARS_PHASE = (0.0, 0.0)
        return
    t_c = sol.t[idx]
    th_c = math.atan2(sol.y[1, idx], sol.y[0, idx])
    _MARS_PHASE = (th_c, t_c)

def find_launch_epoch():
    """Scan for the launch date with the lowest TMI V-infinity (Lambert outbound)."""
    global T_LAUNCH
    t0 = date_to_t(LAUNCH_SEARCH_START)
    best = None
    for d in range(0, int(LAUNCH_SEARCH_DAYS) + 1, int(LAUNCH_SEARCH_STEP)):
        T_LAUNCH = t0 + d * DAY
        try:
            v1, _ = lambert(earth_position(0.0),
                            mars_position(DRA_OUTBOUND_DAYS * DAY),
                            DRA_OUTBOUND_DAYS * DAY)
        except (ValueError, OverflowError):
            continue
        vinf = np.linalg.norm(v1 - earth_velocity(0.0))
        if best is None or vinf < best[0]:
            best = (vinf, T_LAUNCH)
    if best is None:
        raise RuntimeError("Launch-date search found no Lambert solution.")
    return best[1]

def configure(model, outbound=None):
    """Select planet model + outbound option and build the TMI state."""
    global MODEL, OUTBOUND, T_LAUNCH, _INIT
    MODEL = model
    OUTBOUND = outbound if model == "realistic" else "tangential"

    if model == "circular":
        T_LAUNCH = 0.0
        _INIT = (np.array([R_EARTH, 0.0, 0.0]),
                 np.array([0.0, V_EARTH + TMI_VINF, 0.0]))
        _setup_circular_mars()
        return

    if LAUNCH_DATE is not None:
        T_LAUNCH = date_to_t(LAUNCH_DATE)
    else:
        T_LAUNCH = find_launch_epoch()

    r0 = earth_position(0.0)
    v_e = earth_velocity(0.0)
    if OUTBOUND == "lambert":
        v0, _ = lambert(r0, mars_position(DRA_OUTBOUND_DAYS * DAY),
                        DRA_OUTBOUND_DAYS * DAY)
    else:
        v0 = v_e + TMI_VINF * v_e / np.linalg.norm(v_e)
    _INIT = (r0, v0)

def case_label(model, outbound):
    return "circular" if model == "circular" else f"realistic/{outbound}"

def mars_closest_approach(max_days=350):
    """Closest outbound-coast approach to Mars (km) and the day it occurs."""
    r0, v0 = initial_state()
    tau = np.linspace(0.0, max_days * DAY, int(max_days) + 1)
    sol = solve_ivp(equations, (0.0, tau[-1]), np.concatenate([r0, v0]),
                    method="DOP853", rtol=1e-10, atol=1e-8, t_eval=tau)
    d = [np.linalg.norm(sol.y[:3, i] - mars_position(tau[i])) for i in range(len(tau))]
    i = int(np.argmin(d))
    return d[i], tau[i] / DAY

# ---------------------------------------------------------
# EMERGENCY RETURN DELTA-V
# ---------------------------------------------------------

def calculate_return(days_after_tmi):
    """Minimum-delta-v abort burn at `days_after_tmi`, searched over return TOF."""
    r_sc, v_sc = propagate_outbound(days_after_tmi)
    best = None

    for return_days in range(RETURN_MIN_DAYS, RETURN_MAX_DAYS + 1, RETURN_STEP_DAYS):
        arrival_tau = (days_after_tmi + return_days) * DAY
        r_earth_arrival = earth_position(arrival_tau)

        try:
            v_departure, v_arrival = lambert(r_sc, r_earth_arrival,
                                             return_days * DAY, prograde=True)
        except (ValueError, OverflowError):
            continue

        delta_v_vector = v_departure - v_sc
        delta_v = np.linalg.norm(delta_v_vector)

        v_inf_arr = np.linalg.norm(v_arrival - earth_velocity(arrival_tau))
        if MAX_ARRIVAL_VINF is not None and v_inf_arr > MAX_ARRIVAL_VINF:
            continue

        if best is None or delta_v < best["delta_v"]:
            best = {
                "delta_v": delta_v,
                "return_days": return_days,
                "v_departure": v_departure,
                "delta_v_vector": delta_v_vector,
                "v_inf_arrival": v_inf_arr,
            }

    if best is None:
        raise RuntimeError(f"No return solution found at day {days_after_tmi}.")
    return best

# ---------------------------------------------------------
# SELF-TEST: validate the Lambert solver against a known answer
# ---------------------------------------------------------

def lambert_self_test():
    """Earth->Mars Hohmann V-infinity must come out ~2.943 km/s."""
    ang = math.radians(179.99)
    r1 = np.array([R_EARTH, 0.0, 0.0])
    r2 = R_MARS * np.array([math.cos(ang), math.sin(ang), 0.0])
    v1, _ = lambert(r1, r2, HOHMANN_TOF, prograde=True)
    vinf = np.linalg.norm(v1) - V_EARTH
    ok = abs(vinf - 2.943) < 0.02
    print(f"Lambert self-test (Earth->Mars Hohmann V-inf): "
          f"{vinf:.3f} km/s (expect ~2.943)  {'PASS' if ok else 'FAIL'}")
    return ok

# ---------------------------------------------------------
# TRAJECTORY HISTORY (for plotting)
# ---------------------------------------------------------

def outbound_path(days, n=400):
    r0, v0 = initial_state()
    t_eval = np.linspace(0.0, days * DAY, n)
    sol = solve_ivp(equations, (0.0, days * DAY), np.concatenate([r0, v0]),
                    method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-8)
    return sol.t, sol.y[:3].T

def return_path(r_start, v_start, return_days, n=400):
    t_eval = np.linspace(0.0, return_days * DAY, n)
    sol = solve_ivp(equations, (0.0, return_days * DAY),
                    np.concatenate([r_start, v_start]),
                    method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-8)
    return sol.t, sol.y[:3].T

def body_path(fn, t_start, t_end, n=400):
    return np.array([fn(t) for t in np.linspace(t_start, t_end, n)])

def build_case(day):
    """Everything needed to draw / animate one abort case (active config)."""
    res = calculate_return(day)
    t_abort = day * DAY
    t_arr = (day + res["return_days"]) * DAY

    t_out, p_out = outbound_path(day)
    r_sc, v_sc = propagate_outbound(day)
    t_ret, p_ret = return_path(r_sc, res["v_departure"], res["return_days"])

    return {"day": day, "res": res, "t_abort": t_abort, "t_arr": t_arr,
            "t_out": t_out, "p_out": p_out,
            "t_ret": t_ret + t_abort, "p_ret": p_ret,
            "r_abort": r_sc}

def _case_title(day, res):
    when = f", launch {t_to_date(T_LAUNCH)}" if MODEL == "realistic" else ""
    return (f"Mars abort at day {day} after TMI  [{case_label(MODEL, OUTBOUND)}{when}]\n"
            f"ΔV = {res['delta_v']:.2f} km/s (out-of-plane {res['delta_v_vector'][2]:+.2f}), "
            f"return {res['return_days']} d, Earth arrival V∞ = {res['v_inf_arrival']:.2f} km/s")

# ---------------------------------------------------------
# STATIC PLOT: top-down + side view + 3-D
# ---------------------------------------------------------

def _style_topdown(ax, lim=1.9):
    ax.plot(0, 0, "o", color="gold", ms=14, mec="orange", zorder=5, label="Sun")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.grid(alpha=0.25)

def plot_abort(day, show=True):
    """Top-down, edge-on and 3-D views of an abort at `day` days after TMI."""
    c = build_case(day)
    res = c["res"]

    e_orb = planet_orbit_curve("earth") / AU
    m_orb = planet_orbit_curve("mars") / AU
    po, pr = c["p_out"] / AU, c["p_ret"] / AU
    ep = body_path(earth_position, 0.0, c["t_arr"]) / AU
    mp = body_path(mars_position, 0.0, c["t_arr"]) / AU
    ab = c["r_abort"] / AU

    fig = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.35, 1.0], figure=fig)
    ax = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    axs = fig.add_subplot(gs[1, 1])

    # ---------------- top-down ----------------
    _style_topdown(ax)
    ax.plot(e_orb[:, 0], e_orb[:, 1], color="tab:blue", lw=0.8, ls=":", label="Earth orbit")
    ax.plot(m_orb[:, 0], m_orb[:, 1], color="tab:red", lw=0.8, ls=":", label="Mars orbit")
    ax.plot(po[:, 0], po[:, 1], color="tab:green", lw=2, label=f"Outbound coast ({day} d)")
    ax.plot(pr[:, 0], pr[:, 1], color="tab:orange", lw=2,
            label=f"Abort return arc ({res['return_days']} d)")
    ax.plot(ep[:, 0], ep[:, 1], color="tab:blue", lw=4, alpha=0.30, label="Earth during mission")
    ax.plot(mp[:, 0], mp[:, 1], color="tab:red", lw=4, alpha=0.25, label="Mars during mission")
    for t, mk in [(0.0, "o"), (c["t_abort"], "s"), (c["t_arr"], "*")]:
        e, m = earth_position(t) / AU, mars_position(t) / AU
        ax.plot(e[0], e[1], mk, color="tab:blue", ms=12 if mk == "*" else 9, mec="k", zorder=6)
        ax.plot(m[0], m[1], mk, color="tab:red", ms=12 if mk == "*" else 8, mec="k", zorder=6)
    ax.plot(ab[0], ab[1], "X", color="red", ms=13, mec="k", zorder=7,
            label=f"Abort burn (ΔV {res['delta_v']:.2f} km/s)")
    dv = res["delta_v_vector"]
    ax.annotate("", xy=ab[:2] + 0.18 * dv[:2] / np.linalg.norm(dv[:2]), xytext=ab[:2],
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.set_title("Top-down (ecliptic) view   ● TMI   ■ abort   ★ arrival", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)

    # ---------------- side view ----------------
    axs.plot(e_orb[:, 0], e_orb[:, 2], color="tab:blue", lw=0.8, ls=":")
    axs.plot(m_orb[:, 0], m_orb[:, 2], color="tab:red", lw=0.8, ls=":")
    axs.plot(po[:, 0], po[:, 2], color="tab:green", lw=2)
    axs.plot(pr[:, 0], pr[:, 2], color="tab:orange", lw=2)
    axs.plot(ab[0], ab[2], "X", color="red", ms=11, mec="k", zorder=7)
    axs.plot(0, 0, "o", color="gold", ms=10, mec="orange")
    axs.axhline(0, color="k", lw=0.5)
    zmax = max(0.06, 1.2 * max(np.abs(pr[:, 2]).max(), np.abs(po[:, 2]).max(),
                               np.abs(m_orb[:, 2]).max()))
    axs.set_xlim(-1.9, 1.9)
    axs.set_ylim(-zmax, zmax)
    axs.set_xlabel("x (AU)")
    axs.set_ylabel("z (AU)")
    axs.set_title("Edge-on view (z scale exaggerated)", fontsize=10)
    axs.grid(alpha=0.25)

    # ---------------- 3-D ----------------
    ax3.plot(e_orb[:, 0], e_orb[:, 1], e_orb[:, 2], color="tab:blue", lw=0.8, ls=":")
    ax3.plot(m_orb[:, 0], m_orb[:, 1], m_orb[:, 2], color="tab:red", lw=0.8, ls=":")
    ax3.plot(po[:, 0], po[:, 1], po[:, 2], color="tab:green", lw=2)
    ax3.plot(pr[:, 0], pr[:, 1], pr[:, 2], color="tab:orange", lw=2)
    ax3.plot(ep[:, 0], ep[:, 1], ep[:, 2], color="tab:blue", lw=4, alpha=0.3)
    ax3.plot(mp[:, 0], mp[:, 1], mp[:, 2], color="tab:red", lw=4, alpha=0.25)
    ax3.scatter([0], [0], [0], color="gold", s=120, edgecolors="orange")
    ax3.scatter([ab[0]], [ab[1]], [ab[2]], color="red", marker="X", s=90, edgecolors="k")
    ea = earth_position(c["t_arr"]) / AU
    ax3.scatter([ea[0]], [ea[1]], [ea[2]], color="tab:blue", marker="*", s=140, edgecolors="k")
    ax3.set_xlim(-1.9, 1.9)
    ax3.set_ylim(-1.9, 1.9)
    ax3.set_zlim(-zmax, zmax)
    ax3.set_box_aspect((1, 1, 0.35))
    ax3.set_xlabel("x (AU)")
    ax3.set_ylabel("y (AU)")
    ax3.set_zlabel("z (AU)")
    ax3.set_title("3-D view (z exaggerated)", fontsize=10)

    fig.suptitle(_case_title(day, res), fontsize=11)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(f"abort_{case_label(MODEL, OUTBOUND).replace('/', '_')}_day_{day}.png", dpi=150)
        print('plot saved!')
    if show:
        plt.show()
    return fig

# ---------------------------------------------------------
# ANIMATION: top-down + edge-on, Earth AND Mars moving
# ---------------------------------------------------------

def animate_abort(day, save_as=None, frames=300):
    c = build_case(day)
    res = c["res"]
    times = np.linspace(0.0, c["t_arr"], frames)

    e_orb = planet_orbit_curve("earth") / AU
    m_orb = planet_orbit_curve("mars") / AU
    earth_xyz = np.array([earth_position(t) for t in times]) / AU
    mars_xyz = np.array([mars_position(t) for t in times]) / AU

    t_all = np.concatenate([c["t_out"], c["t_ret"]])
    p_all = np.vstack([c["p_out"], c["p_ret"]]) / AU
    n_out = len(c["t_out"])

    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3.2, 1]})
    _style_topdown(ax)
    ax.plot(e_orb[:, 0], e_orb[:, 1], color="tab:blue", lw=0.7, ls=":")
    ax.plot(m_orb[:, 0], m_orb[:, 1], color="tab:red", lw=0.7, ls=":")
    ax.plot(*(c["r_abort"][:2] / AU), "X", color="red", ms=11, mec="k", zorder=6, label="Abort burn")
    l_out, = ax.plot([], [], color="tab:green", lw=2)
    l_ret, = ax.plot([], [], color="tab:orange", lw=2)
    sc, = ax.plot([], [], "o", color="k", ms=7, zorder=8, label="Spacecraft")
    ed, = ax.plot([], [], "o", color="tab:blue", ms=10, mec="k", zorder=7, label="Earth")
    md, = ax.plot([], [], "o", color="tab:red", ms=8, mec="k", zorder=7, label="Mars")
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(_case_title(day, res), fontsize=9)

    zmax = max(0.06, 1.2 * np.abs(p_all[:, 2]).max())
    axs.plot(e_orb[:, 0], e_orb[:, 2], color="tab:blue", lw=0.7, ls=":")
    axs.plot(m_orb[:, 0], m_orb[:, 2], color="tab:red", lw=0.7, ls=":")
    axs.plot(0, 0, "o", color="gold", ms=9, mec="orange")
    axs.axhline(0, color="k", lw=0.5)
    s_out, = axs.plot([], [], color="tab:green", lw=2)
    s_ret, = axs.plot([], [], color="tab:orange", lw=2)
    s_sc, = axs.plot([], [], "o", color="k", ms=6)
    s_md, = axs.plot([], [], "o", color="tab:red", ms=6)
    axs.set_xlim(-1.9, 1.9)
    axs.set_ylim(-zmax, zmax)
    axs.set_xlabel("x (AU)")
    axs.set_ylabel("z (AU)")
    axs.grid(alpha=0.25)

    def update(i):
        t = times[i]
        k = min(int(np.searchsorted(t_all, t)) + 1, len(t_all))
        seg = p_all[:k]
        k_out = min(k, n_out)
        l_out.set_data(p_all[:k_out, 0], p_all[:k_out, 1])
        s_out.set_data(p_all[:k_out, 0], p_all[:k_out, 2])
        if k > n_out:
            l_ret.set_data(p_all[n_out:k, 0], p_all[n_out:k, 1])
            s_ret.set_data(p_all[n_out:k, 0], p_all[n_out:k, 2])
        else:
            l_ret.set_data([], [])
            s_ret.set_data([], [])
        sc.set_data([seg[-1, 0]], [seg[-1, 1]])
        s_sc.set_data([seg[-1, 0]], [seg[-1, 2]])
        ed.set_data([earth_xyz[i, 0]], [earth_xyz[i, 1]])
        md.set_data([mars_xyz[i, 0]], [mars_xyz[i, 1]])
        s_md.set_data([mars_xyz[i, 0]], [mars_xyz[i, 2]])
        clock.set_text(f"Mission day {t / DAY:5.0f}")
        return l_out, l_ret, sc, ed, md, clock, s_out, s_ret, s_sc, s_md

    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
    if save_as:
        anim.save(save_as, fps=30)
    else:
        plt.show()
    return anim

# ---------------------------------------------------------
# COMPARISON PLOT
# ---------------------------------------------------------

def plot_comparison(results, show=True):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, rows in results.items():
        d = [r[0] for r in rows if r[1] is not None]
        dv = [r[1]["delta_v"] for r in rows if r[1] is not None]
        va = [r[1]["v_inf_arrival"] for r in rows if r[1] is not None]
        a1.plot(d, dv, "o-", label=label)
        a2.plot(d, va, "o-", label=label)
    a1.set_xlabel("Abort day after TMI")
    a1.set_ylabel("Minimum abort ΔV (km/s)")
    a1.set_title("Abort ΔV")
    a2.set_xlabel("Abort day after TMI")
    a2.set_ylabel("Earth-arrival V∞ (km/s)")
    a2.set_title("Earth arrival V∞")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig("abort_comparison.png", dpi=150)
        print('plot saved!')
    if show:
        plt.show()
    return fig

# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def run_case(model, outbound):
    """Configure, print the full table for one case, and return the rows."""
    configure(model, outbound)
    label = case_label(model, outbound)
    r0, v0 = initial_state()
    vinf_vec = v0 - earth_velocity(0.0)
    miss, miss_day = mars_closest_approach()

    print()
    print("=" * 98)
    print(f"CASE: {label}")
    if model == "realistic":
        print(f"  TMI date: {t_to_date(T_LAUNCH)}")
    print(f"  TMI V-infinity: {np.linalg.norm(vinf_vec):.3f} km/s "
          f"(out-of-plane component {vinf_vec[2]:+.3f} km/s)")
    print(f"  Outbound coast closest approach to Mars: {miss / AU:.3f} AU "
          f"({miss/1e6:.1f} million km) at day {miss_day:.0f}")
    print("-" * 98)
    print(f"{'Days after TMI':>14} | {'Return TOF (d)':>14} | {'Total time (d)':>14} | "
          f"{'Return dV (km/s)':>16} | {'dV out-of-plane':>15} | {'Earth arr. V-inf':>16}")
    print("-" * 98)

    rows = []
    for day in EMERGENCY_DAYS:
        try:
            r = calculate_return(day)
            rows.append((day, r))
            print(f"{day:14.0f} | {r['return_days']:14.0f} | {day + r['return_days']:14.0f} | "
                  f"{r['delta_v']:16.3f} | {r['delta_v_vector'][2]:+15.3f} | "
                  f"{r['v_inf_arrival']:16.3f}")
        except (ValueError, RuntimeError, OverflowError) as error:
            rows.append((day, None))
            print(f"{day:14.0f} | No solution: {error}")
    return rows

if __name__ == "__main__":
    print("MARS EMERGENCY RETURN ANALYSIS  (circular vs. realistic planets)")
    print(f"Return search window: {RETURN_MIN_DAYS}-{RETURN_MAX_DAYS} days")
    print(f"Reference DRA 5.0 TMI V-infinity: {TMI_VINF:.3f} km/s, "
          f"outbound {DRA_OUTBOUND_DAYS:.0f} d")
    lambert_self_test()

    results = {}
    for model, outbound in RUN_CASES:
        results[case_label(model, outbound)] = run_case(model, outbound)

    # ---- side-by-side delta-v summary ----
    labels = list(results)
    print()
    print("=" * 98)
    print("SUMMARY: minimum abort delta-v (km/s) by case")
    print("-" * 98)
    print(f"{'Days after TMI':>14} | " + " | ".join(f"{l:>20}" for l in labels))
    for i, day in enumerate(EMERGENCY_DAYS):
        cells = []
        for l in labels:
            r = results[l][i][1]
            cells.append(f"{r['delta_v']:20.3f}" if r else f"{'n/a':>20}")
        print(f"{day:14.0f} | " + " | ".join(cells))

    # ---- plots ----
    if PLOT_DAYS or ANIMATE_DAY is not None:
        configure(*PLOT_CASE)
        for day in PLOT_DAYS:
            try:
                plot_abort(day, show=False)
            except (ValueError, RuntimeError, OverflowError) as error:
                print(f"Plot skipped for day {day}: {error}")
        if ANIMATE_DAY is not None:
            animate_abort(ANIMATE_DAY, save_as=SAVE_ANIMATION)
    if PLOT_COMPARISON:
        plot_comparison(results, show=False)
    plt.show()