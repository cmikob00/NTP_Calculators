"""
Preliminary Mars emergency-return (abort) delta-v model  --  CORRECTED
======================================================================

What this does
--------------
1. Launches a spacecraft from Earth with a DRA-5.0-informed TMI V-infinity.
2. Coasts it outbound on a Sun-centered two-body (conic) trajectory.
3. At several "days after TMI" an instantaneous abort burn is applied.
4. A Lambert solution finds the heliocentric velocity needed to intercept
   Earth after a chosen return time; the abort delta-v is the vector
   difference between that velocity and the coasting velocity.
5. The return time of flight is searched for the minimum delta-v.

Assumptions
-----------
- Circular, coplanar Earth and Mars orbits
- Sun-centered two-body propagation outbound
- Instantaneous abort burn
- Lambert (single-revolution, prograde) solution for Earth intercept
- Earth aerobraking / reentry delta-v excluded (arrival V-infinity IS
  reported so you can judge the entry speed)

IMPORTANT
---------
The initial outbound state is DRA-5.0-informed, not an exact reconstruction
of a NASA DRA trajectory. TMI V-infinity is assumed purely prograde/tangential
relative to Earth. This does not guarantee an actual Mars intercept.

BUGS FIXED relative to the original version
-------------------------------------------
(1) LAMBERT TRANSFER-ANGLE SIGN (the main bug, cause of the 5-20 km/s).
    The original used sin(dtheta) = +|r1 x r2| / (r1 r2) for every case, i.e.
    it ALWAYS assumed the transfer angle was < 180 deg. But a *prograde*
    (counter-clockwise) transfer from the spacecraft to Earth's future
    position often sweeps MORE than 180 deg. Whenever it did, the solver
    silently returned a retrograde transfer (flying the "wrong way" around
    the Sun), which costs tens of km/s. Fix: take the sign of sin(dtheta)
    from the z-component of r1 x r2, and use dtheta = 2*pi - dtheta when the
    prograde direction requires the long way (standard Curtis/Bate method).

(2) ROOT-BRACKETING ACROSS INVALID REGIONS.
    When y(z) < 0 the time function is undefined (NaN). The original kept the
    last valid sample and compared it to a sample on the far side of the
    invalid gap, which could produce a fake "sign change" and a bogus root.
    Fix: reset the bracket whenever an invalid sample is hit.

(3) RETURN-TIME WINDOW / ARRIVAL V-INFINITY (design note, not a bug).
    Minimizing departure delta-v alone favors slow returns (up to ~2 years)
    and can hide very high Earth-arrival speeds. The window is unchanged
    (150-350 d), the arrival V-infinity is now printed, and an optional
    MAX_ARRIVAL_VINF cap is provided.

ADDITIONS: Lambert now also returns arrival velocity, so the Earth-arrival
V-infinity (entry-speed driver) is reported; a Lambert self-test is included.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------
# CONSTANTS AND ORBITAL PARAMETERS
# ---------------------------------------------------------

AU = 149597870.7              # km
DAY = 86400.0                 # s
MU_SUN = 1.32712440018e11     # km^3/s^2

R_EARTH = 1.000000 * AU
R_MARS = 1.523679 * AU

# DRA 5.0 Addendum, representative first opportunity
TMI_VINF = 3.940              # km/s
DRA_OUTBOUND_DAYS = 190.0     # days

EMERGENCY_DAYS = [5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180]

# Return-duration search window (days after the abort burn).
# Kept at the original 150-350 d "quick return" window. Widen RETURN_MAX_DAYS
# to ~800 to allow ~2-year returns, which are cheaper in delta-v but are
# probably not acceptable for a crew (see notes in the header).
RETURN_MIN_DAYS = 1    # 150 default
RETURN_MAX_DAYS = 400  # 350 default
RETURN_STEP_DAYS = 2

# Optional cap on Earth-arrival V-infinity (km/s) to reject solutions that
# are cheap to depart but impossible to re-enter (no heat-shield margin).
# None = no cap. A value around 7-8 km/s is a reasonable capsule-entry limit.
MAX_ARRIVAL_VINF = None

# ---------------------------------------------------------
# PLANETARY CIRCULAR ORBITS
# ---------------------------------------------------------

def circular_speed(radius):
    return math.sqrt(MU_SUN / radius)

def angular_rate(radius):
    return math.sqrt(MU_SUN / radius**3)

V_EARTH = circular_speed(R_EARTH)
N_EARTH = angular_rate(R_EARTH)
N_MARS = angular_rate(R_MARS)

HOHMANN_TOF = (
    math.pi * math.sqrt(((R_EARTH + R_MARS) / 2)**3 / MU_SUN)
)
HOHMANN_DAYS = HOHMANN_TOF / DAY

def earth_position(t):
    """Earth heliocentric position at time t (s). Earth starts at (R,0), moves CCW."""
    theta = N_EARTH * t
    return np.array([
        R_EARTH * math.cos(theta),
        R_EARTH * math.sin(theta)
    ])

def earth_velocity(t):
    """Earth heliocentric velocity at time t (s) (circular, CCW)."""
    theta = N_EARTH * t
    return V_EARTH * np.array([-math.sin(theta), math.cos(theta)])

# ---------------------------------------------------------
# OUTBOUND SPACECRAFT MODEL
# ---------------------------------------------------------

def initial_state():
    """
    DRA-informed initial condition at Earth's position (t = 0).

    The TMI V-infinity is added prograde/tangentially to Earth's velocity.
    This is a simplifying assumption, not a fitted DRA trajectory.
    """
    r0 = np.array([R_EARTH, 0.0])
    v0 = np.array([0.0, V_EARTH + TMI_VINF])
    return r0, v0

def equations(t, state):
    """Sun-centered two-body equations of motion."""
    x, y, vx, vy = state
    r = math.sqrt(x*x + y*y)
    ax = -MU_SUN * x / r**3
    ay = -MU_SUN * y / r**3
    return [vx, vy, ax, ay]

def propagate_outbound(days):
    """Return heliocentric position and velocity at the given day after TMI."""
    r0, v0 = initial_state()
    state0 = [r0[0], r0[1], v0[0], v0[1]]

    solution = solve_ivp(
        equations,
        (0.0, days * DAY),
        state0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-8
    )

    final = solution.y[:, -1]
    return final[:2], final[2:]

# ---------------------------------------------------------
# LAMBERT SOLVER: UNIVERSAL-VARIABLE FORMULATION
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
    Solve Lambert's problem (single revolution, 2-D) with the universal
    variable method.

    Inputs:
        r1_vec, r2_vec : endpoint position vectors (km)
        tof            : time of flight (s)
        prograde       : True = counter-clockwise (same sense as the planets)

    Returns:
        (v1, v2): velocity at r1 and at r2 (km/s), heliocentric.
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dtheta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dtheta = float(np.clip(cos_dtheta, -1.0, 1.0))

    # z-component of r1 x r2 tells us which side r2 lies on.
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]

    # ------------------------------------------------------------
    # FIX #1: choose the transfer angle correctly.
    #   acos() only gives 0..pi.  For a prograde (CCW) transfer, if
    #   r2 is clockwise from r1 (cross_z < 0), the CCW sweep is
    #   2*pi - acos(), i.e. > 180 deg ("long way").  Vice-versa for
    #   a retrograde transfer.  Since sin(2*pi - x) = -sin(x), the
    #   long way simply flips the sign of sin(dtheta) and hence of A.
    # ------------------------------------------------------------
    sin_mag = math.sqrt(max(0.0, 1.0 - cos_dtheta**2))
    long_way = (cross_z < 0) if prograde else (cross_z >= 0)
    sin_dtheta = -sin_mag if long_way else sin_mag

    # Avoid the singular, collinear case.
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
        calculated_tof = (x**3 * S + A * math.sqrt(y)) / math.sqrt(MU_SUN)
        return calculated_tof - tof

    # Find a valid bracket for the universal variable z (single revolution:
    # z < 4*pi^2).
    grid = np.linspace(-100.0, 4.0 * math.pi**2 - 1e-5, 3000)
    bracket = None
    previous_z = None
    previous_f = None

    for z in grid:
        try:
            f = time_residual(float(z))
        except (ValueError, OverflowError):
            f = float("nan")

        # FIX #2: an invalid sample breaks the bracket chain, so we never
        # pair two samples that straddle an undefined region.
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

    # Lagrange coefficients
    f_lagrange = 1.0 - y / r1
    g_lagrange = A * math.sqrt(y / MU_SUN)
    gdot_lagrange = 1.0 - y / r2

    if abs(g_lagrange) < 1e-12:
        raise ValueError("Lambert solution has singular g coefficient.")

    v1 = (r2_vec - f_lagrange * r1_vec) / g_lagrange
    v2 = (gdot_lagrange * r2_vec - r1_vec) / g_lagrange
    return v1, v2

# ---------------------------------------------------------
# EMERGENCY RETURN DELTA-V
# ---------------------------------------------------------

def calculate_return(days_after_tmi):
    """
    Minimum-delta-v abort burn at `days_after_tmi`, searched over return TOF.
    """
    r_sc, v_sc = propagate_outbound(days_after_tmi)

    best = None

    for return_days in range(
        RETURN_MIN_DAYS,
        RETURN_MAX_DAYS + 1,
        RETURN_STEP_DAYS
    ):
        arrival_time = (days_after_tmi + return_days) * DAY
        r_earth_arrival = earth_position(arrival_time)

        try:
            v_departure, v_arrival = lambert(
                r_sc,
                r_earth_arrival,
                return_days * DAY,
                prograde=True
            )
        except (ValueError, OverflowError):
            continue

        delta_v_vector = v_departure - v_sc
        delta_v = np.linalg.norm(delta_v_vector)

        v_inf_arr = np.linalg.norm(v_arrival - earth_velocity(arrival_time))
        if MAX_ARRIVAL_VINF is not None and v_inf_arr > MAX_ARRIVAL_VINF:
            continue

        if best is None or delta_v < best["delta_v"]:
            v_inf_arrival = v_inf_arr
            best = {
                "delta_v": delta_v,
                "return_days": return_days,
                "v_departure": v_departure,
                "delta_v_vector": delta_v_vector,
                "v_inf_arrival": v_inf_arrival,
            }

    if best is None:
        raise RuntimeError(
            f"No return solution found at day {days_after_tmi}."
        )

    return best

# ---------------------------------------------------------
# SELF-TEST: validate the Lambert solver against a known answer
# ---------------------------------------------------------

def lambert_self_test():
    """
    Earth -> Mars Hohmann transfer: the Lambert solution over the Hohmann TOF
    between points 180 deg apart (use 179.99 deg to dodge the collinear
    singularity) must give a departure speed of ~32.73 km/s, i.e. a
    V-infinity of ~2.94 km/s relative to Earth.
    """
    ang = math.radians(179.99)
    r1 = np.array([R_EARTH, 0.0])
    r2 = R_MARS * np.array([math.cos(ang), math.sin(ang)])
    v1, _ = lambert(r1, r2, HOHMANN_TOF, prograde=True)
    vinf = np.linalg.norm(v1) - V_EARTH
    ok = abs(vinf - 2.943) < 0.02
    print(f"Lambert self-test (Earth->Mars Hohmann V-inf): "
          f"{vinf:.3f} km/s (expect ~2.943)  {'PASS' if ok else 'FAIL'}")
    return ok

# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    print("MARS EMERGENCY RETURN ANALYSIS")
    print("-" * 98)
    print(f"DRA 5.0 TMI V-infinity: {TMI_VINF:.3f} km/s")
    print(f"DRA 5.0 reference outbound TOF: {DRA_OUTBOUND_DAYS:.1f} days")
    print(f"Ideal circular-orbit Hohmann TOF: {HOHMANN_DAYS:.1f} days")
    print(f"Return search window: {RETURN_MIN_DAYS}-{RETURN_MAX_DAYS} days")
    lambert_self_test()
    print()
    print(
        f"{'Days after TMI':>14} | {'Return TOF (d)':>14} | "
        f"{'Total time (d)':>14} | {'Return delta-v (km/s)':>22} | "
        f"{'Earth arr. V-inf (km/s)':>23}"
    )
    print("-" * 98)

    for day in EMERGENCY_DAYS:
        try:
            result = calculate_return(day)
            total_time = day + result["return_days"]

            print(
                f"{day:14.0f} | "
                f"{result['return_days']:14.0f} | "
                f"{total_time:14.0f} | "
                f"{result['delta_v']:22.3f} | "
                f"{result['v_inf_arrival']:23.3f}"
            )

        except (ValueError, RuntimeError, OverflowError) as error:
            print(f"{day:14.0f} | No solution: {error}")
