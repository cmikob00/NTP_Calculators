#!/usr/bin/env python3

"""
Post-TMI abort delta-v calculator for NASA DRA 5.0 NTR crewed vehicle
plus a dedicated bolt-on NTP abort stage (SI units).

Propulsion groups
-----------------
  "dra": 3 x 25 klbf Pewee-class engines on the DRA 5.0 core stage,
         fed by the LH2 still in the core stage after TMI.
  "ded": 1 x 25 klbf dedicated abort-stage engine, fed by its own tank.

Abort modes (all post-TMI; the DRA drop tank is already jettisoned)
-------------------------------------------------------------------
  "dedicated"  : dedicated engine only; DRA vehicle (and its remaining
                 LH2) is carried as dead mass.
  "dra"        : DRA 3 engines only, no dedicated stage (baseline).
  "sequential" : dedicated engine burns to depletion, the empty stage is
                 jettisoned, then the DRA engines burn.
  "all_four"   : all 4 engines fire together; each group shuts down when
                 its own tank is empty (no propellant cross-feed).

SOURCES (NASA Mars DRA 5.0 NTR numbers)
---------------------------------------
[1] Borowski, McCurdy, Packard, "7-Launch NTR Space Transportation
    System for NASA's Mars Design Reference Architecture (DRA) 5.0,"
    AIAA 2009-5308.  Three 25 klbf engines on the core stage; Isp ~900 s;
    LH2 flow ~12.6 kg/s/engine; engine T/W ~3.43; cooldown propellant
    margin 3 % of usable LH2.
[2] NTRS 20120003776 (NASA version of the 7-Launch paper).  Drop tank
    ~22 t dry / ~102.4 t LH2, jettisoned after TMI.
[3] NTRS 20120009207 (DRA 5.0 NTR crewed-vehicle paper).
    Crewed vehicle: NTR stage ~138.1 t, saddle truss +
    drop tank ~133.4 t, ~178.4 t total LH2, 4 primary burns; engine burn
    times ~55 min TMI / ~14.5 min MOC / ~9.7 min TEI (~79.2 min total);
    ~2 hr accumulated engine burn time demonstrated.
[4] NTRS 20120012928, "Nuclear Thermal Rocket/Vehicle Characteristics and
    Sensitivity Trades for NASA's Mars DRA 5.0 Study."  Alternate
    configurations (e.g. ~180 t LH2, ~80 min total burn); shows
    variation between DRA 5.0 papers.
[P] Primary reference: NASA-SP-2009-566, "Human Exploration of Mars
    Design Reference Architecture 5.0," and its
    Addendum (NASA/SP-2009-566-ADD).
"""

import math


# Standard gravity
G0 = 9.80665  # m/s^2

# Pounds-force to newtons
LBF_TO_N = 4.4482216

MODES = ("dedicated", "dra", "sequential", "all_four")

# Order in which propulsion groups fire in each mode. A step lists the
# groups that fire together; each step runs until the first active
# group's usable propellant is exhausted. "JETTISON" drops the empty
# dedicated stage.
_STEPS = {
    "dedicated":  [["ded"]],
    "dra":        [["dra"]],
    "sequential": [["ded"], ["JETTISON"], ["dra"]],
    "all_four":   [["dra", "ded"], ["dra", "ded"]],  # 2nd step: survivor
}


def run_abort_burn(mode: str,
                   isp_s: float,
                   m_dra_vehicle: float,
                   dra_core_lh2: float,
                   m_ded_dry: float,
                   m_ded_prop: float,
                   thrust_per_engine_N: float = 25000.0 * LBF_TO_N,
                   n_dra_engines: int = 3,
                   n_ded_engines: int = 1,
                   dra_reserve_fraction: float = 0.05,
                   ded_reserve_fraction: float = 0.02,
                   tmi_burn_min_per_dra_engine: float = 55.0,
                   engine_life_limit_min: float = 120.0) -> dict:
    
    """
    Compute ideal (impulsive) delta-v for a post-TMI abort.

    Parameters
    ----------
    mode : str
        One of MODES (see module docstring).

    isp_s : float
        Specific impulse [s] (same for all engines).

    m_dra_vehicle : float
        Total mass of the DRA vehicle at abort initiation (post-TMI,
        drop tank gone), including the LH2 still in its core stage, but
        EXCLUDING the dedicated abort stage [kg]

    dra_core_lh2 : float
        LH2 remaining in the DRA core stage at abort initiation [kg]
        (after TMI this is the propellant that MOC and TEI would use)

    m_ded_dry : float
        Dry mass of the dedicated stage (engine, tank, structure) [kg]

    m_ded_prop : float
        Loaded LH2 in the dedicated stage [kg]

    thrust_per_engine_N : float
        Thrust of each engine [N] (default 25 klbf)

    n_dra_engines, n_ded_engines : int
        Engine counts (DRA 5.0 core stage: 3; dedicated stage: 1)

    dra_reserve_fraction : float
        Fraction of core-stage LH2 not burned (cooldown, reserve,
        trapped residuals). DRA 5.0 carries a 3 % cooldown margin [1]
        plus performance reserve and residuals [3]; the extra ~2 % here
        is an ESTIMATE. Verify against [P].

    ded_reserve_fraction : float
        Same for the dedicated stage (default 2 %, as in the original
        script). Cooldown would also apply to a real NTR; raise this
        toward 0.05 for consistency with the DRA core stage.

    tmi_burn_min_per_dra_engine : float
        Engine time already used by each DRA engine at abort start
        [min]; ~55 min for TMI [3].

    engine_life_limit_min : float
        Accumulated burn-time limit per engine [min]; ~2 hr
        demonstrated on the ground [3].

    Returns
    -------
    dict with total delta-v, per-phase breakdown, and engine-life check
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    ve = G0 * isp_s                       # effective exhaust velocity
    mdot_eng = thrust_per_engine_N / ve   # kg/s per engine

    groups = {
        "dra": {
            "n": n_dra_engines,
            "thrust": n_dra_engines * thrust_per_engine_N,
            "mdot": n_dra_engines * mdot_eng,
            "prop": dra_core_lh2 * (1.0 - dra_reserve_fraction),
        },
        "ded": {
            "n": n_ded_engines,
            "thrust": n_ded_engines * thrust_per_engine_N,
            "mdot": n_ded_engines * mdot_eng,
            "prop": m_ded_prop * (1.0 - ded_reserve_fraction),
        },
    }

    # Mass carried and which groups are allowed to fire in this mode
    if mode == "dra":
        m = m_dra_vehicle                       # no dedicated stage
        groups["ded"]["prop"] = 0.0
        ded_inert = 0.0
    else:
        m = m_dra_vehicle + m_ded_dry + m_ded_prop
        ded_inert = m_ded_dry + m_ded_prop * ded_reserve_fraction
    if mode == "dedicated":
        groups["dra"]["prop"] = 0.0             # DRA engines stay idle

    m0_total = m
    engine_min = {"dra": tmi_burn_min_per_dra_engine, "ded": 0.0}
    ded_attached = mode != "dra"
    phases = []
    total_dv = 0.0
    total_time = 0.0

    for step in _STEPS[mode]:
        if step == ["JETTISON"]:
            if ded_attached:
                m -= ded_inert                  # drop the empty stage
                ded_attached = False
            continue

        active = [g for g in step if groups[g]["prop"] > 1e-9]
        if not active:
            continue

        thrust = sum(groups[g]["thrust"] for g in active)
        mdot = sum(groups[g]["mdot"] for g in active)
        dt = min(groups[g]["prop"] / groups[g]["mdot"] for g in active)
        burned = mdot * dt

        m_start = m
        m_end = m - burned
        dv = ve * math.log(m_start / m_end)

        for g in active:
            groups[g]["prop"] -= groups[g]["mdot"] * dt
            engine_min[g] += dt / 60.0

        phases.append({
            "engines": "+".join(active),
            "n_engines": sum(groups[g]["n"] for g in active),
            "thrust_N": thrust,
            "burn_time_s": dt,
            "propellant_burned_kg": burned,
            "m_start_kg": m_start,
            "m_end_kg": m_end,
            "delta_v_m_s": dv,
            "accel_start_g": thrust / m_start / G0,
            "accel_end_g": thrust / m_end / G0,
            "jet_power_MW": 0.5 * thrust * ve / 1.0e6,
        })

        m = m_end
        total_dv += dv
        total_time += dt

    used = {"dra": mode in ("dra", "sequential", "all_four"),
            "ded": mode in ("dedicated", "sequential", "all_four")}
    life_ok = all(engine_min[g] <= engine_life_limit_min
                  for g in ("dra", "ded") if used[g])

    return {
        "mode": mode,
        "delta_v_m_s": total_dv,
        "delta_v_km_s": total_dv / 1000.0,
        "m0_kg": m0_total,
        "mf_kg": m,
        "total_burn_time_s": total_time,
        "exhaust_velocity_m_s": ve,
        "mdot_per_engine_kg_s": mdot_eng,
        "phases": phases,
        "engine_minutes": engine_min,
        "engine_life_limit_min": engine_life_limit_min,
        "within_engine_life": life_ok,
    }


def print_result(r: dict) -> None:
    print(f"--- Mode: {r['mode']} ---")
    for i, p in enumerate(r["phases"], 1):
        print(f" Phase {i}: {p['engines']:8s} ({p['n_engines']} engine(s), "
              f"{p['thrust_N']/1000:6.1f} kN)")
        print(f"   burn time        : {p['burn_time_s']/60:7.2f} min")
        print(f"   propellant burned: {p['propellant_burned_kg']/1000:7.2f} t")
        print(f"   mass start/end   : {p['m_start_kg']/1000:7.1f} / "
              f"{p['m_end_kg']/1000:7.1f} t")
        print(f"   accel start/end  : {p['accel_start_g']:7.3f} / "
              f"{p['accel_end_g']:7.3f} g")
        print(f"   delta-v          : {p['delta_v_m_s']:7.0f} m/s")
        print(f"   jet power        : {p['jet_power_MW']:7.1f} MW")
    print(f" Total ideal delta-v : {r['delta_v_km_s']:6.3f} km/s "
          f"({r['delta_v_m_s']:.0f} m/s)")
    print(f" Total burn time     : {r['total_burn_time_s']/60:6.2f} min")
    print(f" Engine time (min)   : DRA engines {r['engine_minutes']['dra']:.1f}"
          f" incl. TMI, dedicated {r['engine_minutes']['ded']:.1f}"
          f"  (limit {r['engine_life_limit_min']:.0f})")
    print(f" Within engine life  : {r['within_engine_life']}")
    print()


# ----------------------------------------------------------------------
# Main Program: NASA DRA 5.0-class parameters, post-TMI abort
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Isp: DRA 5.0 nominal ~900 s (~900-910 s) [1][3]
    isp = 900.0                       # s

    # Thrust per engine: 25 klbf Pewee-class [1][3]
    thrust_per_engine = 25000.0 * LBF_TO_N   # N (~111.2 kN)

    # ------------------------------------------------------------------
    # DRA 5.0 crewed vehicle immediately after TMI
    # ------------------------------------------------------------------
    # Published [3]: IMLEO ~336.5 t (NTR stage ~138.1 t + saddle truss /
    # drop tank ~133.4 t + crew payload ~65 t); ~178.4 t LH2 total;
    # engine burn times ~55 / 14.5 / 9.7 min for TMI / MOC / TEI.
    # Drop tank ~22 t dry, jettisoned after TMI [2][3].
    #
    # ESTIMATE: LH2 used at TMI = 55 min x 3 engines x
    # ~12.6 kg/s ~ 125 t, so ~54 t remains in the core stage (this
    # matches the ~33 t MOC + ~22 t TEI implied by [3]).
    # post-TMI mass ~ 336.5 - 125 - 22 ~ 190 t (includes the ~54 t LH2)
    # MET-dependent: replace these two values to model other abort times.
    m_dra_vehicle = 190000.0          # kg  (ESTIMATE)
    dra_core_lh2  = 54000.0           # kg  (ESTIMATE)

    # ------------------------------------------------------------------
    # Dedicated NTP abort stage (design choices + DRA-scaled values)
    # ------------------------------------------------------------------
    # Dry mass: ESTIMATE. Built from:
    #   engine: 25,000 lbf / T/W 3.43 [1] -> ~3.3 t
    #   tank:   DRA drop tank ~22 t per ~102.4 t LH2 [2] scaled to
    #           25 t LH2 -> ~5.4 t
    #   plus structure, shielding, RCS, contingency
    m_ded_dry  = 9000.0               # kg  (ESTIMATE, ~9 t)
    m_ded_prop = 25000.0              # kg  (design choice)

    results = {}
    for mode in MODES:
        results[mode] = run_abort_burn(
            mode,
            isp,
            m_dra_vehicle,
            dra_core_lh2,
            m_ded_dry,
            m_ded_prop,
            thrust_per_engine_N=thrust_per_engine,
        )

    r0 = results["dedicated"]
    print("=== Post-TMI NTP Abort Options ===")
    print(f"DRA vehicle mass (post-TMI)  : {m_dra_vehicle/1000:8.1f} t")
    print(f"DRA core-stage LH2 remaining : {dra_core_lh2/1000:8.1f} t")
    print(f"Dedicated stage dry / LH2    : {m_ded_dry/1000:8.1f} / "
          f"{m_ded_prop/1000:.1f} t")
    print(f"Isp                          : {isp:8.1f} s")
    print(f"Effective exhaust velocity   : {r0['exhaust_velocity_m_s']:8.1f} m/s")
    print(f"Mass flow per engine         : {r0['mdot_per_engine_kg_s']:8.3f} kg/s")
    print()

    for mode in MODES:
        print_result(results[mode])

    print("=== Summary ===")
    print(f"{'Mode':12s} {'dv [km/s]':>10s} {'burn [min]':>11s} "
          f"{'engine life ok':>15s}")
    for mode in MODES:
        r = results[mode]
        print(f"{mode:12s} {r['delta_v_km_s']:10.3f} "
              f"{r['total_burn_time_s']/60:11.1f} "
              f"{str(r['within_engine_life']):>15s}")