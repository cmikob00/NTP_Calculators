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
    AIAA 2009-5308.  NOT consulted directly; cited as ref [4] inside [2].
    Check it (and [P]) before quoting any number as coming from them.
[2] Borowski, McCurdy, Packard, "Nuclear Thermal Propulsion (NTP): A
    Proven Growth Technology for Human NEO / Mars Exploration Missions,"
    NTRS 20120003776.  Primary source for the numbers used here:
      - 3 x 25 klbf Pewee-class engines; Isp ~906 s (Tex ~2790 K,
        composite fuel; range 906-941 s); LH2 flow ~12.5 kg/s/engine;
        engine T/W ~3.50 (Table 2 / Sec. 2)
      - propellant margins (Table 2): cooldown 3 % of usable LH2,
        tank trapped residuals 2 % of tank capacity, 1 % performance
        reserve on delta-v
      - Copernicus crewed MTV: IMLEO ~336.5 t = NTR stage ~138.1 t
        (tank capacity ~87.2 t LH2, incl. ~7.3 t external radiation
        shield) + saddle truss / drop tank ~133.4 t (truss ~9 t, tank
        ~22 t dry, ~102.4 t LH2) + crew payload ~65 t
      - 4 primary burns use ~178.4 t LH2; burn times ~55 min TMI
        (2-perigee) / ~14.5 min MOC / ~9.7 min TEI (~79.2 min total)
      - drained drop tank is jettisoned after TMI; the saddle truss stays
      - ~2 hr accumulated burn demonstrated on NRX-XE (27-28 start/stop
        cycles; the paper gives both counts); composite fuel projected
        ~2-6 hr at full power
      - contingencies: 30 % on NTR system / composite structures,
        15 % on tanks and heritage systems; ZBO cryocooler for the core
        stage ~930 kg and ~8.9 kWe
[3] Borowski, "Nuclear Thermal Rocket (NTR) Propulsion: A Proven
    Game-Changing Technology for Future Human Exploration Missions,"
    NTRS 20120009207.  Only the abstract was checked; no numbers in this
    script are attributed to it.
[4] Borowski, "Nuclear Thermal Rocket/Vehicle Characteristics and
    Sensitivity Trades for NASA's Mars DRA 5.0 Study," NTRS 20120012928.
    Only the abstract was checked (DRA 5.0 crewed MTV, ~6-7 month 1-way
    transits); no numbers here are attributed to it.
[P] Primary reference: NASA-SP-2009-566, "Human Exploration of Mars
    Design Reference Architecture 5.0," and Addendum
    (NASA/SP-2009-566-ADD).  NOT consulted; verify against it.
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
                   ded_reserve_fraction: float = 0.05,
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
        trapped residuals). DRA 5.0 carries 3 % cooldown (of usable
        LH2) plus 2 % trapped residuals (of tank capacity) [2]; the
        sum, 5 %, is used here. The separate 1 % performance reserve on
        delta-v is treated as usable in an abort, so it is not held back.

    ded_reserve_fraction : float
        Same for the dedicated stage (default 5 %, same basis as the
        DRA core stage [2]).

    tmi_burn_min_per_dra_engine : float
        Engine time already used by each DRA engine at abort start
        [min]; ~55 min for the 2-perigee TMI burns [2].

    engine_life_limit_min : float
        Accumulated burn-time limit per engine [min]. ~2 hr is the
        DEMONSTRATED accumulated time on NRX-XE over 27-28 start/stop
        cycles [2], not a certified life limit (composite fuel is
        projected at ~2-6 hr at full power [2]).

    Returns
    -------
    dict with total delta-v, per-phase breakdown, and engine-life check
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    ve = G0 * isp_s                       # effective exhaust velocity
    print(f'Effective exhaust velocity: {ve:.3f} m/s')
    mdot_eng = thrust_per_engine_N / ve   # kg/s per engine
    print(f'mdot per engine:            {mdot_eng:.3f} kg/s')
    print(f'\n')

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
        print(f'Going down DRA loop...')
        m = m_dra_vehicle                       # no dedicated stage
        print(f'DRA mass: {m:.3f} kg')
        groups["ded"]["prop"] = 0.0
        ded_inert = 0.0
    else:
        print(f'Going down loop other than DRA loop...')
        m = m_dra_vehicle + m_ded_dry + m_ded_prop
        print(f'Mass:                 {m:.3f} kg')
        ded_inert = m_ded_dry + m_ded_prop * ded_reserve_fraction
        print(f'Dedicated inert mass: {ded_inert:.3f} kg')
    if mode == "dedicated":
        print(f'Going down dedicated loop...')
        groups["dra"]["prop"] = 0.0             # DRA engines stay idle
    
    print(f'\n')

    print(f'Setting initial parameters for {mode}...')
    m0_total = m
    print(f'Initial mass:                    {m0_total:.3f} kg')
    engine_min = {"dra": tmi_burn_min_per_dra_engine, "ded": 0.0}
    print(f'Setting initial engine run time: {engine_min} min')
    ded_attached = mode != "dra"
    phases = []
    total_dv = 0.0
    print(f'Setting initial dv to:           {total_dv:.3f} m/s')
    total_time = 0.0
    print(f'Setting initial time to:         {total_time:.3f} s')
    print(f'\n')

    for step in _STEPS[mode]:
        if step == ["JETTISON"]:
            print(f'Jettison performed because mode is {mode}!\n')
            if ded_attached:
                m -= ded_inert                  # drop the empty stage
                ded_attached = False
            continue

        active = [g for g in step if groups[g]["prop"] > 1e-9]
        print(f'Active stages: {active}...\n')
        if not active:
            continue

        thrust = sum(groups[g]["thrust"] for g in active)
        print(f'Performing delta m calculations for {mode}...')
        print(f'Thrust:      {thrust:.3f} N')
        mdot = sum(groups[g]["mdot"] for g in active)
        print(f'Mdot:        {mdot:.3f} kg')
        dt = min(groups[g]["prop"] / groups[g]["mdot"] for g in active)
        print(f'Delta time:  {dt:.3f} s')
        print(f'Delta time:  {dt/60:.3f} min')
        mburned = mdot * dt
        print(f'Mass burned: {mburned:.3f} kg')
        print(f'\n')

        m_start = m
        print(f'Performing delta v calculations for {mode}...')
        print(f'Starting mass: {m_start:.3f} kg')
        m_end = m - mburned
        print(f'Ending mass:   {m_end:.3f} kg')
        dv = ve * math.log(m_start / m_end)
        print(f'Delta v:       {dv:.3f} m/s')
        print(f'\n')

        for g in active:
            groups[g]["prop"] -= groups[g]["mdot"] * dt
            engine_min[g] += dt / 60.0

        phases.append({
            "engines": "+".join(active),
            "n_engines": sum(groups[g]["n"] for g in active),
            "thrust_N": thrust,
            "burn_time_s": dt,
            "propellant_burned_kg": mburned,
            "m_start_kg": m_start,
            "m_end_kg": m_end,
            "delta_v_m_s": dv,
            "accel_start_g": thrust / m_start / G0,
            "accel_end_g": thrust / m_end / G0,
            "jet_power_MW": 0.5 * thrust * ve / 1.0e6,
        })

        m = m_end
        print(f'Final values:')
        print(f'Ending mass:     {m:.3f} kg')
        total_dv += dv
        print(f'Total delta v:   {total_dv:.3f} m/s')
        total_time += dt
        print(f'Total burn time: {total_time:.3f} s')
        print(f'Total burn time: {total_time/60:.3f} min')
        print(f'\n')

    used = {"dra": mode in ("dra", "sequential", "all_four"),
            "ded": mode in ("dedicated", "sequential", "all_four")}
    life_ok = all(engine_min[g] <= engine_life_limit_min
                  for g in ("dra", "ded") if used[g])
    
    print(f'Calculations complete for mode {mode}!\n')
    print(f'========================\n')

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

    # Isp: DRA 5.0 baseline is ~906 s (composite fuel, Tex ~2790 K) [2];
    # 900 s is used here as a rounded, slightly conservative value
    # (~0.7 % lower exhaust velocity). Flow is then ~12.6 kg/s/engine
    # vs. the published ~12.5 kg/s at 906 s.
    isp = 900.0     # s

    # Thrust per engine: 25 klbf Pewee-class [2]
    thrust_per_engine = 25000.0 * LBF_TO_N   # N (~111.2 kN)

    # ------------------------------------------------------------------
    # DRA 5.0 crewed vehicle immediately after TMI
    # ------------------------------------------------------------------
    # Published [2]: IMLEO ~336.5 t (NTR stage ~138.1 t + saddle truss /
    # drop tank ~133.4 t + crew payload ~65 t); ~178.4 t LH2 used over
    # 4 burns; burn times ~55 / 14.5 / 9.7 min for TMI / MOC / TEI.
    # Drop tank ~22 t dry (jettisoned after TMI); saddle truss ~9 t stays.
    #
    # Dry mass after TMI (published components):
    #   NTR stage 138.1 - 87.2 (LH2 capacity) = 50.9 t
    #   + saddle truss 9 t + payload 65 t  ~ 125 t
    # LH2 in the core stage after TMI (ESTIMATE):
    #   loaded: 87.2 (stage) + 102.4 (drop tank) = 189.6 t
    #   TMI burn: 55 min x 3 engines x 12.5 kg/s = 123.75 t, plus ~3 %
    #   cooldown -> ~127.5 t
    #   remaining ~62 t. (The ~54 t that MOC + TEI burn time implies,
    #   24.2 min x 2.25 t/min, is burn propellant only; the tank also
    #   holds margins/residuals.)
    # post-TMI mass ~ 125 + 62 = 187 t. Cross-check:
    #   336.5 - 127.5 - 22 = 187 t.
    # MET-dependent only weakly (RCS propellant use, ~9 t total [2]);
    # replace these two values to model other abort times.
    m_dra_vehicle = 187000.0          # kg  (ESTIMATE)
    dra_core_lh2  = 62000.0           # kg  (ESTIMATE)

    # ------------------------------------------------------------------
    # Dedicated NTP abort stage (design choices + DRA-scaled values)
    # ------------------------------------------------------------------
    # Dry mass: ESTIMATE, bottom-up (NOT from a published source; run a
    # sensitivity sweep, ~9-15 t changes delta-v by < ~140 m/s):
    #   engine:    25,000 lbf / T/W 3.50 [2]                    ~3.2 t
    #   tank:      DRA drop tank ~22 t per ~102.4 t LH2 [2]
    #              scaled linearly to 25 t LH2 (likely optimistic
    #              for a small tank)                             ~5.4 t
    #   shielding: DRA ~7.3 t external shield for 3 engines [2],
    #              ~1/3 for one engine                           ~2.4 t
    #   ZBO cryocooler / thermal protection (core stage: 930 kg [2])
    #              scaled down; stage idles for months           ~0.5 t
    #   structure, interstage, avionics, RCS                     ~1.5 t
    #   total ~13 t. 30 % / 15 % DRA contingencies [2] are NOT yet
    #   applied; add them for a conservative case.
    m_ded_dry  = 3200.0 + 5400.0 + 2400.0 + 500.0 + 1500.0   # kg (~13 t)
    m_ded_prop = 25000.0              # kg  (design choice)

    results = {}
    for mode in MODES:
        print(f'Running calculations for {mode}...')
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