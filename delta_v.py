#!/usr/bin/env python3
"""
Simple NTP bolt-on abort stage delta-v calculator (SI units).

Calculates ideal delta-v, mass flow rate, burn time,
initial/final acceleration, and propulsive (jet) power for
a bolt-on NTP stage.

SOURCES (NASA Mars DRA 5.0 NTR numbers)
---------------------------------------
[1] Borowski, McCurdy, Packard, "7-Launch NTR Space Transportation
    System for NASA's Mars Design Reference Architecture (DRA) 5.0."
    NTRS 20120003776 (also AIAA 2009-5308).
    Engine performance, crewed vehicle ("Copernicus") masses.
[2] Borowski et al., "Nuclear Thermal Rocket/Vehicle Characteristics and
    Sensitivity Trades for NASA's Mars DRA 5.0 Study."
    NTRS 20120012928.  Engine and stage characteristics, LH2 flow rate.
[3] Borowski et al., "Conventional and Bimodal Nuclear Thermal Rocket
    (NTR) Propulsion for Mars ..."  NTRS 20140017461.
    Reference 25 klbf Pewee-class engine used in DRA 5.0.
[4] Primary reference: NASA-SP-2009-566, "Human
    Exploration of Mars Design Reference Architecture 5.0," and
    Addendum (NASA/SP-2009-566-ADD).
"""

import math


# Standard gravity
G0 = 9.80665  # m/s^2

# Pounds-force to newtons
LBF_TO_N = 4.4482216


def delta_v(isp_s: float,
            m_spacecraft: float,
            m_stage_dry: float,
            m_propellant: float,
            thrust_N: float,
            residual_fraction: float = 0.02) -> dict:
    
    """
    Compute ideal delta-v and basic performance parameters
    for a bolt-on NTP abort stage.

    Parameters
    ----------
    isp_s : float
        Specific impulse [s]

    m_spacecraft : float
        Mass of the main DRA-class vehicle at abort initiation
        (excluding the abort stage) [kg]

    m_stage_dry : float
        Dry mass of the abort NTP stage, including engine,
        structure, and empty tanks [kg]

    m_propellant : float
        Loaded propellant of the abort stage [kg]

    thrust_N : float
        Thrust of the NTP system [N]

    residual_fraction : float
        Fraction of loaded propellant left unused as residuals
        (default 0.02 = 2 %). NOTE: DRA 5.0 also carries cooldown
        ("post-burn") propellant and performance reserves [1][2];
        check ground-rules table for the actual percentages and
        fold them in here if needed.

    Returns
    -------
    dict
        Calculated performance results
    """

    # Calculate residual and usable propellant
    m_residuals   = m_propellant * residual_fraction
    m_prop_usable = m_propellant - m_residuals

    # Initial and final mass of the total spacecraft
    m0 = m_spacecraft + m_stage_dry + m_propellant
    mf = m_spacecraft + m_stage_dry + m_residuals

    if mf >= m0 or m_prop_usable <= 0:
        raise ValueError(
            "Invalid masses: final mass must be less than initial mass"
        )

    # Effective exhaust velocity
    exhaust_velocity_m_s = G0 * isp_s

    # Ideal rocket equation
    # (impulsive; ignores gravity/finite-burn losses, which matter for a
    #  ~0.05 g single-engine stage)
    dv = exhaust_velocity_m_s * math.log(m0 / mf)

    # Mass flow rate
    m_dot = thrust_N / exhaust_velocity_m_s

    # Burn time
    burn_time_s = m_prop_usable / m_dot

    # Initial and final acceleration
    acceleration_initial_m_s2 = thrust_N / m0
    acceleration_final_m_s2   = thrust_N / mf

    acceleration_initial_g = acceleration_initial_m_s2 / G0
    acceleration_final_g   = acceleration_final_m_s2 / G0

    # Propulsive (jet) power, not reactor thermal power
    # P = 0.5 * thrust * effective exhaust velocity
    power_W  = 0.5 * thrust_N * exhaust_velocity_m_s
    power_MW = power_W / 1.0e6

    return {
        "delta_v_m_s": dv,
        "delta_v_km_s": dv / 1000.0,
        "m0_kg": m0,
        "mf_kg": mf,
        "mass_ratio": m0 / mf,
        "usable_propellant_kg": m_prop_usable,
        "residuals_kg": m_residuals,
        "thrust_N": thrust_N,
        "m_dot_kg_s": m_dot,
        "burn_time_s": burn_time_s,
        "exhaust_velocity_m_s": exhaust_velocity_m_s,
        "power_W": power_W,
        "power_MW": power_MW,
        "acceleration_initial_m_s2": acceleration_initial_m_s2,
        "acceleration_final_m_s2": acceleration_final_m_s2,
        "acceleration_initial_g": acceleration_initial_g,
        "acceleration_final_g": acceleration_final_g,
    }


# ----------------------------------------------------------------------
# Main Program: using NASA DRA 5.0-class numbers
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Main vehicle (DRA 5.0 crewed "Copernicus" MTV) at abort
    # ------------------------------------------------------------------
    # Published [1]: IMLEO ~336.5 t = NTR stage ~138.1 t
    #                + saddle truss / LH2 drop tank ~133.4 t
    #                + crew payload ~65 t; ~178.4 t LH2 over 4 primary
    #                burns; drop tank (~22 t dry, ~102.4 t LH2) jettisoned
    #                after TMI.
    # NOTE: crewed payload mass differs between DRA 5.0 papers (~65 t in
    # [1], ~111 t in earlier cuts).
    #
    # ESTIMATE: post-TMI mass of roughly 195-200 t,
    # back-calculated from IMLEO, an assumed TMI delta-v of ~3.8-4.0 km/s
    # at Isp ~900 s, and drop-tank jettison. Includes the remaining
    # core-stage LH2 for MOC/TEI.
    # MET-dependent: mass steps down at drop-tank jettison (TMI) and
    # again at Mars orbit capture (MOC).
    m_sc = 200000.0          # kg   (ESTIMATE, ~200 t)

    # ------------------------------------------------------------------
    # Dedicated NTP abort stage (design choices + DRA-scaled values)
    # ------------------------------------------------------------------
    # Isp: DRA 5.0 nominal ~900 s (~900-910 s; ~906 s in crewed burn
    # analysis) [1][2][3].
    isp = 900.0              # s

    # Thrust: one 25 klbf Pewee-class engine [1][2][3]
    # (DRA core stage uses three, i.e. 75 klbf; abort stage engine count
    #  is a design choice for this study.)
    thrust = 25000.0 * LBF_TO_N   # N (~111.2 kN)

    # Dry mass: ESTIMATE. Built from:
    #   engine: 25,000 lbf / T/W 3.43 [1][2] -> ~3.3 t
    #   tank:   DRA drop tank ~22 t per ~102.4 t LH2 [1] scaled to
    #           25 t LH2 -> ~5.4 t
    #   plus structure, shielding, RCS, contingency
    m_stage_dry = 9000.0     # kg   (ESTIMATE, ~9 t)

    # Propellant load (LH2): your design choice
    m_prop      = 25000.0    # kg

    results = delta_v(
        isp,
        m_sc,
        m_stage_dry,
        m_prop,
        thrust
    )

    print("=== Bolt-on NTP Abort Stage Performance ===")
    print(f"Spacecraft mass at abort     : {m_sc/1000:8.1f} t")
    print(f"Abort stage dry mass         : {m_stage_dry/1000:8.1f} t")
    print(f"Abort propellant load        : {m_prop/1000:8.1f} t")
    print(f"Isp                          : {isp:8.1f} s")
    print(f"Thrust                       : {thrust/1000:8.1f} kN")
    print(f"Effective exhaust velocity   : {results['exhaust_velocity_m_s']:8.1f} m/s")
    print(f"Initial total mass           : {results['m0_kg']/1000:8.1f} t")
    print(f"Final total mass             : {results['mf_kg']/1000:8.1f} t")
    print(f"Mass ratio                   : {results['mass_ratio']:8.3f}")
    print(f"Ideal delta-v                : {results['delta_v_km_s']:8.3f} km/s")
    print(f"                             : {results['delta_v_m_s']:8.0f} m/s")
    print(f"Mass flow rate               : {results['m_dot_kg_s']:8.3f} kg/s")
    print(f"Burn time                    : {results['burn_time_s']/60:8.2f} min")
    print(f"Initial acceleration         : {results['acceleration_initial_m_s2']:8.3f} m/s²")
    print(f"                             : {results['acceleration_initial_g']:8.3f} g")
    print(f"Final acceleration           : {results['acceleration_final_m_s2']:8.3f} m/s²")
    print(f"                             : {results['acceleration_final_g']:8.3f} g")
    print(f"Jet power                    : {results['power_MW']:8.2f} MW")
    print(f"                             : {results['power_W']:8.3e} W")