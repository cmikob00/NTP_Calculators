#!/usr/bin/env python3
"""
Simple NTP bolt-on abort stage delta-v calculator (SI units).

Calculates ideal delta-v, mass flow rate, burn time, and
initial/final acceleration for a bolt-on NTP stage.
"""

import math


# Standard gravity
G0 = 9.80665  # m/s²


def delta_v(isp_s: float,
            m_spacecraft: float,
            m_stage_dry: float,
            m_propellant: float,
            thrust_N: float,
            residual_fraction: float = 0.0) -> dict:
    """
    Compute ideal delta-v for a bolt-on NTP abort stage.

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
        Usable propellant load of the abort stage [kg]

    thrust_N : float
        Thrust of the NTP system [N]

    residual_fraction : float
        Fraction of loaded propellant left as residuals
        (default 0 %)

    Returns
    -------
    dict
        Calculated performance results
    """

    # Calculate residual and usable propellant
    m_residuals = m_propellant * residual_fraction
    m_prop_usable = m_propellant - m_residuals

    # Initial and final mass of the combined stack
    m0 = m_spacecraft + m_stage_dry + m_propellant
    mf = m_spacecraft + m_stage_dry + m_residuals

    if mf >= m0 or m_prop_usable <= 0:
        raise ValueError(
            "Invalid masses: final mass must be less than initial mass"
        )

    # Ideal rocket equation
    dv = isp_s * G0 * math.log(m0 / mf)

    # Mass flow rate
    m_dot = thrust_N / (isp_s * G0)

    # Burn time
    burn_time_s = m_prop_usable / m_dot

    # Initial and final acceleration
    acceleration_initial_m_s2 = thrust_N / m0
    acceleration_final_m_s2 = thrust_N / mf

    acceleration_initial_g = acceleration_initial_m_s2 / G0
    acceleration_final_g = acceleration_final_m_s2 / G0

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
        "acceleration_initial_m_s2": acceleration_initial_m_s2,
        "acceleration_final_m_s2": acceleration_final_m_s2,
        "acceleration_initial_g": acceleration_initial_g,
        "acceleration_final_g": acceleration_final_g,
    }


# ----------------------------------------------------------------------
# Example usage with representative DRA 5.0-class numbers
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Representative post-TMI mass of the remaining crewed stack
    # (habitat + residual main propulsion + systems). Sweep this value.
    m_sc = 180000.0          # kg   (~180 t)

    # Example bolt-on NTP abort stage
    isp = 900.0              # s
    m_stage_dry = 4000.0     # kg   (engine + tanks + structure, dry)
    m_prop = 25000.0         # kg   (LH2 load)
    thrust = 111200.0        # N    (~25 klbf)

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
    print(f"Initial stack mass           : {results['m0_kg']/1000:8.1f} t")
    print(f"Final stack mass             : {results['mf_kg']/1000:8.1f} t")
    print(f"Mass ratio                   : {results['mass_ratio']:8.3f}")
    print(f"Ideal delta-v                : {results['delta_v_km_s']:8.3f} km/s")
    print(f"                             : {results['delta_v_m_s']:8.0f} m/s")
    print(f"Mass flow rate               : {results['m_dot_kg_s']:8.3f} kg/s")
    print(f"Burn time                    : {results['burn_time_s']/60:8.2f} min")
    print(f"Initial acceleration         : {results['acceleration_initial_m_s2']:8.3f} m/s²")
    print(f"                             : {results['acceleration_initial_g']:8.3f} g")
    print(f"Final acceleration           : {results['acceleration_final_m_s2']:8.3f} m/s²")
    print(f"                             : {results['acceleration_final_g']:8.3f} g")