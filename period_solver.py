"""
Stable reactor period T versus reactivity for a NERVA-style HEU core
and a simplified modern HALEU nuclear thermal rocket concept.

Two plots are generated:
  Plot 1 – Period vs. reactivity in pcm
  Plot 2 – Period vs. reactivity in dollars

All numerical values and their literature sources are documented in the
comments below.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ------------------------------------------------------------------
# Delayed-neutron group data
# ------------------------------------------------------------------
# Classic Keepin 6-group parameters for thermal fission of U-235.
# Source: Keepin, G. R., "Physics of Nuclear Kinetics",
#         Addison-Wesley, 1965.
# Also tabulated in many reactor-physics references (e.g., Lamarsh &
# Baratta, "Introduction to Nuclear Engineering").
lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])   # s^-1
beta_i_frac = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])  # relative abundances

# ------------------------------------------------------------------
# Core-specific kinetic parameters
# ------------------------------------------------------------------
# NERVA-style HEU (highly enriched uranium, thermal spectrum, graphite moderated)
#   β_eff ≈ 0.0065  – standard value for thermal U-235
#     Source: Keepin (1965); also used throughout NERVA-era documentation
#   ℓ ≈ 1.0e-4 s   – representative prompt-neutron lifetime for a thermal,
#                     graphite-moderated core of NERVA size
#     Sources: typical values quoted in NERVA reactor-physics summaries
#              and modern NTR point-kinetics studies (order 10^{-4} s)

beta_nerva = 0.0065
ell_nerva  = 1.0e-4          # s

# Simplified HALEU concept (≈19.75 % enrichment) Widdicombe et al. (2024)
#   β_eff ≈ 0.0072  – modest increase relative to pure HEU caused by a
#                     small contribution of fast fissions in U-238
#     Source: illustrative value guided by published HALEU NTR neutronics
#             (e.g., reports noting β_eff ≈ 0.007–0.0077 when U-238 is present)
#   ℓ ≈ 6.0e-5 s   – slightly shorter lifetime reflecting possible differences
#                     in moderation / spectrum relative to classic NERVA
#     Source: representative engineering assumption for a modern HALEU NTR

beta_haleu = 0.0072
ell_haleu  = 8.0e-5          # s   or 6.e-5 s

# ------------------------------------------------------------------
# Inhour equation residual and numerical solver
# ------------------------------------------------------------------
def inhour_residual(T, rho, beta_total, ell, lambda_i, beta_i_frac):
    """
    Residual of the multi-group Inhour equation:
        rho - (ell/T + sum [beta_i / (1 + lambda_i * T)])
    """
    if T <= 0.0:
        return 1.0e6
    beta_i = beta_total * beta_i_frac
    delayed = np.sum(beta_i / (1.0 + lambda_i * T))
    return rho - (ell / T + delayed)

def period_from_rho(rho, beta_total, ell, lambda_i, beta_i_frac):
    """
    Solve the Inhour equation for the stable (asymptotic) period T > 0.
    Returns NaN if no physical positive root is found.
    """
    if rho <= 0.0:
        return np.nan
    try:
        T = brentq(inhour_residual, 1.0e-3, 1.0e5,
                   args=(rho, beta_total, ell, lambda_i, beta_i_frac))
        return T
    except ValueError:
        return np.nan

# ------------------------------------------------------------------
# Generate data
# ------------------------------------------------------------------
# Absolute reactivity range (covers from low positive reactivity past prompt critical)
rho_abs = np.linspace(0.0003, 0.010, 400)          # absolute Δk/k

T_nerva = np.array([period_from_rho(r, beta_nerva, ell_nerva, lambda_i, beta_i_frac)
                    for r in rho_abs])
T_haleu = np.array([period_from_rho(r, beta_haleu, ell_haleu, lambda_i, beta_i_frac)
                    for r in rho_abs])

# Convert to the two unit systems requested by the user
pcm_nerva   = rho_abs * 1.0e5
pcm_haleu   = rho_abs * 1.0e5
dollars_nerva = rho_abs / beta_nerva
dollars_haleu = rho_abs / beta_haleu

# ------------------------------------------------------------------
# Plotting – two side-by-side figures
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ----- Plot 1: Period vs. reactivity in pcm -----
ax1.semilogy(pcm_nerva, T_nerva, 'b-',  lw=2.2,
             label=f'NERVA HEU  (β = {beta_nerva})')
ax1.semilogy(pcm_haleu, T_haleu, 'r--', lw=2.2,
             label=f'HALEU      (β = {beta_haleu})')

ax1.axvline(beta_nerva * 1.0e5, color='b', ls=':', alpha=0.7,
            label='NERVA prompt critical')
ax1.axvline(beta_haleu * 1.0e5, color='r', ls=':', alpha=0.7,
            label='HALEU prompt critical')

ax1.set_xlabel('Reactivity (pcm)')
ax1.set_ylabel('Stable reactor period T (s)')
ax1.set_title('Period vs. Reactivity (pcm)')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper right', fontsize=8)
ax1.set_xlim(0, 900)
ax1.set_ylim(0.1, 1.0e4)

# ----- Plot 2: Period vs. reactivity in dollars -----
ax2.semilogy(dollars_nerva, T_nerva, 'b-',  lw=2.2,
             label=f'NERVA HEU  (β = {beta_nerva})')
ax2.semilogy(dollars_haleu, T_haleu, 'r--', lw=2.2,
             label=f'HALEU      (β = {beta_haleu})')

ax2.axvline(1.0, color='k', ls=':', lw=1.5, alpha=0.8,
            label='Prompt critical ($1)')

ax2.set_xlabel('Reactivity (dollars)')
ax2.set_ylabel('Stable reactor period T (s)')
ax2.set_title('Period vs. Reactivity (dollars)')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(loc='upper right', fontsize=8)
ax2.set_xlim(0.05, 1.4)
ax2.set_ylim(0.1, 1.0e4)

fig.suptitle('Stable Reactor Period – NERVA-style HEU vs. Simplified HALEU NTR',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()