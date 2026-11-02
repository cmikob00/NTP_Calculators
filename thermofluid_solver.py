"""
Simple 1-D thermofluid model of hydrogen flow in an NTR core channel
(Emrich-style equivalent-channel analysis)

Author: for class poster on Nuclear Thermal Propulsion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# ==========  TWEAKABLE INPUT PARAMETERS  ====================
# ============================================================

# Geometry
L          = 0.89          # channel length [m]          (typical NERVA ~0.89 m)
Dh         = 0.0025        # hydraulic diameter [m]      (≈ 2.5 mm)
Ac         = np.pi*(Dh/2)**2   # flow area [m²] (circular for simplicity)
Ph         = np.pi*Dh      # heated perimeter [m]

# Operating conditions
mdot       = 0.0015        # mass flow per channel [kg/s]
Tb_in      = 300.0         # inlet bulk temperature [K]
p_in       = 6.0e6         # inlet pressure [Pa]        (60 bar)
q_prime_avg = 1.2e5        # average linear heat rate [W/m]  (q'' * Ph)

# Axial power shape (cosine-like, peaked in middle)
# q'(x) = q_prime_avg * (1 + a*cos(pi*(x/L - 0.5)))
power_peaking = 1.3        # max/avg power

# Heat-transfer & friction
roughness  = 5e-6          # absolute roughness [m]
# Simple Nu correlation: Nu = 0.023 * Re**0.8 * Pr**0.4 * (Tw/Tb)**(-0.5)
# (property-ratio correction for large temperature difference)

# Integration
n_points   = 201           # number of axial stations for output

# ============================================================
# ==========  HYDROGEN PROPERTY APPROXIMATIONS  ==============
# ============================================================
# Very simple ideal-gas model with temperature-dependent cp
# (good enough for a classroom demo; real NTR codes use tables + dissociation)

R_univ = 8314.46           # J/(kmol·K)
M_H2   = 2.016             # kg/kmol
Rs     = R_univ / M_H2     # specific gas constant J/(kg·K)

def cp_H2(T):
    """Approximate cp [J/(kg·K)] for H2 (valid ~300-3000 K, no dissociation)"""
    # Simple polynomial fit (rough)
    return 14000 + 1.5*(T-300)          # rises slowly

def gamma_H2(T):
    cp = cp_H2(T)
    cv = cp - Rs
    return cp / cv

def mu_H2(T):
    """Dynamic viscosity [Pa·s] – Sutherland-like"""
    return 8.5e-6 * (T/300)**0.7

def k_H2(T):
    """Thermal conductivity [W/(m·K)]"""
    return 0.18 * (T/300)**0.7

def Pr_H2(T):
    return mu_H2(T) * cp_H2(T) / k_H2(T)

# ============================================================
# ==========  FRICTION & HEAT TRANSFER  ======================
# ============================================================

def friction_factor(Re, Dh, eps):
    """Explicit approximation of Colebrook (Haaland)"""
    if Re < 2300:
        return 64 / Re
    inv_sqrt_f = -1.8 * np.log10((eps/Dh/3.7)**1.11 + 6.9/Re)
    return 1 / inv_sqrt_f**2

def heat_transfer_coeff(Re, Pr, k, Dh, Tw_Tb_ratio):
    """Dittus-Boelter with simple property-ratio correction"""
    Nu = 0.023 * Re**0.8 * Pr**0.4 * (Tw_Tb_ratio)**(-0.5)
    return Nu * k / Dh

# ============================================================
# ==========  ODE SYSTEM  ====================================
# ============================================================

def channel_odes(x, y, params):
    """
    State vector y = [Tb, p, u]
    Returns dTb/dx, dp/dx, du/dx
    """
    Tb, p, u = y
    mdot, Ac, Ph, Dh, eps, q_prime_func = params

    # Density from ideal gas
    rho = p / (Rs * Tb)

    # Local heat addition
    q_prime = q_prime_func(x)          # W/m
    q_flux  = q_prime / Ph             # W/m²

    # Transport properties
    mu  = mu_H2(Tb)
    k   = k_H2(Tb)
    Pr  = Pr_H2(Tb)
    Re  = rho * u * Dh / mu

    # Friction
    f = friction_factor(Re, Dh, eps)

    # Heat transfer (need Tw estimate – iterate once)
    # First guess Tw ≈ Tb + q_flux / h0
    h0 = heat_transfer_coeff(Re, Pr, k, Dh, 1.0)
    Tw_guess = Tb + q_flux / max(h0, 1.0)
    Tw_Tb = max(Tw_guess / Tb, 1.01)
    h = heat_transfer_coeff(Re, Pr, k, Dh, Tw_Tb)
    Tw = Tb + q_flux / h

    # Energy: mdot * cp * dTb/dx = q'
    cp = cp_H2(Tb)
    dTb_dx = q_prime / (mdot * cp)

    # Momentum: dp/dx = -f/Dh * rho*u²/2 - rho*u*du/dx
    # From continuity: du/dx = - (u/rho) * drho/dx
    # We will use the combined form that avoids explicit drho

    # Velocity change from continuity + energy (approximate)
    # Better: use the differential form of momentum + continuity
    # dp/dx + rho*u*du/dx + (f/Dh)*(rho*u**2)/2 = 0
    # and d(rho*u)/dx = 0  →  rho*du + u*drho = 0

    # Simple explicit march-friendly form:
    # Use acceleration term from energy (density change mainly from T)
    # For ideal gas at nearly constant p the dominant effect is T
    du_dx = u * (dTb_dx / Tb)          # first-order approximation (constant p)
    # More accurate would solve the algebraic system, but this is fine for demo

    dp_dx = - (f/Dh)*(rho*u**2)/2 - rho*u*du_dx

    return [dTb_dx, dp_dx, du_dx]

# ============================================================
# ==========  POWER SHAPE  ===================================
# ============================================================

def make_q_prime(x, L, q_avg, peaking):
    """Cosine power shape"""
    return q_avg * (1.0 + (peaking-1.0)*np.cos(np.pi*(x/L - 0.5)))

# ============================================================
# ==========  MAIN CALCULATION  ==============================
# ============================================================

def run_channel():
    # Initial velocity
    rho_in = p_in / (Rs * Tb_in)
    u_in   = mdot / (rho_in * Ac)

    # Pack parameters
    q_prime_func = lambda x: make_q_prime(x, L, q_prime_avg, power_peaking)
    params = (mdot, Ac, Ph, Dh, roughness, q_prime_func)

    # Solve ODE
    x_span = (0.0, L)
    y0 = [Tb_in, p_in, u_in]
    sol = solve_ivp(channel_odes, x_span, y0, args=(params,),
                    dense_output=True, rtol=1e-5, atol=1e-7,
                    method='RK45')

    # Evaluate on uniform grid
    x = np.linspace(0, L, n_points)
    Tb, p, u = sol.sol(x)

    # Derived quantities
    rho = p / (Rs * Tb)
    a   = np.sqrt(gamma_H2(Tb) * Rs * Tb)
    M   = u / a
    Tw  = np.zeros_like(Tb)

    # Recompute Tw for plotting
    for i in range(len(x)):
        mu  = mu_H2(Tb[i])
        k   = k_H2(Tb[i])
        Pr  = Pr_H2(Tb[i])
        Re  = rho[i] * u[i] * Dh / mu
        q_prime = q_prime_func(x[i])
        q_flux  = q_prime / Ph
        h0 = heat_transfer_coeff(Re, Pr, k, Dh, 1.0)
        Tw_guess = Tb[i] + q_flux / max(h0, 1.)
        Tw_Tb = max(Tw_guess/Tb[i], 1.01)
        h = heat_transfer_coeff(Re, Pr, k, Dh, Tw_Tb)
        Tw[i] = Tb[i] + q_flux / h

    return x, Tb, Tw, p, u, M, rho

# ============================================================
# ==========  RUN & PLOT  ====================================
# ============================================================

if __name__ == "__main__":
    x, Tb, Tw, p, u, M, rho = run_channel()

    print("=== Channel Results ===")
    print(f"Exit bulk temperature : {Tb[-1]:8.1f} K")
    print(f"Exit wall temperature : {Tw[-1]:8.1f} K")
    print(f"Exit pressure         : {p[-1]/1e6:8.2f} MPa")
    print(f"Pressure drop         : {(p[0]-p[-1])/1e6:8.2f} MPa")
    print(f"Exit Mach number      : {M[-1]:8.3f}")
    print(f"Exit velocity         : {u[-1]:8.1f} m/s")

    if M[-1] > 0.4:
        print("\n*** WARNING: Exit Mach number is getting high – "
              "approaching thermal choking risk ***")
    if np.any(M > 0.9):
        print("*** FLOW IS APPROACHING OR HAS REACHED CHOKING ***")

    # -------------------- Plots --------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("NTR Core Channel – Simplified 1-D Thermofluid Model", fontsize=14)

    # Temperatures
    ax = axes[0, 0]
    ax.plot(x*100, Tb, 'b-', lw=2, label='Bulk $T_b$')
    ax.plot(x*100, Tw, 'r--', lw=2, label='Wall $T_w$')
    ax.set_xlabel("Axial distance [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Temperature Profiles")

    # Pressure
    ax = axes[0, 1]
    ax.plot(x*100, p/1e6, 'g-', lw=2)
    ax.set_xlabel("Axial distance [cm]")
    ax.set_ylabel("Pressure [MPa]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Pressure Drop")

    # Mach number
    ax = axes[1, 0]
    ax.plot(x*100, M, 'm-', lw=2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='Sonic')
    ax.set_xlabel("Axial distance [cm]")
    ax.set_ylabel("Mach number")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Mach Number (watch for choking)")

    # Density & velocity
    ax = axes[1, 1]
    ax.plot(x*100, rho, 'c-', lw=2, label='Density [kg/m³]')
    ax.set_xlabel("Axial distance [cm]")
    ax.set_ylabel("Density [kg/m³]", color='c')
    ax.tick_params(axis='y', labelcolor='c')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x*100, u, 'y-', lw=2, label='Velocity [m/s]')
    ax2.set_ylabel("Velocity [m/s]", color='y')
    ax2.tick_params(axis='y', labelcolor='y')
    ax.set_title("Density & Velocity")

    plt.tight_layout()
    plt.savefig("ntr_channel_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPlot saved as 'ntr_channel_results.png'")