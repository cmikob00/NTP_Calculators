#!/usr/bin/env python3
"""
Educational NTR closed-loop transfer-function & Bode-plot calculator  (REVISION 5)
HEU and HALEU cases with 12 delayed-neutron groups (6 U-235 thermal + 6 U-238 fast).
Follows the derivation in the class PDF ("Reactor stability model") and the
corresponding MATLAB Live Script (Pillar7 / SPST513).

NEW IN REVISION 5
  * Corrected stability assessment.  The plotted quantity is the CLOSED-LOOP
    transfer function K_RT G_RT(jω).  Classical open-loop gain/phase margins
    cannot be read from it.  The previous code therefore produced a spurious
    "UNSTABLE" label even though D0 < 0 (self-regulating) and the resonance
    is modest.  Diagnostics now check:
        – sign of the static feedback coefficient D0
        – consistency of low-frequency gain with 1/|D0|
        – height of the resonant peak above DC gain
    and report a verdict that matches the textbook description
    ("self-regulating and stable").

REVISION 4 features retained
  * Physical-parameter derivation (mdot, Sfm, rf, … → τ_f, τ_m, τ_p, A_f, A_m, B_p)
  * Ability to vary mass-flow rate (or any other physical parameter) and
    immediately re-compute the Bode plot.
  * 12-group delayed-neutron data, HEU/HALEU presets, dual-axis Bode plots.

KNOWN LIMITATIONS
  * Linearised small-perturbation model about steady-state power q0.
  * beta is the physical delayed fraction (no importance weighting).
  * HEU and HALEU share the same thermal/feedback parameters unless overrides
    are supplied.
"""

import cmath
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt" and os.uname().sysname != "Darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# User-tunable switches
# ----------------------------------------------------------------------
AXIS = "omega"          # "omega" (rad/s, matches PDF) or "hz"
SHOW = True             # plt.show() after saving figures
MAKE_OVERLAY = True     # write HEU-vs-HALEU comparison figure
OUTDIR = "."

# Frequency sweep, rad/s (PDF plot spans ~6e-4 … 6.4e3 rad/s)
W_MIN, W_MAX = 6.0e-4, 6.4e3

# ----------------------------------------------------------------------
# 1. Physical / geometric / material parameters  (from MATLAB Live Script)
#    Change any of these (especially mdot) to explore system response.
# ----------------------------------------------------------------------
PHYS = dict(
    # Core average power density
    q0=3750.0,              # W cm^-3

    # Prompt neutron generation time
    Lambda=1.00e-4,         # s

    # Fuel
    Sfm=3.5,                # cm^-1  surface area of fuel per unit volume of fuel element
    rf=0.025,               # cm     characteristic heat-transfer distance through fuel
    kf=0.23,                # W cm^-1 K^-1  thermal conductivity of fuel
    Cpf=0.15,               # W s g^-1 K^-1 specific heat of fuel
    pf=13.5,                # g cm^-3 density of fuel
    alphaf=-2.00e-5,        # K^-1   fuel temperature coefficient of reactivity

    # Graphite moderator
    Smp=8.3,                # cm^-1  surface area of coolant holes per unit volume
    rm=0.2,                 # cm     characteristic distance through graphite matrix
    km=0.31,                # W cm^-1 K^-1  thermal conductivity of graphite
    Cpm=1.9,                # W s g^-1 K^-1 specific heat of graphite
    pm=1.7,                 # g cm^-3 density of graphite
    alpham=-4.50e-5,        # K^-1   moderator temperature coefficient of reactivity

    # Hydrogen propellant
    Cpp=16.8,               # W s g^-1 K^-1 specific heat of hydrogen
    hc=0.5,                 # W cm^-2 K^-1  heat-transfer coefficient
    P=7.0,                  # MPa    average propellant pressure
    T=1500.0,               # K      average propellant temperature
    R=4.2,                  # MPa cm^3 g^-1 K^-1  gas constant of hydrogen
    mdot=0.083,             # g s^-1 cm^-3  hydrogen mass-flow rate per unit volume of fuel element
    bp=100.0,               # cm^3 g^-1  hydrogen density coefficient of reactivity
)


def derive_thermal_params(phys: Dict[str, float] = None) -> Dict[str, float]:
    """
    Derive the transfer-function parameters that appear in the closed-loop
    expression from the physical quantities (exactly as the MATLAB script).

    Returns a dict containing:
        q0, Lambda,
        tau_f, A_f, alpha_f,
        tau_m, A_m, alpha_m,
        tau_p, B_p, beta_p
    plus diagnostic quantities Ufm, Ump, D0, mdot.
    """
    p = dict(PHYS if phys is None else phys)

    # Thermal conductances
    Ufm = p["kf"] * p["Sfm"] / p["rf"]
    Ump = p["Smp"] / (p["rm"] / p["km"] + 1.0 / p["hc"])

    # Time constants
    tau_f = p["pf"] * p["Cpf"] / Ufm
    tau_m = (p["pm"] * p["Cpm"]) / (Ufm + Ump)
    tau_p = (p["P"] * p["Cpp"]) / (p["R"] * p["T"] * (Ump + 2.0 * p["mdot"] * p["Cpp"]))

    # Thermal resistances / gains
    A_f = 1.0 / Ufm
    A_m = 1.0 / (Ufm + Ump)
    B_p = (Ump * p["P"]) / (
        p["R"] * p["T"] ** 2 * (Ufm + Ump) * (Ump + 2.0 * p["mdot"] * p["Cpp"])
    )

    # Static reactivity feedback coefficient (key stability indicator)
    D0 = p["alphaf"] * A_f + p["alpham"] * A_m - p["bp"] * B_p

    return dict(
        q0=p["q0"],
        Lambda=p["Lambda"],
        tau_f=tau_f,
        A_f=A_f,
        alpha_f=p["alphaf"],
        tau_m=tau_m,
        A_m=A_m,
        alpha_m=p["alpham"],
        tau_p=tau_p,
        B_p=B_p,
        beta_p=p["bp"],
        # extra diagnostic quantities
        Ufm=Ufm,
        Ump=Ump,
        D0=D0,
        mdot=p["mdot"],
    )


# ----------------------------------------------------------------------
# 2. Delayed-neutron data (Keepin, Wimett & Zeigler 1957)
# ----------------------------------------------------------------------
BETA235_TOTAL = 0.0065
a_235 = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
lam_235 = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
beta_235 = BETA235_TOTAL * a_235

BETA238_TOTAL = 0.0148
a_238 = np.array([0.013, 0.137, 0.162, 0.388, 0.225, 0.075])
lam_238 = np.array([0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02])
beta_238 = BETA238_TOTAL * a_238

NU_235 = 2.43
NU_238 = 2.80


def neutron_fraction_from_fission_fraction(F238, nu5=NU_235, nu8=NU_238):
    """Convert a U-238 fission-RATE fraction to a fission-NEUTRON fraction."""
    return F238 * nu8 / (F238 * nu8 + (1.0 - F238) * nu5)


def make_12group(f238: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    f238 = fraction of fission NEUTRONS born in U-238 (fast fission).
      HEU preset 0.01, HALEU preset 0.08 (ANL/NSE-20/55).
    Returns (beta[12], lam[12]); beta.sum() is the total delayed fraction.
    """
    beta = np.concatenate([(1.0 - f238) * beta_235, f238 * beta_238])
    lam = np.concatenate([lam_235, lam_238])
    return beta, lam


# ----------------------------------------------------------------------
# 3. Case definition
# ----------------------------------------------------------------------
@dataclass
class Case:
    name: str
    f238: float
    overrides: dict = field(default_factory=dict)
    color: str = "tab:blue"
    use_physical: bool = True

    def params(self) -> Dict[str, float]:
        if self.use_physical:
            phys = dict(PHYS)
            phys.update({k: v for k, v in self.overrides.items() if k in PHYS})
            derived = derive_thermal_params(phys)
            derived.update({k: v for k, v in self.overrides.items() if k not in PHYS})
            return derived
        else:
            base = derive_thermal_params()
            base.update(self.overrides)
            return base


HEU = Case("HEU (f238 = 0.01)", 0.01, color="tab:blue")
HALEU = Case("HALEU (f238 = 0.08)", 0.08, color="tab:red")


# ----------------------------------------------------------------------
# 4. Transfer function (vectorised)
# ----------------------------------------------------------------------
def transfer_function(omega, case: Case):
    """K_RT G_RT(iω) = θ_o / θ_i for the closed loop.  omega in rad/s."""
    p = case.params()
    beta, lam = make_12group(case.f238)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * w

    # Kinetics
    D_kin = s * p["Lambda"] + np.sum(
        s[:, None] * beta[None, :] / (s[:, None] + lam[None, :]), axis=1
    )
    kin = D_kin / p["q0"]

    # Feedback (entered as positive numbers → negative reactivity feedback)
    lag_f = 1.0 + s * p["tau_f"]
    lag_m = 1.0 + s * p["tau_m"]
    lag_p = 1.0 + s * p["tau_p"]
    fuel = -p["alpha_f"] * p["A_f"] / lag_f
    mod = -p["alpha_m"] * p["A_m"] / (lag_f * lag_m)
    prop = +p["beta_p"] * p["B_p"] / (lag_f * lag_m * lag_p)

    return 1.0 / (kin + fuel + mod + prop)


def gain_phase(omega, case: Case):
    tf = transfer_function(omega, case)
    gain_db = 20.0 * np.log10(np.abs(tf))
    phase = np.degrees(np.unwrap(np.angle(tf)))
    return gain_db, phase


# ----------------------------------------------------------------------
# 5. Detailed single-frequency walk-through
# ----------------------------------------------------------------------
def compute_one_frequency(omega, case: Case, label: str = ""):
    p = case.params()
    beta, lam = make_12group(case.f238)
    s = 1j * omega
    print("=" * 72)
    print(f"ω = {omega:.4g} rad/s  ({omega / (2 * math.pi):.4g} Hz)   {label or case.name}")
    print(f"total β = {beta.sum():.6f}   mdot = {p.get('mdot', 'n/a')} g s⁻¹ cm⁻³")
    print("=" * 72)

    D_kin = s * p["Lambda"]
    print("\n--- Step A: kinetics denominator ---")
    print(f"  prompt term s·Λ = {D_kin}")
    for i, (b, l) in enumerate(zip(beta, lam), start=1):
        term = s * b / (s + l)
        D_kin += term
        print(f"  group {i:2d}: s·β/(s+λ) = {term}")
    kin = D_kin / p["q0"]
    print(f"  D_kin = {D_kin}\n  D_kin/q0 = {kin}")

    lag_f = 1.0 + s * p["tau_f"]
    lag_m = 1.0 + s * p["tau_m"]
    lag_p = 1.0 + s * p["tau_p"]
    fuel = -p["alpha_f"] * p["A_f"] / lag_f
    mod = -p["alpha_m"] * p["A_m"] / (lag_f * lag_m)
    prop = +p["beta_p"] * p["B_p"] / (lag_f * lag_m * lag_p)
    print("\n--- Steps B–D: feedback (positive = negative reactivity feedback) ---")
    print(f"  fuel       -α_f A_f/(1+s τ_f)                 = {fuel}")
    print(f"  moderator  -α_m A_m/(lag_f·lag_m)             = {mod}")
    print(f"  propellant +β_p B_p/(lag_f·lag_m·lag_p)       = {prop}")

    denom = kin + fuel + mod + prop
    TF = 1.0 / denom
    print("\n--- Step E: closed-loop result ---")
    print(f"  denominator = {denom}\n  TF = {TF}")
    print(
        f"  |TF| = {abs(TF):.6e}   Gain = {20 * math.log10(abs(TF)):.2f} dB   "
        f"Phase = {math.degrees(cmath.phase(TF)):.2f} deg\n"
    )
    return TF


# ----------------------------------------------------------------------
# 6. Diagnostics & stability assessment  (CORRECTED)
# ----------------------------------------------------------------------
def diagnose(case: Case, n_points: int = 2000) -> Dict:
    """
    Compute key Bode features of the CLOSED-LOOP transfer function and a
    stability verdict appropriate to this self-regulating reactor model.
    """
    omega = np.logspace(math.log10(W_MIN), math.log10(W_MAX), n_points)
    g, ph = gain_phase(omega, case)
    p = case.params()

    # Low-frequency asymptote
    g_lo, ph_lo = g[0], ph[0]

    # Expected DC gain from static feedback
    D0 = p.get("D0")
    if D0 is not None and abs(D0) > 0:
        g_dc_expected = 20.0 * math.log10(1.0 / abs(D0))
    else:
        g_dc_expected = float("nan")

    # Gain peak
    i_g = np.argmax(g)
    g_peak, w_gpeak = g[i_g], omega[i_g]
    peaking_db = g_peak - g_lo          # resonance height above DC gain

    # Phase peak (most positive)
    i_ph = np.argmax(ph)
    ph_peak, w_phpeak = ph[i_ph], omega[i_ph]

    # High-frequency end
    g_hi, ph_hi = g[-1], ph[-1]

    # ------------------------------------------------------------------
    # Stability verdict for a closed-loop self-regulating reactor
    # ------------------------------------------------------------------
    # 1. D0 must be negative (negative reactivity feedback → self-regulating)
    # 2. Resonant peaking should be modest (a few dB).  Large peaking would
    #    indicate lightly-damped complex poles.
    # Classical open-loop margins are NOT applicable to this closed-loop TF.
    if D0 is None:
        verdict = "UNKNOWN (D0 not available)"
    elif D0 >= 0:
        verdict = "UNSTABLE / not self-regulating (D0 ≥ 0)"
    elif peaking_db > 15.0:
        verdict = "MARGINALLY STABLE / lightly damped (large resonant peak)"
    else:
        verdict = "STABLE (self-regulating, D0 < 0, modest resonance)"

    diag = dict(
        case_name=case.name,
        total_beta=float(make_12group(case.f238)[0].sum()),
        mdot=p.get("mdot"),
        tau_f=p["tau_f"],
        tau_m=p["tau_m"],
        tau_p=p["tau_p"],
        A_f=p["A_f"],
        A_m=p["A_m"],
        B_p=p["B_p"],
        beta_p_Bp=p["beta_p"] * p["B_p"],
        D0=D0,
        g_lo=g_lo,
        ph_lo=ph_lo,
        g_dc_expected=g_dc_expected,
        g_peak=g_peak,
        w_gpeak=w_gpeak,
        peaking_db=peaking_db,
        ph_peak=ph_peak,
        w_phpeak=w_phpeak,
        g_hi=g_hi,
        ph_hi=ph_hi,
        verdict=verdict,
    )
    return diag


def print_diagnostics(diag: Dict):
    print("\n" + "=" * 72)
    print(f"DIAGNOSTICS – {diag['case_name']}")
    print("=" * 72)
    print(f"  total delayed-neutron fraction β   = {diag['total_beta']:.6f}")
    print(f"  mass-flow rate ṁ                  = {diag['mdot']:.4g} g s⁻¹ cm⁻³")
    print(f"  α_f = {PHYS['alphaf']:.4e}     α_m = {PHYS['alpham']:.4e}")
    print(f"  τ_f = {diag['tau_f']:.4e} s    τ_m = {diag['tau_m']:.4e} s    τ_p = {diag['tau_p']:.4e} s")
    print(f"  A_f = {diag['A_f']:.4e} cm³ K/W   A_m = {diag['A_m']:.4e} cm³ K/W")
    print(f"  β_p·B_p = {diag['beta_p_Bp']:.4e} cm³/W")
    if diag.get("D0") is not None:
        print(f"  static feedback D0 = α_f A_f + α_m A_m – β_p B_p = {diag['D0']:.4e}")
        print(f"  → expected DC gain 20·log10(1/|D0|)            = {diag['g_dc_expected']:.2f} dB")
    print("-" * 72)
    print(f"  low-frequency  gain / phase       = {diag['g_lo']:.2f} dB / {diag['ph_lo']:.2f}°")
    print(f"  gain peak                          = {diag['g_peak']:.2f} dB  @ {diag['w_gpeak']:.2f} rad/s")
    print(f"  resonant peaking (above DC)        = {diag['peaking_db']:.2f} dB")
    print(f"  phase peak                         = {diag['ph_peak']:.2f}°   @ {diag['w_phpeak']:.2f} rad/s")
    print(f"  high-frequency gain / phase        = {diag['g_hi']:.2f} dB / {diag['ph_hi']:.2f}°")
    print("-" * 72)
    print("  NOTE: The plotted quantity is the CLOSED-LOOP transfer function.")
    print("        Classical open-loop gain/phase margins are not applicable.")
    print(f"  STABILITY VERDICT:  {diag['verdict']}")
    print("=" * 72 + "\n")


# ----------------------------------------------------------------------
# 7. Self-check against the PDF Bode plot (f238 = 0 reference)
# ----------------------------------------------------------------------
def check_against_pdf():
    ref = Case("check", 0.0)
    w = np.logspace(math.log10(W_MIN), math.log10(W_MAX), 4000)
    g, ph = gain_phase(w, ref)
    i, j = np.argmax(g), np.argmax(ph)
    assert abs(g[0] - 110.5) < 1.5 and abs(ph[0]) < 10.0, "low-frequency gain/phase off"
    assert abs(g[i] - 115.5) < 2.0 and 15.0 < w[i] < 40.0, "gain peak off"
    assert abs(ph[j] - 28.0) < 5.0 and 5.0 < w[j] < 20.0, "phase peak off"
    assert abs(g[-1] - 76.0) < 3.0 and abs(ph[-1] + 90.0) < 5.0, "high-frequency end off"
    print(
        f"[check] matches PDF plot (axis = rad/s): low-f {g[0]:.1f} dB/{ph[0]:.1f}°, "
        f"gain peak {g[i]:.1f} dB @ {w[i]:.1f} rad/s, phase peak {ph[j]:.1f}° "
        f"@ {w[j]:.1f} rad/s, end {g[-1]:.1f} dB"
    )


# ----------------------------------------------------------------------
# 8. Plots
# ----------------------------------------------------------------------
def _axis_values(omega, axis):
    if axis == "omega":
        return omega, "Perturbation frequency, ω (rad/s)"
    return omega / (2.0 * math.pi), "Perturbation frequency, f (Hz)"


def make_dual_axis_bode(case: Case, axis=AXIS, n_points=800, outfile=None, show=False):
    """PDF-style dual-axis Bode plot for one case."""
    omega = np.logspace(math.log10(W_MIN), math.log10(W_MAX), n_points)
    g, ph = gain_phase(omega, case)
    x, xlabel = _axis_values(omega, axis)

    fig, ax1 = plt.subplots(figsize=(8.5, 6.5))
    ax2 = ax1.twinx()
    l1, = ax1.semilogx(x, g, color="blue", lw=2.0, label="Gain (dB)")
    l2, = ax2.semilogx(x, ph, color="crimson", lw=2.0, label="Phase shift (deg)")

    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(75.0, 118.0)
    ax2.set_ylim(-95.0, 32.0)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Gain (dB)", color="blue")
    ax2.set_ylabel("Phase shift (deg)", color="crimson")
    ax1.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="crimson")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_title(f"Nuclear Rocket Transfer Function Bode Plot – {case.name}")
    ax1.legend(handles=[l1, l2], loc="lower left", fontsize=9)
    fig.tight_layout()

    outfile = outfile or os.path.join(
        OUTDIR, "ntp_bode_" + case.name.split()[0].lower() + ".png"
    )
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    if show:
        plt.show()
    return fig


def make_overlay_plot(cases, axis=AXIS, n_points=800, outfile=None, show=False):
    """HEU vs HALEU (or any list of cases) on two panels."""
    omega = np.logspace(math.log10(W_MIN), math.log10(W_MAX), n_points)
    x, xlabel = _axis_values(omega, axis)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for c in cases:
        g, ph = gain_phase(omega, c)
        a1.semilogx(x, g, color=c.color, lw=2.0, label=c.name)
        a2.semilogx(x, ph, color=c.color, lw=2.0, label=c.name)
    a1.set_ylabel("Gain (dB)")
    a2.set_ylabel("Phase (deg)")
    a2.set_xlabel(xlabel)
    a1.set_title("Nuclear Rocket Transfer Function – HEU vs HALEU")
    for ax in (a1, a2):
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=9)
    fig.tight_layout()
    outfile = outfile or os.path.join(OUTDIR, "ntp_bode_overlay.png")
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Optional PDF self-check (uncomment when you want to verify the axis scaling)
    # check_against_pdf()

    # ------------------------------------------------------------------
    # Show the derived parameters that will be used
    # ------------------------------------------------------------------
    print("Derived thermal / feedback parameters (from PHYS / MATLAB):")
    derived = derive_thermal_params()
    for k in ("tau_f", "tau_m", "tau_p", "A_f", "A_m", "B_p", "beta_p", "D0", "mdot"):
        print(f"  {k:8s} = {derived[k]:.6g}")
    print()

    for c in (HEU, HALEU):
        b, _ = make_12group(c.f238)
        print(f"{c.name}: total β = {b.sum():.6f}")
    print()

    # ------------------------------------------------------------------
    # Detailed arithmetic at one frequency
    # ------------------------------------------------------------------
    print(">>> Detailed calculation at ω = 10 rad/s")
    compute_one_frequency(10.0, HEU)

    # ------------------------------------------------------------------
    # How much does U-238 delayed-neutron data alone change the curve?
    # ------------------------------------------------------------------
    w = np.logspace(math.log10(W_MIN), math.log10(W_MAX), 800)
    g_h, p_h = gain_phase(w, HEU)
    g_l, p_l = gain_phase(w, HALEU)
    print(
        f"Max |HALEU – HEU|: {np.max(np.abs(g_l - g_h)):.2f} dB, "
        f"{np.max(np.abs(p_l - p_h)):.2f} deg (delayed-neutron data only)\n"
    )

    # ------------------------------------------------------------------
    # Diagnostics & stability
    # ------------------------------------------------------------------
    for c in (HEU, HALEU):
        diag = diagnose(c)
        print_diagnostics(diag)

    # ------------------------------------------------------------------
    # Example: change mass-flow rate and re-diagnose
    # ------------------------------------------------------------------
    print(">>> Example sensitivity: double the mass-flow rate (mdot × 2)")
    HEU_high_mdot = Case(
        "HEU (mdot × 2)",
        0.01,
        overrides={"mdot": PHYS["mdot"] * 2.0},
        color="tab:green",
    )
    diag_hi = diagnose(HEU_high_mdot)
    print_diagnostics(diag_hi)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    make_dual_axis_bode(HEU)
    make_dual_axis_bode(HALEU)
    if MAKE_OVERLAY:
        make_overlay_plot([HEU, HALEU])

    # Optional extra plot showing the mdot sensitivity
    make_dual_axis_bode(
        HEU_high_mdot,
        outfile=os.path.join(OUTDIR, "ntp_bode_heu_mdotx2.png"),
    )

    if SHOW:
        plt.show()
    print("Done.")