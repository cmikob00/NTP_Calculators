#!/usr/bin/env python3
"""
Educational NTR closed-loop transfer-function & Bode-plot calculator  (REVISION 3)
HEU and HALEU cases with 12 delayed-neutron groups (6 U-235 thermal + 6 U-238 fast).
Follows the derivation in the class PDF ("Reactor stability model"), step by step.

CHANGES FROM REVISION 1
  1. SIGN FIX.  The PDF block diagram has MINUS signs at all three summing junctions,
     so the feedback blocks are entered as positive numbers (negative reactivity
     feedback):   K_F G_F = -alpha_f A_f/(1+i w tau_f), etc.  Rev. 1 added the raw
     (negative) alpha*A terms, giving ~-177 deg phase at low frequency.
  2. X-AXIS IS ANGULAR FREQUENCY (rad/s).  The PDF plot is labelled "Hz", but I
     digitised its gain and phase curves (page 12 image) and compared them with this
     model: with the axis read as omega in rad/s the RMS error is 0.09 dB (gain) and
     0.29 deg (phase); read as Hz (omega = 2*pi*f) it is 9.3 dB and 28 deg.  A free
     fit of the frequency scale gives 0.997 (i.e. 1.0).  So the "Hz" label in the PDF
     is a mislabel.  AXIS = "omega" (default) reproduces the PDF; AXIS = "hz" plots
     the same curves against f = omega/(2*pi).
  3. U-238 handling.  f238 is the fraction of fission NEUTRONS born in U-238, so
     beta_i(mix) = (1-f238)*beta235_i + f238*beta238_i is exact.  Presets follow
     period_solver.py (ANL/NSE-20/55): HEU 0.01, HALEU 0.08.  If you only know the
     fission-RATE fraction F238, use neutron_fraction_from_fission_fraction().
  4. Delayed-neutron data are built as a_i * beta (same numbers as period_solver.py):
     beta235 = 0.0065, beta238 = 0.0148 (Keepin, Wimett & Zeigler 1957).
  5. Phase is unwrapped; the sweep is vectorised.
  6. NEW: one dual-axis figure per case (gain on the left axis in blue, phase on the
     right axis in red, like the PDF), plus an optional HEU-vs-HALEU overlay.
  7. A self-check against the PDF plot values is run before plotting.

KNOWN LIMITATIONS
  * HEU and HALEU share the PDF's thermal/feedback/Lambda parameters by default, so
    the two cases differ ONLY through delayed-neutron data (< ~1 dB).  A real HALEU
    model needs its own alpha's and Lambda: use the `overrides` field of Case.
  * beta is the physical delayed fraction (no delayed-neutron importance weighting).
  * Linearised (small-perturbation) model about a steady state at power q0.
"""

import cmath
import math
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt" and os.uname().sysname != "Darwin":
    matplotlib.use("Agg")          # headless: figures are saved to PNG
import matplotlib.pyplot as plt

AXIS = "omega"                      # "omega" (rad/s, matches the PDF plot) or "hz"
SHOW = True                         # plt.show() after saving the figures
MAKE_OVERLAY = True                 # also write an HEU-vs-HALEU comparison figure
OUTDIR = "."

# Frequency sweep, rad/s.  The PDF plot spans ~6e-4 to ~6.4e3 rad/s.
W_MIN, W_MAX = 6.0e-4, 6.4e3

# ----------------------------------------------------------------------
# 1. Thermal / feedback parameters (PDF Bode-plot page)
# ----------------------------------------------------------------------
PARAMS = dict(
    q0=3750.0,          # W cm^-3
    Lambda=1.00e-4,     # s  (neutron generation time)
    tau_f=6.30e-2,      # s
    A_f=3.10e-2,        # cm^3 K W^-1
    alpha_f=-2.00e-5,   # K^-1
    tau_m=9.10e-2,      # s
    A_m=2.80e-2,        # cm^3 K W^-1
    alpha_m=-4.50e-5,   # K^-1
    tau_p=3.10e-3,      # s
    B_p=1.11e-8,        # g W^-1
    beta_p=100.0,       # cm^3 g^-1   (beta_p * B_p = 1.11e-6 cm^3/W)
)

# ----------------------------------------------------------------------
# 2. Delayed-neutron data (Keepin, Wimett & Zeigler, Phys. Rev. 107, 1044 (1957))
#    Built as relative abundance a_i x total beta, identical to period_solver.py.
# ----------------------------------------------------------------------
# U-235, thermal fission
BETA235_TOTAL = 0.0065
a_235   = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
lam_235 = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
beta_235 = BETA235_TOTAL * a_235

# U-238, fast fission
BETA238_TOTAL = 0.0148
a_238   = np.array([0.013, 0.137, 0.162, 0.388, 0.225, 0.075])
lam_238 = np.array([0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02])
beta_238 = BETA238_TOTAL * a_238

NU_235 = 2.43       # neutrons per thermal U-235 fission
NU_238 = 2.80       # neutrons per fast U-238 fission (approx.; ~2.6-2.9)


def neutron_fraction_from_fission_fraction(F238, nu5=NU_235, nu8=NU_238):
    """Convert a U-238 fission-RATE fraction to a fission-NEUTRON fraction (nu-weighting)."""
    return F238 * nu8 / (F238 * nu8 + (1.0 - F238) * nu5)


def make_12group(f238):
    """
    f238 = fraction of fission NEUTRONS born in U-238 (fast fission); rest is U-235.
      HEU   preset 0.01   HALEU preset 0.08   (ANL/NSE-20/55, as in period_solver.py)
    Returns (beta[12], lam[12]); beta.sum() is the total delayed fraction.
    """
    beta = np.concatenate([(1.0 - f238) * beta_235, f238 * beta_238])
    lam = np.concatenate([lam_235, lam_238])
    return beta, lam


@dataclass
class Case:
    name: str
    f238: float
    overrides: dict = field(default_factory=dict)   # e.g. {"alpha_f": -3.0e-5, "Lambda": 5e-5}
    color: str = "tab:blue"

    def params(self):
        p = dict(PARAMS)
        p.update(self.overrides)
        return p


HEU = Case("HEU (f238 = 0.01)", 0.01, color="tab:blue")
HALEU = Case("HALEU (f238 = 0.08)", 0.08, color="tab:red")


# ----------------------------------------------------------------------
# 3. Transfer function (vectorised; works for scalar or array omega)
# ----------------------------------------------------------------------
def transfer_function(omega, case):
    """K_RT G_RT(i*omega) = theta_o/theta_i for the closed loop. omega in rad/s."""
    p = case.params()
    beta, lam = make_12group(case.f238)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * w

    # Step 8 (kinetics):  D_kin/q0 = [ s*Lambda + sum_i s*beta_i/(s+lam_i) ] / q0
    D_kin = s * p["Lambda"] + np.sum(
        s[:, None] * beta[None, :] / (s[:, None] + lam[None, :]), axis=1)
    kin = D_kin / p["q0"]

    # Step 7 (feedback, entered as POSITIVE = negative reactivity feedback)
    lag_f = 1.0 + s * p["tau_f"]
    lag_m = 1.0 + s * p["tau_m"]
    lag_p = 1.0 + s * p["tau_p"]
    fuel = -p["alpha_f"] * p["A_f"] / lag_f
    mod = -p["alpha_m"] * p["A_m"] / (lag_f * lag_m)
    prop = +p["beta_p"] * p["B_p"] / (lag_f * lag_m * lag_p)

    # print(f'Denominator terms:')
    # print(f'Point kinetics: {kin}')
    # print(f'Fuel: {fuel}')
    # print(f'Moderator: {mod}')
    # print(f'Propellant: {prop}')

    return 1.0 / (kin + fuel + mod + prop)


def gain_phase(omega, case):
    tf = transfer_function(omega, case)
    gain_db = 20.0 * np.log10(np.abs(tf))
    phase = np.degrees(np.unwrap(np.angle(tf)))
    return gain_db, phase


# ----------------------------------------------------------------------
# 4. Single-frequency walk-through (every arithmetic step visible)
# ----------------------------------------------------------------------
def compute_one_frequency(omega, case, label=""):
    p = case.params()
    beta, lam = make_12group(case.f238)
    s = 1j * omega
    print("=" * 70)
    print(f"omega = {omega:.4g} rad/s ({omega / (2 * math.pi):.4g} Hz)   {label or case.name}")
    print(f"total beta = {beta.sum():.6f}")
    print("=" * 70)

    D_kin = s * p["Lambda"]
    print("\n--- Step A: kinetics denominator ---")
    print(f"  prompt term s*Lambda = {D_kin}")
    for i, (b, l) in enumerate(zip(beta, lam), start=1):
        term = s * b / (s + l)
        D_kin += term
        print(f"  group {i:2d}: s*beta/(s+lambda) = {term}")
    kin = D_kin / p["q0"]
    print(f"  D_kin = {D_kin}\n  D_kin/q0 = {kin}")

    lag_f, lag_m, lag_p = (1 + s * p["tau_f"]), (1 + s * p["tau_m"]), (1 + s * p["tau_p"])
    fuel = -p["alpha_f"] * p["A_f"] / lag_f
    mod = -p["alpha_m"] * p["A_m"] / (lag_f * lag_m)
    prop = +p["beta_p"] * p["B_p"] / (lag_f * lag_m * lag_p)
    print("\n--- Steps B-D: feedback (positive = negative reactivity feedback) ---")
    print(f"  fuel       -alpha_f*A_f/(1+s*tau_f)           = {fuel}")
    print(f"  moderator  -alpha_m*A_m/(lag_f*lag_m)         = {mod}")
    print(f"  propellant +beta_p*B_p/(lag_f*lag_m*lag_p)    = {prop}")

    denom = kin + fuel + mod + prop
    TF = 1.0 / denom
    print("\n--- Step E: closed-loop result ---")
    print(f"  denominator = {denom}\n  TF = {TF}")
    print(f"  |TF| = {abs(TF):.6e}   Gain = {20 * math.log10(abs(TF)):.2f} dB   "
          f"Phase = {math.degrees(cmath.phase(TF)):.2f} deg\n")
    return TF


# ----------------------------------------------------------------------
# 5. Self-check against the PDF Bode plot (f238 = 0 reference)
# ----------------------------------------------------------------------
def check_against_pdf():
    ref = Case("check", 0.0)
    w = np.logspace(math.log10(W_MIN), math.log10(W_MAX), 4000)
    g, ph = gain_phase(w, ref)
    i, j = np.argmax(g), np.argmax(ph)
    assert abs(g[0] - 110.5) < 1.0 and abs(ph[0]) < 8.0, "low-frequency gain/phase off"
    assert abs(g[i] - 115.5) < 1.0 and 20.0 < w[i] < 33.0, "gain peak off"
    assert abs(ph[j] - 28.0) < 3.0 and 8.0 < w[j] < 14.0, "phase peak off"
    assert abs(g[-1] - 76.0) < 1.5 and abs(ph[-1] + 90.0) < 3.0, "high-frequency end off"
    print(f"[check] matches PDF plot (axis = rad/s): low-f {g[0]:.1f} dB/{ph[0]:.1f} deg, "
          f"gain peak {g[i]:.1f} dB @ {w[i]:.1f} rad/s, phase peak {ph[j]:.1f} deg "
          f"@ {w[j]:.1f} rad/s, end {g[-1]:.1f} dB")


# ----------------------------------------------------------------------
# 6. Plots
# ----------------------------------------------------------------------
def _axis_values(omega, axis):
    if axis == "omega":
        return omega, "Perturbation frequency, \u03c9 (rad/s)"
    return omega / (2.0 * math.pi), "Perturbation frequency, f (Hz)"


def make_dual_axis_bode(case, axis=AXIS, n_points=800, outfile=None, show=False):
    """
    PDF-style plot for ONE case: gain (blue, left axis) and phase (red, right axis)
    on the same log-frequency axis.
    """
    omega = np.logspace(math.log10(W_MIN), math.log10(W_MAX), n_points)
    g, ph = gain_phase(omega, case)
    x, xlabel = _axis_values(omega, axis)

    fig, ax1 = plt.subplots(figsize=(8.5, 6.5))
    ax2 = ax1.twinx()
    l1, = ax1.semilogx(x, g, color="blue", lw=2.0, label="Gain (dB)")
    l2, = ax2.semilogx(x, ph, color="crimson", lw=2.0, label="Phase shift (deg)")

    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(75.0, 118.0)              # same limits for every case -> comparable
    ax2.set_ylim(-95.0, 32.0)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Gain (dB)", color="blue")
    ax2.set_ylabel("Phase shift (deg)", color="crimson")
    ax1.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="crimson")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_title(f"Nuclear Rocket Transfer Function Bode Plot \u2013 {case.name}")
    ax1.legend(handles=[l1, l2], loc="lower left", fontsize=9)
    fig.tight_layout()

    outfile = outfile or os.path.join(
        OUTDIR, "ntp_bode_" + case.name.split()[0].lower() + ".png")
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    if show:
        plt.show()
    return fig


def make_overlay_plot(cases, axis=AXIS, n_points=800, outfile=None, show=False):
    """Optional: HEU and HALEU together (gain and phase panels)."""
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
    a1.set_title("Nuclear Rocket Transfer Function \u2013 HEU vs HALEU")
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
# 7. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # check_against_pdf()  # this step ensures that parameters match Emrich's plot

    for c in (HEU, HALEU):
        b, _ = make_12group(c.f238)
        print(f"{c.name}: total beta = {b.sum():.6f}")
    print()

    print(">>> Detailed calculation at omega = 10 rad/s")
    compute_one_frequency(10.0, HEU)

    # How much does U-238 delayed-neutron data alone change the curve?
    w = np.logspace(math.log10(W_MIN), math.log10(W_MAX), 800)
    g_h, p_h = gain_phase(w, HEU)
    g_l, p_l = gain_phase(w, HALEU)
    print(f"Max |HALEU - HEU|: {np.max(np.abs(g_l - g_h)):.2f} dB, "
          f"{np.max(np.abs(p_l - p_h)):.2f} deg (delayed-neutron data only)\n")

    make_dual_axis_bode(HEU)                  # gain + phase together, HEU
    make_dual_axis_bode(HALEU)                # gain + phase together, HALEU
    if MAKE_OVERLAY:
        make_overlay_plot([HEU, HALEU])
    if SHOW:
        plt.show()
    print("Done.")