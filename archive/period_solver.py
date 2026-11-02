"""
Point-kinetics study of a NERVA-style HEU core vs. a simplified HALEU nuclear
thermal rocket (NTR) concept.

Contents
  1. Delayed-neutron data with explicit provenance / verification status
  2. Isotope-resolved delayed-neutron groups (U-235 thermal + U-238 fast)
  3. Robust inhour-equation solver (ALL roots, no hard-coded bracket)
  4. Analytic step-insertion response from the inhour roots
  5. Point-kinetics ODE with ramp reactivity and lumped thermal feedback
  6. Figures + a JSON record of every parameter and its source

Conventions
  rho   = (k - 1)/k   (absolute, dimensionless);  1 pcm = 1e-5;  $1 = beta_total
  Lam   = neutron GENERATION time (Lambda = l/k).  For k ~ 1 the two differ by <1 %.
  omega = 1/T is the inverse period (s^-1).

PROVENANCE TAGS used throughout the comments
  [VERIFIED]   value checked against a published source (cited)
  [PARTIAL]    some entries checked, others NOT -- see note, check before publishing
  [ESTIMATE]   engineering assumption or illustrative value; NOT from a peer-reviewed
               source.  Replace with core-specific numbers from a neutronics code
               (e.g., MCNP/Serpent adjoint-weighted kinetics) before drawing conclusions.
"""

import json
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")            # remove this line for interactive windows
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

OUTDIR = "."                      # where figures and the parameter record are written


# ======================================================================
# 1. DELAYED-NEUTRON DATA
# ======================================================================
@dataclass(frozen=True)
class DelayedData:
    name: str
    lam: np.ndarray      # decay constants, s^-1
    a: np.ndarray        # relative abundances (sum = 1)
    beta: float          # total delayed-neutron fraction (delayed n per fission neutron)
    status: str
    source: str


# U-235, THERMAL fission.  [VERIFIED against a secondary tabulation]
#   Keepin, Wimett & Zeigler, Phys. Rev. 107, 1044 (1957); same data tabulated in
#   Hetrick, "Dynamics of Nuclear Reactors", ANS (1993): yields 0.00052, 0.00346,
#   0.00310, 0.00624, 0.00182, 0.00066 n/fission, total 0.0158, beta = 0.0065.
#   NOTE: your original script used exactly this set.
U235_THERMAL = DelayedData(
    name="U-235 thermal",
    lam=np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]),
    a=np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042]),
    beta=0.0065,
    status="VERIFIED (vs. Hetrick 1993 tabulation of Keepin data)",
    source="Keepin, Wimett & Zeigler, Phys. Rev. 107, 1044 (1957); Hetrick (1993)",
)

# U-238, FAST fission.  [PARTIAL]
#   Total yield nu_d = 0.0412 +/- 0.0017 n/fission is VERIFIED (Keepin et al., LA-2118 /
#   Phys. Rev. 107, 1044), and beta_eff(U-238) ~ 0.0148 is quoted in the literature.
#   The six lam/a values below are the Keepin fast-fission U-238 set as I recall them;
#   they were NOT re-checked against the original table.  CHECK before publishing.
U238_FAST = DelayedData(
    name="U-238 fast",
    lam=np.array([0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02]),
    a=np.array([0.013, 0.137, 0.162, 0.388, 0.225, 0.075]),
    beta=0.0148,
    status="PARTIAL (yield verified; group constants unchecked)",
    source="Keepin, Wimett & Zeigler, Phys. Rev. 107, 1044 (1957)",
)


def build_groups(f238):
    """
    Isotope-resolved delayed groups.  f238 = fraction of fission NEUTRONS born in U-238
    (fast fission); the rest are U-235 thermal.  Each isotope keeps its own decay
    constants, so we carry 12 groups (no averaging of lambda needed):
        beta_i(k, group) = w_k * beta_k * a_k,i
    Returns (lam, beta_i) sorted by lambda.  beta_i.sum() is the total beta.
    Caveat: this is a fission-weighted beta, not an adjoint-weighted beta_eff.
    """
    lam = [U235_THERMAL.lam]
    bi = [(1.0 - f238) * U235_THERMAL.beta * U235_THERMAL.a]
    if f238 > 0.0:
        lam.append(U238_FAST.lam)
        bi.append(f238 * U238_FAST.beta * U238_FAST.a)
    lam, bi = np.concatenate(lam), np.concatenate(bi)
    order = np.argsort(lam)
    return lam[order], bi[order]


# ======================================================================
# 2. CORE DEFINITIONS
# ======================================================================
@dataclass
class Core:
    name: str
    f238: float            # neutron fraction from U-238 fast fission
    Lam: float             # generation time, s
    Lam_range: tuple       # sensitivity band for Lam
    alpha_T: float         # lumped temperature coefficient, d(rho)/dK
    provenance: dict

    def __post_init__(self):
        self.lam, self.beta_i = build_groups(self.f238)
        self.beta = float(self.beta_i.sum())


# NERVA-style HEU, graphite-moderated, thermal spectrum
HEU = Core(
    name="NERVA-style HEU",
    f238=0.0,        # [ESTIMATE] negligible U-238 fission in a ~93 % enriched thermal core
    Lam=1.0e-4,      # [ESTIMATE] order-of-magnitude thermal graphite-core value; no source
    Lam_range=(3.0e-5, 3.0e-4),   # [ESTIMATE] sensitivity band, not a literature range
    alpha_T=-1.0e-6,  # [ESTIMATE] illustrative; NERVA-specific values need a source
    provenance={
        "delayed groups": "VERIFIED: U-235 thermal (Keepin 1957 via Hetrick 1993)",
        "f238": "ESTIMATE: ~0 for HEU; not from a source",
        "Lambda": "ESTIMATE: 1e-4 s, band 3e-5..3e-4 s; replace with adjoint-weighted value",
        "alpha_T": "ESTIMATE: -1e-6 /K illustrative",
    },
)

# Simplified HALEU (19.75 %) concept
HALEU = Core(
    name="Simplified HALEU",
    f238=0.05,       # [ESTIMATE] gives beta ~0.0069; your earlier 0.0072 target needs f238~0.085.
                     # Replace with the fission-neutron fraction from a Serpent/MCNP run, or
                     # use the beta_eff in ANL/NSE-20/55 (HALEU vs HEU NTP study) directly.
    Lam=8.0e-5,      # [ESTIMATE] not from a source (earlier script text said 6e-5 and 8e-5)
    Lam_range=(1.0e-6, 3.0e-4),   # [ESTIMATE] wide: fast/cermet cores can be << 1e-5 s,
                                  # moderated cores ~1e-4 s
    alpha_T=-1.5e-6,  # [ESTIMATE] magnitude illustrative; the DIRECTION (stronger Doppler in
                      # the HALEU core) is reported in ANL/NSE-20/55, magnitude is not taken
                      # from it
    provenance={
        "delayed groups": "PARTIAL: U-235 thermal verified; U-238 fast groups unchecked",
        "f238": "ESTIMATE: 0.05 assumed",
        "Lambda": "ESTIMATE: 8e-5 s, band 1e-6..3e-4 s",
        "alpha_T": "ESTIMATE: -1.5e-6 /K illustrative",
    },
)

# Lumped thermal model for the feedback case -- ALL [ESTIMATE], illustrative only
P_INIT = 1.0e6      # W   initial power level (n = 1)                       [ESTIMATE]
C_TH = 1.5e6        # J/K lumped core heat capacity (~1500 kg x ~1 kJ/kg/K) [ESTIMATE]
H_COOL = 2.0e5      # W/K lumped propellant cooling conductance             [ESTIMATE]


# ======================================================================
# 3. INHOUR EQUATION: ALL ROOTS
# ======================================================================
def _f(w, rho, Lam, lam, bi):
    """f(omega) = rho - Lam*omega - sum_i beta_i*omega/(omega + lambda_i);  roots = inhour roots."""
    return rho - Lam * w - np.sum(bi * w / (w + lam))


def inhour_roots(rho, Lam, lam, bi):
    """
    All N+1 real roots omega_j of the inhour equation, sorted descending.
    f(omega) is monotone between its poles (-lambda_i), so there is exactly one root in
    each interval between poles, one above the first pole, and one below the last pole
    (the prompt root).  Each is bracketed and solved with brentq -- robust, no polynomial
    conditioning issues, no hard-coded period bracket.  rho must be nonzero.
    """
    if rho == 0.0:
        raise ValueError("rho = 0 has a trivial root omega = 0; use rho != 0")
    poles = np.sort(-lam)[::-1]                    # -lambda_min (closest to 0) first
    args = (rho, Lam, lam, bi)
    roots = []

    # Dominant (stable-period) root: above the first pole
    lo = poles[0] + 1e-9 * abs(poles[0] - poles[1])
    hi = max(rho / Lam, 0.0) * (1.0 + 1e-9) + 1e-12
    roots.append(brentq(_f, lo, hi, args=args, xtol=1e-14, rtol=1e-13))

    # One root between each pair of adjacent poles
    for p_hi, p_lo in zip(poles[:-1], poles[1:]):
        gap = p_hi - p_lo
        roots.append(brentq(_f, p_lo + 1e-9 * gap, p_hi - 1e-9 * gap, args=args,
                            xtol=1e-14, rtol=1e-13))

    # Prompt root: below the last pole
    hi = poles[-1] - 1e-9 * abs(poles[-1])
    lo = hi - 1.0
    while _f(lo, *args) < 0.0:
        lo = hi - 2.0 * (hi - lo)
    roots.append(brentq(_f, lo, hi, args=args, xtol=1e-12, rtol=1e-13))
    return np.array(sorted(roots, reverse=True))


def asymptotic_period(rho, core, Lam=None):
    """Stable period T = 1/omega_max (s).  rho<0 returns a NEGATIVE T (decay e-folding time)."""
    Lam = core.Lam if Lam is None else Lam
    return 1.0 / inhour_roots(rho, Lam, core.lam, core.beta_i)[0]


def step_response(t, rho, core):
    """
    Analytic n(t)/n0 for a STEP reactivity rho from equilibrium, no feedback:
        n(t) = sum_j A_j exp(omega_j t),
        A_j  = [Lam + sum_i beta_i/(w_j+lam_i)] / [Lam + sum_i beta_i*lam_i/(w_j+lam_i)^2]
    (residues of the Laplace-transformed point-kinetics equations).
    """
    w = inhour_roots(rho, core.Lam, core.lam, core.beta_i)
    num = core.Lam + np.array([np.sum(core.beta_i / (wj + core.lam)) for wj in w])
    den = core.Lam + np.array([np.sum(core.beta_i * core.lam / (wj + core.lam) ** 2) for wj in w])
    A = num / den
    t = np.atleast_1d(t)
    return (A[None, :] * np.exp(np.outer(t, w))).sum(axis=1), A


# ======================================================================
# 4. POINT-KINETICS ODE (ramp insertion + lumped thermal feedback)
# ======================================================================
def run_transient(core, ramp_dollars_per_s, rho_final_dollars, t_end, feedback=True,
                  n_cap=2.0e3):
    """
    State y = [n, c_1..c_N, theta]; n is power relative to P_INIT, theta is the lumped
    temperature rise (K).
        dn/dt   = ((rho - beta)/Lam) n + sum lam_i c_i
        dc_i/dt = beta_i n/Lam - lam_i c_i
        C dtheta/dt = P_INIT n - H theta
        rho     = rho_ext(t) + alpha_T theta         (feedback optional)
    rho_ext ramps at ramp_dollars_per_s * beta until rho_final_dollars * beta, then holds.
    Initial condition: equilibrium precursors at n = 1, theta = 0, rho = 0.
    Stops early if n exceeds n_cap (runaway without feedback).
    """
    lam, bi, Lam, beta = core.lam, core.beta_i, core.Lam, core.beta
    rho_f = rho_final_dollars * beta
    rate = ramp_dollars_per_s * beta

    def rhs(t, y):
        n, c, th = y[0], y[1:-1], y[-1]
        rho = min(rate * t, rho_f) + (core.alpha_T * th if feedback else 0.0)
        dn = (rho - beta) / Lam * n + np.dot(lam, c)
        dc = bi * n / Lam - lam * c
        dth = (P_INIT * n - H_COOL * th) / C_TH
        return np.concatenate(([dn], dc, [dth]))

    def runaway(t, y):
        return y[0] - n_cap
    runaway.terminal = True

    y0 = np.concatenate(([1.0], bi / (lam * Lam), [0.0]))
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="Radau", rtol=1e-8, atol=1e-10,
                    events=runaway, dense_output=False,
                    t_eval=np.linspace(0.0, t_end, 4000))
    return sol


# ======================================================================
# 5. SELF-CHECKS
# ======================================================================
def self_checks():
    """Cross-check the root solver, the analytic response and the ODE before plotting."""
    for core in (HEU, HALEU):
        # (a) all roots satisfy the inhour equation
        for rho in (0.3 * core.beta, 0.9 * core.beta, 1.2 * core.beta, -0.5 * core.beta):
            w = inhour_roots(rho, core.Lam, core.lam, core.beta_i)
            res = max(abs(_f(wj, rho, core.Lam, core.lam, core.beta_i)) for wj in w)
            assert res < 1e-9, f"{core.name}: inhour residual {res:.2e}"
        # (b) analytic amplitudes sum to 1 at t = 0
        _, A = step_response(0.0, 0.3 * core.beta, core)
        assert abs(A.sum() - 1.0) < 1e-8, "amplitudes do not sum to 1"
        # (c) ODE (step, no feedback) agrees with the analytic response
        rho = 0.3 * core.beta
        sol = run_transient(core, ramp_dollars_per_s=1e9, rho_final_dollars=0.3,
                            t_end=30.0, feedback=False)
        n_an, _ = step_response(sol.t, rho, core)
        m = sol.t > 0.05                       # skip the first instants (prompt jump)
        err = np.max(np.abs(sol.y[0][m] / n_an[m] - 1.0))
        assert err < 5e-3, f"{core.name}: ODE vs analytic mismatch {err:.2e}"
        print(f"[check] {core.name}: roots, amplitudes, ODE-vs-analytic OK "
              f"(max rel. err {err:.1e}); beta = {core.beta:.5f}")


# ======================================================================
# 6. FIGURES
# ======================================================================
COL = {"HEU": "tab:blue", "HALEU": "tab:red"}


def period_curve(core, rho_abs, Lam):
    return np.array([asymptotic_period(r, core, Lam) for r in rho_abs])


def fig_period():
    rho = np.linspace(3e-4, 1.0e-2, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for core, key, ls in ((HEU, "HEU", "-"), (HALEU, "HALEU", "--")):
        T = period_curve(core, rho, core.Lam)
        Tlo = period_curve(core, rho, core.Lam_range[0])
        Thi = period_curve(core, rho, core.Lam_range[1])
        lab = f"{core.name} (\u03b2={core.beta:.4f}, \u039b={core.Lam:.1e} s)"
        for ax, x in ((a1, rho * 1e5), (a2, rho / core.beta)):
            ax.semilogy(x, T, color=COL[key], ls=ls, lw=2.2, label=lab)
            ax.fill_between(x, np.minimum(Tlo, Thi), np.maximum(Tlo, Thi),
                            color=COL[key], alpha=0.12, lw=0)
        a1.axvline(core.beta * 1e5, color=COL[key], ls=":", alpha=0.7)
    a2.axvline(1.0, color="k", ls=":", label="prompt critical ($1)")
    a1.set(xlabel="Reactivity (pcm)", ylabel="Stable reactor period T (s)",
           title="Period vs. reactivity (pcm)", xlim=(0, 1000), ylim=(1e-3, 1e4))
    a2.set(xlabel="Reactivity (dollars)", ylabel="Stable reactor period T (s)",
           title="Period vs. reactivity (dollars)", xlim=(0.05, 1.5), ylim=(1e-3, 1e4))
    for ax in (a1, a2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Asymptotic period (shaded = \u039b sensitivity band, ESTIMATE ranges)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_period_positive.png", dpi=150)


def fig_shutdown():
    rho = -np.logspace(np.log10(5e-5), np.log10(5e-2), 200)     # -5 pcm ... -5000 pcm
    fig, ax = plt.subplots(figsize=(7, 5))
    for core, key, ls in ((HEU, "HEU", "-"), (HALEU, "HALEU", "--")):
        T = -np.array([asymptotic_period(r, core) for r in rho])   # e-folding decay time
        ax.semilogx(-rho * 1e5, T, color=COL[key], ls=ls, lw=2.2, label=core.name)
    ax.set(xlabel="Negative reactivity (pcm)", ylabel="Asymptotic decay time |T| (s)",
           title="Shutdown: asymptotic e-folding time vs. negative reactivity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_shutdown.png", dpi=150)


def fig_transient():
    # Control-drum-style ramp.  Ramp rate and final worth are [ESTIMATE] scenario inputs.
    ramp, final, t_end = 0.02, 0.5, 120.0      # $/s, $, s
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for core, key in ((HEU, "HEU"), (HALEU, "HALEU")):
        for fb, ls in ((False, ":"), (True, "-")):
            s = run_transient(core, ramp, final, t_end, feedback=fb)
            lab = f"{core.name}, {'with' if fb else 'no'} feedback"
            a1.semilogy(s.t, s.y[0], color=COL[key], ls=ls, lw=2, label=lab)
            if fb:
                a2.plot(s.t, s.y[-1], color=COL[key], lw=2, label=core.name)
    a1.set(xlabel="Time (s)", ylabel="Power relative to initial (n)",
           title=f"Ramp +{ramp} \\$/s to +\\${final}: power response")
    a2.set(xlabel="Time (s)", ylabel="Lumped temperature rise (K)",
           title="Lumped temperature rise (with feedback)")
    for ax in (a1, a2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Illustrative transient: thermal model and feedback coefficients are ESTIMATES",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_transient.png", dpi=150)


# ======================================================================
# 7. MAIN
# ======================================================================
def save_record():
    rec = {}
    for core in (HEU, HALEU):
        rec[core.name] = {
            "beta_total": core.beta, "Lambda_s": core.Lam, "Lambda_range_s": core.Lam_range,
            "f238": core.f238, "alpha_T_per_K": core.alpha_T,
            "lambda_i": core.lam.tolist(), "beta_i": core.beta_i.tolist(),
            "provenance": core.provenance,
        }
    rec["delayed_data_sources"] = {d.name: {"status": d.status, "source": d.source}
                                   for d in (U235_THERMAL, U238_FAST)}
    rec["thermal_model_ESTIMATES"] = {"P_INIT_W": P_INIT, "C_TH_J_per_K": C_TH,
                                      "H_COOL_W_per_K": H_COOL}
    with open(f"{OUTDIR}/run_parameters.json", "w") as fh:
        json.dump(rec, fh, indent=2)


if __name__ == "__main__":
    self_checks()
    fig_period()
    fig_shutdown()
    fig_transient()
    save_record()
    for core in (HEU, HALEU):
        print(f"{core.name}: beta={core.beta:.5f}, "
              f"T(+$0.5)={asymptotic_period(0.5 * core.beta, core):.2f} s, "
              f"T(-$1)={asymptotic_period(-core.beta, core):.1f} s")
    print("Figures and run_parameters.json written to", OUTDIR)