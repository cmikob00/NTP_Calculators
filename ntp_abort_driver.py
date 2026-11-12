#!/usr/bin/env python3
"""
NTP abort-capability driver
===========================

Question answered
-----------------
For each abort time (days after TMI), can each of the four NTP abort
configurations deliver at least the delta-v needed to get back to Earth?

    REQUIRED delta-v  <-  patched_conic_3.py  (Lambert abort model)
    AVAILABLE delta-v <-  delta_v.py          (rocket equation, 4 modes)

Modes (from delta_v.py)
-----------------------
  dedicated  : dedicated abort-stage engine only (DRA vehicle is dead mass)
  dra        : the 3 DRA core-stage engines only (no dedicated stage)
  sequential : dedicated engine to depletion, jettison, then DRA engines
  all_four   : all 4 engines together, each group stops when its tank is dry

How to run
----------
Put this file in the same folder as delta_v.py and patched_conic_3.py, then

    python ntp_abort_feasibility.py

Edit the CONFIGURATION block below to change cases, abort days, vehicle
masses, margin, etc.  Nothing else needs to change.

Feasible  =  available dv >= required dv * (1 + DV_MARGIN_FRACTION)
             (and, if REQUIRE_ENGINE_LIFE, the engines stay inside the
              burn-time limit used by delta_v.py).

Both sides are impulsive (ideal) delta-v, consistent with the project's
impulsive-burn assumption: no gravity or steering losses, no finite-burn
effects.  The required delta-v is the minimum over the return-time window
searched in patched_conic_3.py, so it depends on RETURN_MIN/MAX_DAYS (a
long window lets the optimizer pick slow, cheap returns).
"""

import contextlib
import csv
import io
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

import delta_v as ntp
import patched_conic_3 as pc

# =====================================================================
# CONFIGURATION
# =====================================================================

# ---- Required-delta-v side (patched_conic_3.py) ----------------------
# Each entry is (model, outbound) exactly as in patched_conic_3.RUN_CASES:
#   ("circular", None), ("realistic", "tangential"), ("realistic", "lambert")
# One results table and one figure is produced per case.
REQUIRED_CASES = [("realistic", "lambert")]

# Abort days after TMI.  None -> use patched_conic_3.EMERGENCY_DAYS.
# For smooth curves / better crossover estimates use a dense list, e.g.
#   ABORT_DAYS = list(range(1, 31))
ABORT_DAYS = None

# Override the return-time search window in patched_conic_3 (days after the
# abort burn).  None -> keep whatever patched_conic_3.py has set.
RETURN_MIN_DAYS = None
RETURN_MAX_DAYS = 400     # crew-relevant cap; patched_conic_3.py itself is set to 1000

# ---- Available-delta-v side (delta_v.py) ------------------------------
# Same numbers as the main block of delta_v.py.  Change these if you revise
# the vehicle model (e.g. 185 t / 60 t / 9 t dry).
ISP_S = 900.0
THRUST_PER_ENGINE_N = 25000.0 * ntp.LBF_TO_N
M_DRA_VEHICLE_KG = 187000.0      # post-TMI DRA vehicle incl. its core LH2
DRA_CORE_LH2_KG = 62000.0        # LH2 left in the DRA core stage
M_DED_DRY_KG = 3200.0 + 5400.0 + 2400.0 + 500.0 + 1500.0   # ~13 t
M_DED_PROP_KG = 25000.0

# Extra keyword arguments passed straight to ntp.run_abort_burn, e.g.
#   {"dra_reserve_fraction": 0.05, "engine_life_limit_min": 120.0}
NTP_EXTRA_KWARGS = {}

def abort_vehicle_state(day):
    """
    Vehicle state at the abort, as a function of abort day after TMI.
    Returns (m_dra_vehicle_kg, dra_core_lh2_kg).

    Default: constant (mass depends only weakly on MET; RCS use is small).
    Edit this function if you want, e.g., RCS/boiloff to reduce mass or
    core LH2 with time.
    """
    return M_DRA_VEHICLE_KG, DRA_CORE_LH2_KG

# ---- Feasibility test -------------------------------------------------
DV_MARGIN_FRACTION = 0.0      # e.g. 0.05 -> require 5 % more than the minimum
REQUIRE_ENGINE_LIFE = True    # count a mode as infeasible if engines exceed life limit

# ---- Output -----------------------------------------------------------
OUT_DIR = "figures"           # everything below is written here (created if missing)
FIG_FORMATS = ("png", "pdf")  # png for slides, pdf (vector) for the paper; add "svg" if wanted
FIG_DPI = 300                 # applies to raster formats

SAVE_CSV = True               # ntp_abort_feasibility_<case>.csv
SAVE_FIGURES = True           # feasibility figure (available vs required, margin, grid)

# Plots that come from patched_conic_3.py (delta_v.py itself makes no plots)
SAVE_TRAJECTORY_PLOTS = True  # top-down / edge-on / 3-D view for each abort day
TRAJECTORY_PLOT_DAYS = None   # None -> every day in ABORT_DAYS, or e.g. [5, 15, 30]
SAVE_COMPARISON_PLOT = True   # required dv & arrival V-inf vs abort day, all cases
ANIMATE_DAY = None            # e.g. 15 -> also saves an animation of that abort day
ANIMATION_FPS = 30            # (animation needs Pillow for .gif)

SHOW_FIGURES = True           # only the summary figures are shown; trajectory plots
                              # are saved and closed so 30 windows don't pop up

MODES = ntp.MODES             # ("dedicated", "dra", "sequential", "all_four")

# =====================================================================
# CORE CALCULATIONS
# =====================================================================

def case_tag(model, outbound):
    return pc.case_label(model, outbound).replace("/", "_")

def available_dv(day):
    """Run delta_v.py for all four modes at this abort day (output silenced)."""
    m_dra, lh2 = abort_vehicle_state(day)
    out = {}
    for mode in MODES:
        with contextlib.redirect_stdout(io.StringIO()):     # delta_v.py is chatty
            r = ntp.run_abort_burn(
                mode, ISP_S, m_dra, lh2, M_DED_DRY_KG, M_DED_PROP_KG,
                thrust_per_engine_N=THRUST_PER_ENGINE_N,
                **NTP_EXTRA_KWARGS)
        out[mode] = r
    return out

def evaluate_case(model, outbound):
    """Required vs available delta-v for every abort day in one planet case."""
    if RETURN_MIN_DAYS is not None:
        pc.RETURN_MIN_DAYS = RETURN_MIN_DAYS
    if RETURN_MAX_DAYS is not None:
        pc.RETURN_MAX_DAYS = RETURN_MAX_DAYS

    pc.configure(model, outbound)
    days = ABORT_DAYS if ABORT_DAYS is not None else pc.EMERGENCY_DAYS

    header = {
        "label": pc.case_label(model, outbound),
        "launch": pc.t_to_date(pc.T_LAUNCH) if model == "realistic" else None,
        "return_window": (pc.RETURN_MIN_DAYS, pc.RETURN_MAX_DAYS),
    }

    rows = []
    for day in days:
        row = {"day": day, "req": None, "return_days": None, "v_inf": None, "modes": {}}
        try:
            req = pc.calculate_return(day)
            row["req"] = req["delta_v"]                    # km/s
            row["return_days"] = req["return_days"]
            row["v_inf"] = req["v_inf_arrival"]
        except (ValueError, RuntimeError, OverflowError):
            pass                                           # no Lambert solution

        avail = available_dv(day)
        for mode, r in avail.items():
            a = r["delta_v_km_s"]
            life = bool(r["within_engine_life"])
            if row["req"] is None:
                margin, ok = None, False
            else:
                margin = a - row["req"] * (1.0 + DV_MARGIN_FRACTION)
                ok = margin >= 0.0 and (life or not REQUIRE_ENGINE_LIFE)
            row["modes"][mode] = {
                "avail": a, "life_ok": life, "margin": margin, "ok": ok,
                "burn_min": r["total_burn_time_s"] / 60.0,
            }
        rows.append(row)
    return header, rows

def crossover_days(rows):
    """
    For each mode, the last abort day that is still feasible and, where the
    margin changes sign between two evaluated days, a linearly interpolated
    day at which required dv overtakes available dv.
    """
    out = {}
    for mode in MODES:
        seq = [(r["day"], r["modes"][mode]["margin"], r["modes"][mode]["ok"])
               for r in rows if r["modes"][mode]["margin"] is not None]
        if not seq:
            out[mode] = "no required-dv solution"
            continue
        if all(ok for _, _, ok in seq):
            out[mode] = f"feasible at every day evaluated (through day {seq[-1][0]})"
            continue
        if not any(ok for _, _, ok in seq):
            out[mode] = f"infeasible at every day evaluated (from day {seq[0][0]})"
            continue
        last_ok = max(d for d, _, ok in seq if ok)
        txt = f"last feasible evaluated day = {last_ok}"
        for (d0, m0, ok0), (d1, m1, ok1) in zip(seq, seq[1:]):
            if m0 is not None and m1 is not None and m0 >= 0.0 > m1:
                d_star = d0 + (d1 - d0) * m0 / (m0 - m1)
                txt += f"; dv margin crosses zero near day {d_star:.1f}"
                break
        out[mode] = txt
    return out

# =====================================================================
# OUTPUT
# =====================================================================

def save_fig(fig, stem):
    """Save one figure in every format in FIG_FORMATS under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in FIG_FORMATS:
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"Wrote {path}")

def save_trajectory_plots(header, model, outbound):
    """
    Save patched_conic_3.plot_abort() for each requested abort day.
    Must be called right after evaluate_case() for the same case, so that
    patched_conic_3 is still configured for it.
    """
    days = TRAJECTORY_PLOT_DAYS if TRAJECTORY_PLOT_DAYS is not None else \
        (ABORT_DAYS if ABORT_DAYS is not None else pc.EMERGENCY_DAYS)
    tag = case_tag(model, outbound)
    pc.SAVE_FIGURES = False                  # we do the saving here instead
    for day in days:
        try:
            fig = pc.plot_abort(day, show=False)
        except (ValueError, RuntimeError, OverflowError) as err:
            print(f"Trajectory plot skipped for day {day}: {err}")
            continue
        save_fig(fig, f"trajectory_{tag}_day{day:03d}")
        plt.close(fig)                       # free memory; don't pop up 30 windows

def save_animation(model, outbound):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"animation_{case_tag(model, outbound)}_day{ANIMATE_DAY:03d}.gif")
    anim = pc.animate_abort(ANIMATE_DAY, save_as=path)

    print(f"Wrote {path}")
    return anim

def save_comparison_plot(all_results):
    """patched_conic_3.plot_comparison() built from the driver's own rows."""
    pc_results = {}
    for label, (header, rows) in all_results.items():
        pc_results[label] = [
            (r["day"], None if r["req"] is None else
             {"delta_v": r["req"], "v_inf_arrival": r["v_inf"]})
            for r in rows]
    pc.SAVE_FIGURES = False
    fig = pc.plot_comparison(pc_results, show=False)
    save_fig(fig, "required_dv_comparison")
    return fig

def print_results(header, rows):
    w = 28 + 17 * len(MODES)
    print()
    print("=" * w)
    print(f"REQUIRED-DV CASE: {header['label']}"
          + (f"   (TMI {header['launch']})" if header["launch"] else ""))
    print(f"Return-time window searched: {header['return_window'][0]}-"
          f"{header['return_window'][1]} d     "
          f"Margin required: {DV_MARGIN_FRACTION*100:.1f} %     "
          f"Engine-life test: {'on' if REQUIRE_ENGINE_LIFE else 'off'}")
    print("-" * w)
    print(f"{'Abort':>5} | {'Req dV':>7} | {'Ret.':>5} | "
          + " | ".join(f"{m:>13}" for m in MODES))
    print(f"{'day':>5} | {'km/s':>7} | {'days':>5} | "
          + " | ".join(f"{'avail  ok?':>13}" for _ in MODES))
    print("-" * w)
    for r in rows:
        req = f"{r['req']:7.3f}" if r["req"] is not None else f"{'n/a':>7}"
        ret = f"{r['return_days']:5d}" if r["return_days"] is not None else f"{'n/a':>5}"
        cells = []
        for m in MODES:
            c = r["modes"][m]
            flag = "YES" if c["ok"] else "NO"
            if not c["life_ok"]:
                flag += "*"
            cells.append(f"{c['avail']:7.3f} {flag:>5}")
        print(f"{r['day']:5d} | {req} | {ret} | " + " | ".join(f"{c:>13}" for c in cells))
    print("-" * w)
    print("  * = engines exceed the burn-time life limit used in delta_v.py")
    longest = max((r["return_days"] for r in rows if r["return_days"]), default=0)
    if longest > 500:
        print(f"  NOTE: the required dv assumes returns as long as {longest} d "
              f"(~{longest/365.25:.1f} yr).")
        print("        If the crew cannot wait that long, set RETURN_MAX_DAYS "
              "(e.g. 400) in the CONFIGURATION block.")
    print()
    print("Crossover summary (impulsive delta-v, margin applied):")
    for mode, txt in crossover_days(rows).items():
        print(f"  {mode:11s}: {txt}")

def write_csv(header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, f"ntp_abort_feasibility_{case_tag(*CASE_KEY[header['label']])}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["abort_day", "required_dv_km_s", "return_days", "arrival_vinf_km_s"]
        for m in MODES:
            cols += [f"{m}_avail_km_s", f"{m}_margin_km_s", f"{m}_engine_life_ok", f"{m}_feasible"]
        w.writerow(cols)
        for r in rows:
            line = [r["day"], r["req"], r["return_days"], r["v_inf"]]
            for m in MODES:
                c = r["modes"][m]
                line += [c["avail"], c["margin"], c["life_ok"], c["ok"]]
            w.writerow(line)
    print(f"Wrote {fn}")

def plot_results(header, rows):
    days = np.array([r["day"] for r in rows])
    req = np.array([np.nan if r["req"] is None else r["req"] * (1 + DV_MARGIN_FRACTION)
                    for r in rows])
    colors = {"dedicated": "tab:purple", "dra": "tab:green",
              "sequential": "tab:orange", "all_four": "tab:blue"}
    marks = {"dedicated": "s", "dra": "^", "sequential": "D", "all_four": "o"}

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, height_ratios=[1.6, 1.0], figure=fig)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, :])

    # ---- available vs required -------------------------------------
    a1.plot(days, req, "k-o", lw=3, ms=7, zorder=5,
            label=f"Required (Lambert min"
                  + (f" +{DV_MARGIN_FRACTION*100:.0f}%" if DV_MARGIN_FRACTION else "") + ")")
    for m in MODES:
        av = np.array([r["modes"][m]["avail"] for r in rows])
        ok = np.array([r["modes"][m]["ok"] for r in rows])
        a1.plot(days, av, "-", color=colors[m], lw=1.6, alpha=0.8, label=f"{m} available")
        a1.scatter(days[ok], av[ok], marker=marks[m], color=colors[m], s=55, zorder=6)
        a1.scatter(days[~ok], av[~ok], marker=marks[m], facecolors="none",
                   edgecolors=colors[m], s=55, zorder=6)
    a1.set_xlabel("Abort day after TMI")
    a1.set_ylabel("Delta-v (km/s)")
    a1.set_title("Available NTP delta-v vs required abort delta-v\n(filled = feasible, hollow = not)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8, loc="best")

    # ---- margin -----------------------------------------------------
    a2.axhline(0, color="k", lw=1.5)
    for m in MODES:
        mg = np.array([np.nan if r["modes"][m]["margin"] is None else r["modes"][m]["margin"]
                       for r in rows])
        a2.plot(days, mg, "-" + marks[m], color=colors[m], label=m)
    vals = [r["modes"][m]["margin"] for r in rows for m in MODES
            if r["modes"][m]["margin"] is not None] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = 0.1 * max(hi - lo, 0.5)
    a2.axhspan(lo - pad, 0.0, color="red", alpha=0.07)
    a2.set_ylim(lo - pad, hi + pad)
    a2.set_xlabel("Abort day after TMI")
    a2.set_ylabel("Available − required (km/s)")
    a2.set_title("Delta-v margin  (below zero = cannot make it home)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8, loc="best")

    # ---- feasibility matrix ----------------------------------------
    grid = np.array([[1 if r["modes"][m]["ok"] else 0 for r in rows] for m in MODES])
    a3.imshow(grid, aspect="auto", cmap=ListedColormap(["#e57373", "#81c784"]),
              vmin=0, vmax=1)
    a3.set_xticks(range(len(days)))
    a3.set_xticklabels([str(d) for d in days])
    a3.set_yticks(range(len(MODES)))
    a3.set_yticklabels(MODES)
    for i, m in enumerate(MODES):
        for j, r in enumerate(rows):
            c = r["modes"][m]
            txt = "YES" if c["ok"] else "NO"
            if not c["life_ok"]:
                txt += "*"
            a3.text(j, i, txt, ha="center", va="center", fontsize=8, color="k")
    a3.set_xlabel("Abort day after TMI")
    a3.set_title("Can the configuration get the crew home?  (green = yes)")

    fig.suptitle(
        f"NTP abort feasibility  |  required dv: {header['label']}"
        + (f", TMI {header['launch']}" if header["launch"] else "")
        + f"  |  Isp {ISP_S:.0f} s", fontsize=12)
    fig.tight_layout()
    if SAVE_FIGURES:
        save_fig(fig, f"ntp_abort_feasibility_{case_tag(*CASE_KEY[header['label']])}")
    return fig

# =====================================================================
# MAIN
# =====================================================================

CASE_KEY = {pc.case_label(m, o): (m, o) for m, o in REQUIRED_CASES}

def main():
    print("NTP ABORT FEASIBILITY: available (delta_v.py) vs required (patched_conic_3.py)")
    print(f"Isp {ISP_S:.0f} s | DRA vehicle {M_DRA_VEHICLE_KG/1000:.0f} t, "
          f"core LH2 {DRA_CORE_LH2_KG/1000:.0f} t | dedicated stage "
          f"{M_DED_DRY_KG/1000:.1f} t dry + {M_DED_PROP_KG/1000:.0f} t LH2")

    all_results = {}
    for model, outbound in REQUIRED_CASES:
        header, rows = evaluate_case(model, outbound)
        all_results[header["label"]] = (header, rows)
        print_results(header, rows)
        if SAVE_CSV:
            write_csv(header, rows)
        plot_results(header, rows)
        if SAVE_TRAJECTORY_PLOTS:
            save_trajectory_plots(header, model, outbound)
        if ANIMATE_DAY is not None:
            save_animation(model, outbound)

    if SAVE_COMPARISON_PLOT:
        save_comparison_plot(all_results)

    if SHOW_FIGURES:
        plt.show()
    return all_results

if __name__ == "__main__":
    main()
