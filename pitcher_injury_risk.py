#!/usr/bin/env python3
"""
Pitcher Injury Risk Analysis — BC PBL Hackathon 2026
=====================================================
Identifies pitchers at highest injury risk heading into the playoffs
based on regular-season workload profiles only.

Usage:
    python3 pitcher_injury_risk.py

Outputs (in FOLDER_PATH/outputs/):
    pitcher_risk_summary.csv
    risk_score_distribution.png
    risk_label_counts.png
    workload_timeseries.png
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe for script mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.2f}".format)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FOLDER_PATH = Path("/Users/pranayratan/Desktop/AdvancedLevel_SSAC Sport Med Hackathon 2026 copy")
OUTPUT_PATH = FOLDER_PATH / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

# Workload thresholds (Pitch Smart / sports-medicine literature)
THRESH_75       = 75    # high-effort outing marker
THRESH_80       = 80    # elevated stress marker
THRESH_95       = 95    # extreme single-game load
SHORT_REST_LT3  = 3     # days — very short rest
SHORT_REST_LT4  = 4     # days — short rest
ACWR_SPIKE      = 1.5   # acute:chronic workload ratio danger threshold

GAME_PATTERN = re.compile(r"(\d{8})-(\d+)@(\d+)-([MAR]\d+)\.csv$")

LEVEL_LABEL = {"M": "Major (M)", "A": "Advanced (A)", "R": "Rookie (R)"}
PALETTE     = {"M": "#E74C3C",   "A": "#F39C12",      "R": "#2ECC71"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_files(folder_path: Path) -> dict:
    """
    Recursively load every CSV and Excel file under folder_path.
    Returns {filepath_str: DataFrame}.
    """
    data = {}
    patterns = list(folder_path.rglob("*.csv")) + list(folder_path.rglob("*.xlsx"))
    for fp in patterns:
        try:
            df = pd.read_csv(fp) if fp.suffix == ".csv" else pd.read_excel(fp)
            data[str(fp)] = df
        except Exception as e:
            print(f"  ⚠  Could not read {fp.name}: {e}")
    print(f"Loaded {len(data)} files.")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2. PITCHER–GAME TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_pitcher_game_table(game_files: dict, player_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate pitch-by-pitch game CSVs → one row per (pitcher, game).
    Level (M/A/R), game_date, and teams are derived from the filename.
    If player_df is provided, merge the pitcher's Role into the final table.
    """
    rows = []
    for fp, df in game_files.items():
        fname = Path(fp).name
        m = GAME_PATTERN.search(fname)
        if not m:
            continue
        date_str, away_team, home_team, game_id = m.groups()
        level = game_id[0]

        if "PitcherId" not in df.columns:
            continue

        # Calculate total pitches
        grp = (
            df.groupby(["PitcherId", "Pitcher", "PitcherTeam"])
            .size()
            .reset_index(name="pitches")
        )
        
        # Calculate Advanced Biomechanical Features per pitcher in this game
        stats = []
        for pid, pdf in df.groupby("PitcherId"):
            if "Inning" in pdf.columns:
                inning_counts = pdf.groupby("Inning").size()
                high_stress_innings = (inning_counts >= 25).sum()
            else:
                high_stress_innings = 0
                
            if "RelSpeed" in pdf.columns:
                rel_speed = pd.to_numeric(pdf["RelSpeed"], errors="coerce").dropna()
                if len(rel_speed) >= 20:
                    velo_drop = rel_speed.head(10).mean() - rel_speed.tail(10).mean()
                    max_velo = rel_speed.max()
                else:
                    velo_drop = np.nan
                    max_velo = rel_speed.max() if len(rel_speed) > 0 else np.nan
            else:
                velo_drop = np.nan
                max_velo = np.nan
                
            stats.append({
                "PitcherId": pid,
                "high_stress_innings": high_stress_innings,
                "max_velo": max_velo,
                "velo_drop": velo_drop
            })
            
        stats_df = pd.DataFrame(stats)
        grp = grp.merge(stats_df, on="PitcherId", how="left")
        grp["game_id"]   = game_id
        grp["game_date"] = pd.to_datetime(date_str, format="%Y%m%d")
        grp["home_team"] = int(home_team)
        grp["away_team"] = int(away_team)
        grp["level"]     = level
        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={
        "PitcherId":   "pitcher_id",
        "Pitcher":     "pitcher_name",
        "PitcherTeam": "team",
    })
    out = out.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)

    # Days rest between consecutive outings for each pitcher
    out["prev_date"] = out.groupby("pitcher_id")["game_date"].shift(1)
    out["days_rest"] = (out["game_date"] - out["prev_date"]).dt.days

    if player_df is not None and not player_df.empty:
        pp = player_df.rename(columns={"PitcherId": "pitcher_id"})[["pitcher_id", "Role"]]
        out = out.merge(pp, on="pitcher_id", how="left")
        
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes workload metrics per pitcher. Return a single-row-per-pitcher
    summary DataFrame and the enriched game-level DataFrame.
    """
    if df.empty:
        return pd.DataFrame(), df

    # Failsafe for Jupyter notebooks where 'Role' wasn't merged in the previous step
    if "Role" not in df.columns:
        player_df = pd.read_csv(FOLDER_PATH / "player_pitching.csv")
        pp = player_df.rename(columns={"PitcherId": "pitcher_id"})[["pitcher_id", "Role"]]
        df = df.merge(pp, on="pitcher_id", how="left")

    df = df.sort_values(["pitcher_id", "game_date"]).copy()
    df["pitches"] = pd.to_numeric(df["pitches"], errors="coerce").fillna(0)

    # Single-outing flags
    df["gt_75"] = df["pitches"] > THRESH_75
    df["gt_80"] = df["pitches"] > THRESH_80
    df["gt_95"] = df["pitches"] > THRESH_95

    # Fill first-outing NaN rest with large number (not short-rest)
    df["days_rest_filled"] = df["days_rest"].fillna(999)

    # ── Acute:Chronic Workload Ratio ───────────────────────────────────────────
    # Acute  = 7-day rolling pitch sum
    # Chronic = 28-day rolling pitch sum ÷ 4 (converts to weekly average)
    # ACWR   = acute / chronic
    acwr_parts = []
    for pid, grp in df.groupby("pitcher_id"):
        g = grp.set_index("game_date")[["pitches"]].sort_index()
        g["acute_7d"]    = g["pitches"].rolling("7D",  min_periods=1).sum()
        g["chronic_28d"] = g["pitches"].rolling("28D", min_periods=1).sum() / 4
        g["acwr"]        = g["acute_7d"] / g["chronic_28d"].replace(0, np.nan)
        g["acwr"]        = g["acwr"].replace([np.inf, -np.inf], np.nan)
        
        # Cold-start fix: Blank out ACWR for the first 14 days of the pitcher's season
        g.loc[g.index < g.index.min() + pd.Timedelta(days=14), "acwr"] = np.nan
        
        g["pitcher_id"]  = pid
        acwr_parts.append(g.reset_index())

    acwr_df = pd.concat(acwr_parts, ignore_index=True)
    # Drop duplicate game_date rows in acwr_df if any doubleheaders exist to avoid inflating df
    acwr_df = acwr_df.drop_duplicates(subset=["pitcher_id", "game_date"], keep="last")
    df = df.merge(
        acwr_df[["pitcher_id", "game_date", "acute_7d", "chronic_28d", "acwr"]],
        on=["pitcher_id", "game_date"],
        how="left",
    )
    df["acwr_spike"] = df["acwr"].gt(ACWR_SPIKE)

    # ── End-of-season fatigue windows ─────────────────────────────────────────
    season_end            = df["game_date"].max()
    df["days_before_end"] = (season_end - df["game_date"]).dt.days
    df["in_last_7d"]      = df["days_before_end"] <= 7
    df["in_last_14d"]     = df["days_before_end"] <= 14
    df["bb_late"]         = (df["days_rest_filled"] <= 1) & df["in_last_14d"]

    # ── Season aggregate per pitcher ───────────────────────────────────────────
    agg = df.groupby("pitcher_id").agg(
        pitcher_name        =("pitcher_name",     "first"),
        team                =("team",              "first"),
        level               =("level",             "first"),
        Role                =("Role",              "first"),
        outings             =("game_id",           "nunique"),
        total_pitches       =("pitches",           "sum"),
        max_pitches_outing  =("pitches",           "max"),
        mean_pitches_outing =("pitches",           "mean"),
        std_pitches_outing  =("pitches",           "std"),
        gt75_outings        =("gt_75",             "sum"),
        gt80_outings        =("gt_80",             "sum"),
        gt95_outings        =("gt_95",             "sum"),
        high_stress_innings =("high_stress_innings", "sum"),
        max_game_velo       =("max_velo",          "max"),
        mean_velo_drop      =("velo_drop",         "mean"),
        short_rest_lt3      =("days_rest_filled",  lambda x: (x < SHORT_REST_LT3).sum()),
        short_rest_lt4      =("days_rest_filled",  lambda x: (x < SHORT_REST_LT4).sum()),
        mean_rest_days      =("days_rest",         "mean"),
        median_rest_days    =("days_rest",         "median"),
        max_acwr            =("acwr",              "max"),
        acwr_spikes         =("acwr_spike",        "sum"),
        end_pitches_7d      =("pitches",           lambda s: s[df.loc[s.index, "in_last_7d"]].sum()),
        end_pitches_14d     =("pitches",           lambda s: s[df.loc[s.index, "in_last_14d"]].sum()),
        end_outings_7d      =("in_last_7d",        "sum"),
        end_outings_14d     =("in_last_14d",       "sum"),
        back_to_back_late   =("bb_late",           "sum"),
    ).reset_index()

    agg["std_pitches_outing"] = agg["std_pitches_outing"].fillna(0)
    return agg, df


# ══════════════════════════════════════════════════════════════════════════════
# 4. RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; returns 0.0 if constant."""
    lo, hi = series.min(), series.max()
    if lo == hi:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_risk_scores(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a transparent 0–100 risk score per pitcher, stratified within each
    level (M / A / R) and Role (SP / RP) so pitchers are compared only against their peers.

    Component weights
    -----------------
    Seasonal volume          20 %
    High-pitch outings       20 %
    Max ACWR spike           20 %
    Short-rest frequency     15 %
    End-season workload      15 %
    Peak single-outing load  10 %

    Labels
    ------
    0–39  → Low
    40–69 → Moderate
    70+   → High
    """
    parts = []
    # Use fillna for missing roles to avoid dropping pitchers
    features["Role"] = features["Role"].fillna("Unknown")
    for (level, role), sub in features.groupby(["level", "Role"]):
        sub = sub.copy()

        vol_score        = _minmax(sub["total_pitches"])
        peak_score       = _minmax(sub["max_pitches_outing"])
        high_pitch_score = _minmax(sub["gt80_outings"] + 0.5 * sub["gt95_outings"])
        acwr_score       = _minmax(sub["max_acwr"].fillna(0))
        short_rest_score = _minmax(sub["short_rest_lt3"] + 0.5 * sub["short_rest_lt4"])
        end_score        = _minmax(sub["end_pitches_7d"] + 0.5 * sub["end_pitches_14d"])

        combined = (
            0.20 * vol_score
            + 0.20 * high_pitch_score
            + 0.20 * acwr_score
            + 0.15 * short_rest_score
            + 0.15 * end_score
            + 0.10 * peak_score
        )

        sub["risk_score_0_100"] = (combined * 100).round(1)
        sub["risk_label"] = np.select(
            [sub["risk_score_0_100"] >= 70, sub["risk_score_0_100"] >= 40],
            ["High", "Moderate"],
            default="Low",
        )
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("risk_score_0_100", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARIZE & EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def summarize_and_export(
    risk_df: pd.DataFrame,
    player_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Merge season-level stats from player_pitching.csv, select final columns,
    write pitcher_risk_summary.csv, and return the result indexed by pitcher_id.
    """
    pp = player_df.rename(columns={"PitcherId": "pitcher_id"})
    pp = pp[["pitcher_id", "IP", "ERA", "FIP", "BB", "K", "GS"]]

    out = risk_df.merge(pp, on="pitcher_id", how="left")
    out["org"] = out["pitcher_id"].str.extract(r"P(\d{2})").astype(float).astype("Int64")

    final_cols = [
        "pitcher_id", "pitcher_name", "team", "org", "level", "Role",
        "outings", "GS", "total_pitches", "IP",
        "max_pitches_outing", "mean_pitches_outing", "std_pitches_outing",
        "gt75_outings", "gt80_outings", "gt95_outings",
        "short_rest_lt3", "short_rest_lt4",
        "mean_rest_days", "median_rest_days",
        "max_acwr", "acwr_spikes",
        "high_stress_innings", "max_game_velo", "mean_velo_drop",
        "end_pitches_7d", "end_pitches_14d", "end_outings_7d", "end_outings_14d",
        "back_to_back_late",
        "ERA", "FIP",
        "risk_score_0_100", "risk_label",
    ]
    out = out[[c for c in final_cols if c in out.columns]]

    csv_path = output_path / "pitcher_risk_summary.csv"
    out.to_csv(csv_path, index=False)
    print(f"✅  Saved: {csv_path}")

    return out.set_index("pitcher_id")


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPLAIN PITCHER RISK
# ══════════════════════════════════════════════════════════════════════════════

def explain_pitcher_risk(pitcher_id: str, df: pd.DataFrame) -> str:
    """
    Return (and print) a coach-friendly narrative for a single pitcher.

    Parameters
    ----------
    pitcher_id : str   e.g. 'P18003'
    df         : pd.DataFrame indexed by pitcher_id (the summary table)
    """
    if pitcher_id not in df.index:
        msg = f"Pitcher '{pitcher_id}' not found in summary."
        print(msg)
        return msg

    r     = df.loc[pitcher_id]
    name  = r.get("pitcher_name", pitcher_id)
    team  = r.get("team", "?")
    level = r.get("level", "?")
    label = r["risk_label"]
    score = r["risk_score_0_100"]
    role  = r.get("Role", "?")
    ip    = r.get("IP", "?")

    emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(label, "⚪")
    lines = []
    lines.append(f"{emoji}  {name}  |  Team {team}  |  Level {level}  |  {label} Risk  |  Score {score}/100")
    lines.append("═" * 65)
    lines.append(f"  Role / IP         : {role} / {ip} IP")
    lines.append(f"  Season volume     : {int(r['total_pitches'])} pitches across {int(r['outings'])} outings")
    lines.append(f"  Peak outing       : {int(r['max_pitches_outing'])} pitches  "
                 f"(avg {r['mean_pitches_outing']:.1f} ± {r['std_pitches_outing']:.1f})")

    # High-pitch games
    if r["gt80_outings"] > 0:
        lines.append(
            f"  High-pitch games  : {int(r['gt80_outings'])} above 80 pitches, "
            f"{int(r['gt95_outings'])} above 95 pitches\n"
            f"                      → repeated high-effort outings raise cumulative tissue stress."
        )
    else:
        lines.append("  High-pitch games  : none above 80 pitches ✓")

    # Short rest
    if r["short_rest_lt3"] > 0:
        lines.append(
            f"  Short rest        : {int(r['short_rest_lt3'])}× on <3 days rest, "
            f"{int(r['short_rest_lt4'])}× on <4 days rest  (mean {r['mean_rest_days']:.1f}d)\n"
            f"                      → insufficient recovery amplifies injury risk."
        )
    else:
        lines.append(f"  Short rest        : no outings on <3 days rest  (mean {r['mean_rest_days']:.1f}d) ✓")

    # ACWR
    max_acwr = r.get("max_acwr", np.nan)
    if pd.notna(max_acwr):
        spike_tag = "  ⚠️  DANGER ZONE" if max_acwr > ACWR_SPIKE else "  ✓"
        lines.append(
            f"  Peak ACWR         : {max_acwr:.2f}{spike_tag}\n"
            f"                      ({int(r['acwr_spikes'])} outing(s) above the {ACWR_SPIKE} spike threshold)"
        )

    # End-of-season fatigue
    if r["end_outings_7d"] > 0:
        lines.append(
            f"  End-season push   : {int(r['end_pitches_7d'])} pitches / {int(r['end_outings_7d'])} outings in last 7 days\n"
            f"                      {int(r['end_pitches_14d'])} pitches / {int(r['end_outings_14d'])} outings in last 14 days\n"
            f"                      → reduced recovery buffer before playoffs."
        )
    else:
        lines.append("  End-season push   : no outings in final 7 days ✓")

    if r.get("back_to_back_late", 0) > 0:
        lines.append(
            f"  Back-to-back late : {int(r['back_to_back_late'])} outing(s) on ≤1 day rest in last 2 weeks\n"
            f"                      → significant cumulative fatigue risk."
        )

    lines.append("═" * 65)
    text = "\n".join(lines)
    print(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_distributions(summary: pd.DataFrame, out_path: Path) -> None:
    """Plot 1: Risk score histogram by level + Plot 2: Label stacked bar."""
    import matplotlib
    matplotlib.use("Agg")

    # ── Histogram ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pitcher Injury-Risk Score Distribution by Level",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, lvl in zip(axes, ["M", "A", "R"]):
        sub = summary[summary["level"] == lvl]["risk_score_0_100"]
        ax.hist(sub, bins=20, color=PALETTE[lvl], edgecolor="white", alpha=0.85, lw=0.6)
        ax.axvline(40, color="orange", lw=1.5, ls="--", label="Moderate (40)")
        ax.axvline(70, color="red",    lw=1.5, ls="--", label="High (70)")
        ax.set_title(LEVEL_LABEL[lvl], fontsize=12, fontweight="bold")
        ax.set_xlabel("Risk Score (0–100)")
        ax.set_ylabel("# Pitchers")
        ax.legend(fontsize=8)
        n_high = (sub >= 70).sum()
        n_mod  = ((sub >= 40) & (sub < 70)).sum()
        ax.text(0.97, 0.93, f"High={n_high}  Mod={n_mod}",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    fp = out_path / "risk_score_distribution.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")

    # ── Stacked bar ────────────────────────────────────────────────────────────
    label_counts = (
        summary.groupby(["level", "risk_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Low", "Moderate", "High"], fill_value=0)
        .reindex(["M", "A", "R"])
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    label_counts.plot(
        kind="bar", stacked=True, ax=ax2,
        color=["#2ECC71", "#F39C12", "#E74C3C"],
        edgecolor="white", linewidth=0.5,
    )
    ax2.set_xticklabels([LEVEL_LABEL[l] for l in label_counts.index], rotation=0)
    ax2.set_ylabel("Number of Pitchers")
    ax2.set_title("Risk Label Counts per Level", fontsize=13, fontweight="bold")
    ax2.legend(title="Risk Label", loc="upper right")
    for c in ax2.containers:
        ax2.bar_label(c, label_type="center", fontsize=9, color="white", fontweight="bold")
    plt.tight_layout()
    fp2 = out_path / "risk_label_counts.png"
    fig2.savefig(fp2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fp2}")


def plot_workload_timeseries(
    high_pid: str,
    low_pid: str,
    games_df: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot time series of pitch count + ACWR for a high-risk vs low-risk pitcher."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Workload Profiles: High-Risk vs Low-Risk Pitcher",
                 fontsize=14, fontweight="bold")

    for col, pid, risk_tag in [(0, high_pid, "High Risk"), (1, low_pid, "Low Risk")]:
        g = games_df[games_df["pitcher_id"] == pid].sort_values("game_date")
        if pid in summary.index:
            name  = summary.loc[pid, "pitcher_name"]
            score = summary.loc[pid, "risk_score_0_100"]
        else:
            name, score = pid, "?"
        title = f"{name} — {risk_tag} (score={score})"
        color = "#E74C3C" if col == 0 else "#2ECC71"

        # Pitch count per outing
        ax0 = axes[0, col]
        ax0.bar(g["game_date"], g["pitches"], color=color, alpha=0.75, width=1.5)
        ax0.axhline(THRESH_80, color="orange", ls="--", lw=1.2, label=f"{THRESH_80}p line")
        ax0.axhline(THRESH_95, color="red",    ls="--", lw=1.2, label=f"{THRESH_95}p line")
        ax0.set_title(title, fontweight="bold")
        ax0.set_ylabel("Pitches per Outing")
        ax0.legend(fontsize=8)
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax0.get_xticklabels(), rotation=30, ha="right")

        # ACWR over season
        ax1 = axes[1, col]
        acwr = g[["game_date", "acwr"]].dropna()
        if not acwr.empty:
            ax1.plot(acwr["game_date"], acwr["acwr"],
                     color=color, lw=2, marker="o", ms=4)
            ax1.axhline(ACWR_SPIKE, color="red", ls="--", lw=1.2,
                        label=f"ACWR={ACWR_SPIKE} threshold")
            ax1.fill_between(
                acwr["game_date"], acwr["acwr"], ACWR_SPIKE,
                where=acwr["acwr"] > ACWR_SPIKE,
                color="red", alpha=0.2, label="Spike zone",
            )
        ax1.set_ylabel("ACWR (7d / 28d chronic)")
        ax1.set_xlabel("Date")
        ax1.legend(fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fp = out_path / "workload_timeseries.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PITCHER INJURY RISK ANALYSIS — BC PBL 2026")
    print("=" * 65)

    # ── Load ───────────────────────────────────────────────────────────────────
    raw_files  = load_all_files(FOLDER_PATH)
    game_files = {k: v for k, v in raw_files.items() if GAME_PATTERN.search(k)}
    player_df  = raw_files.get(str(FOLDER_PATH / "player_pitching.csv"), pd.DataFrame())
    injury_df  = raw_files.get(str(FOLDER_PATH / "injuries.csv"),        pd.DataFrame())

    print(f"\nGame CSVs: {len(game_files)}")
    print(f"Players  : {len(player_df)}")
    print(f"Injuries : {len(injury_df)}")

    # ── Build pitcher–game table ───────────────────────────────────────────────
    print("\n[1] Building pitcher–game table...")
    pitcher_games = build_pitcher_game_table(game_files, player_df)
    print(f"    Rows: {len(pitcher_games):,}  |  Pitchers: {pitcher_games['pitcher_id'].nunique():,}")
    print(f"    Date range: {pitcher_games['game_date'].min().date()} → {pitcher_games['game_date'].max().date()}")

    # ── Feature engineering ────────────────────────────────────────────────────
    print("\n[2] Engineering features...")

    features, pitcher_games_enriched = engineer_features(pitcher_games)
    print(f"    Feature table: {features.shape}")

    # ── Risk scoring ───────────────────────────────────────────────────────────
    print("\n[3] Computing risk scores (level and role stratified)...")
    risk_df = compute_risk_scores(features)
    dist = risk_df.groupby(["level", "Role", "risk_label"]).size().unstack(fill_value=0)
    print(dist.to_string())

    # ── Export ─────────────────────────────────────────────────────────────────
    print("\n[4] Exporting summary...")
    summary = summarize_and_export(risk_df, player_df, OUTPUT_PATH)

    # ── explain_pitcher_risk demos ─────────────────────────────────────────────
    print("\n[5] Sample risk narratives (top-3 at-risk pitchers):")
    top3 = (
        summary[summary["risk_label"] != "Low"]
        .sort_values("risk_score_0_100", ascending=False)
        .head(3)
        .index
    )
    for pid in top3:
        explain_pitcher_risk(pid, summary)
        print()

    # ── Injury cross-check ─────────────────────────────────────────────────────
    if not injury_df.empty:
        print("[6] Known-injury cross-check:")
        inj   = injury_df.rename(columns={"PitcherId": "pitcher_id"})
        cross = inj.merge(
            summary[["pitcher_name", "risk_score_0_100", "risk_label"]].reset_index(),
            on="pitcher_id", how="left",
        )
        flagged = cross[cross["risk_label"].isin(["Moderate", "High"])]
        print(cross[["pitcher_id", "Name", "Date", "PitchesThisGame",
                      "InjuryDays", "risk_score_0_100", "risk_label"]].to_string(index=False))
        print(f"\n  {len(flagged)} / {len(cross)} known-injured pitchers flagged Moderate/High.")
        print("\n  📝 NOTE: All 6 injuries occurred DURING the regular season. Injured pitchers")
        print("  subsequently missed games, reducing their end-of-season volume totals.")
        print("  Our model targets PRE-PLAYOFF risk (end-of-season cumulative fatigue),")
        print("  not in-season event prediction — these are different questions.")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n[7] Generating plots...")
    plot_risk_distributions(summary.reset_index(), OUTPUT_PATH)

    game_pids = set(pitcher_games_enriched["pitcher_id"].unique())
    summary_sub = summary[summary.index.isin(game_pids)]
    high_cands = summary_sub[summary_sub["risk_label"] == "High"].sort_values("risk_score_0_100", ascending=False)
    mod_cands  = summary_sub[summary_sub["risk_label"] == "Moderate"].sort_values("risk_score_0_100", ascending=False)
    low_cands  = summary_sub[summary_sub["risk_label"] == "Low"].sort_values("risk_score_0_100")

    high_pid = (high_cands if len(high_cands) else mod_cands).index[0]
    low_pid  = low_cands.index[0]
    plot_workload_timeseries(high_pid, low_pid, pitcher_games_enriched, summary, OUTPUT_PATH)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL RESULTS".center(65))
    print("=" * 65)
    for lvl in ["M", "A", "R"]:
        sub  = summary[summary["level"] == lvl]
        high = (sub["risk_label"] == "High").sum()
        mod  = (sub["risk_label"] == "Moderate").sum()
        low  = (sub["risk_label"] == "Low").sum()
        print(f"  {LEVEL_LABEL[lvl]:20s} → 🔴 High: {high:3d}  🟡 Moderate: {mod:3d}  🟢 Low: {low:3d}")
    print(f"\n  Output folder: {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
