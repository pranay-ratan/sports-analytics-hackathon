# Submission Manifest: Individual Pitcher Injury Risk Analysis

## Objective
Identify individual pitchers at the highest risk of injury for the playoffs based on their regular-season workload profiles.

## Methodology
The risk assessment uses a weighted aggregate score (0–100) based on multiple sports-medicine and workload monitoring principles:

1.  **Seasonal Volume (20%)**: Total cumulative pitch count.
2.  **High-Pitch Outings (20%)**: Counts of outings exceeding 80 and 95 pitches, which raise cumulative tissue stress.
3.  **Maximum ACWR Spike (20%)**: The highest Acute:Chronic Workload Ratio (7-day vs 28-day load). Ratios > 1.5 are flagged as danger zones for soft-tissue injuries.
4.  **Short-Rest Frequency (15%)**: Frequency of outings on less than 3 or 4 days of rest.
5.  **End-of-Season Fatigue (15%)**: Workload intensity in the final 7 and 14 days of the regular season, measuring the lack of a "recovery buffer" before the playoffs.
6.  **Peak Single-Outing Load (10%)**: The maximum stress placed on the arm in a single game.

### Analytical Rigor: Individual Consolidation
To ensure the most accurate identifies for "individual pitchers," this analysis (Version 2) deduplicates by **Pitcher Name**. Because a single individual may have multiple IDs across different organizations or levels, we consolidate all their game logs into a single seasonal profile. This prevents "diluted" risk scores that would occur if their workload was split.

## Risk Definitions
- **High (70–100)**: Multiple recurring spikes in ACWR, frequent short-rest outings, and high cumulative volume. These pitchers should be monitored closely or have limited playoff usage.
- **Moderate (40–69)**: Significant workload markers present, but showing some consistency or recovery windows.
- **Low (0–39)**: Managed workload within sustainable thresholds.

## Submission Files
- `pitcher_injury_risk_v2.py`: The complete analytical pipeline (Deduplication + Scoring + Visualization).
- `pitcher_risk_summary.csv`: The finalized dataset containing 301 unique pitcher risk profiles.
- `risk_score_distribution.png`: Visual summary of risk distribution across league levels.
- `risk_label_counts.png`: Categorical breakdown of at-risk pitchers per level.
- `workload_timeseries.png`: Comparative example of High-Risk vs Low-Risk workload profiles.

---
*Created for the SSAC Sport Med Hackathon 2026*
