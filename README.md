# Cricket Cross-Format Performance Prediction

Code for comparing and predicting cricket performance across Test, One Day
International (ODI) and Twenty20 (T20) formats. Separate multivariate Bayesian
mixed-effects models are fitted for batting and bowling: format and player
covariates enter as fixed effects, and a player-level random effect carries
latent ability shared across formats. For a player with no record in the target
format, that latent effect is borrowed from statistically similar players.

## Setup

Python 3.12.

```bash
python -m pip install -r requirements.txt
```

Run every command from the repository root. The scripts read `Data/new/raw/`
(career statistics) and `Data/new/external/` (player metadata), and write
everything to `outputs/`.

## Reproducing the analysis

```bash
# 1. Build and check the model-ready data
python PrepareData.py
python ValidateData.py

# 2. Fit the four models
python FitBatting.py          --tune 2000 --draws 2000 --chains 4 --cores 1 --target_accept 0.95
python FitBowling.py          --tune 2000 --draws 2000 --chains 4 --cores 1 --target_accept 0.95
python FitBattingEnriched.py  --tune 2000 --draws 2000 --chains 4 --cores 1 --target_accept 0.95
python FitBowlingEnriched.py  --tune 3000 --draws 3000 --chains 4 --cores 1 --target_accept 0.95

# 3. Derive format-only effects and in-sample results
python FormatOnlyEffects.py
python EvaluateBatting.py
python EvaluateBowling.py

# 4. Held-out evaluation and sensitivity checks
python EvaluateKFold.py --plot
python EvaluateAgeCurve.py
python EvaluateExposure.py

# 5. Figures and exported tables
python FormatEffects.py
python ModelDiagnostics.py
python PlayerRankings.py
python AgeCurve.py
python BuildExports.py
```

Step 4 is the longest stage: `EvaluateKFold.py` refits ten models, five folds
per family, so that no held-out player influences the model predicting them.
If the fold fits already exist, rerun the scoring alone with
`python EvaluateKFold.py --skip-fits --plot`.

The enriched bowling model uses the longer sampling run shown above to keep all
R-hat values below 1.01. Sampling uses one core for cross-platform reliability.

## Where the paper's numbers come from

| Paper item | File |
|---|---|
| Held-out evaluation table | `outputs/mixed_effects/lopo_kfold_evaluation.csv` |
| Format effects | `outputs/mixed_effects/format_effects_summary.csv` |
| Raw outcomes by format | `outputs/sr_vs_avg_by_format.png`, `outputs/bowling_econ_vs_avg_by_format.png` |
| Format-effect posteriors | `outputs/plots/format_effects_posterior.png` |
| Age spline | `outputs/plots/age_curve_simple.png` |
| Held-out predictions | `outputs/plots/cross_format_transition_kfold.png` |
| Player random effects | `outputs/plots/player_random_effects.png` |

`ResidualExposure.py` defines the exposure-weighted residual likelihood shared
by all four models. `TransferMethods.py` holds the transfer and scoring
functions used by `EvaluateKFold.py`; neither is run directly.

`PredictBatting.py` and `PredictBowling.py` project a single player into a
format they have not played. Neither is needed to reproduce the paper.

## Data sources

Career-format statistics, country and debut year were collected from
[ESPN Cricinfo](https://www.espncricinfo.com/). Player metadata are principally
ESPN-derived, with [Cricbuzz](https://www.cricbuzz.com/) as a verified fallback
for historical players with incomplete records. Both sources are publicly
accessible.
