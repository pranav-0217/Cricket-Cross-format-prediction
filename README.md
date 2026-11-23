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

Run every command from the repository root. The scripts read the model-ready
tables in `Data/new/processed/` and write everything to `outputs/`.

## Reproducing the analysis

```bash
# 1. Check the model-ready data
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
python EvaluateKFold.py --plot --k-check
python EvaluateAgeCurve.py
python EvaluateExposure.py

# 5. Figures and exported tables
python FormatEffects.py
python ModelDiagnostics.py
python PlayerRankings.py
python AgeCurve.py
python BuildExports.py
python PaperFigures.py
```

`PrepareData.py` built the tables in `Data/new/processed/` from the raw career
statistics and player-metadata files. The metadata files are not included, so
the processed tables are provided instead.

Step 4 is the longest stage: `EvaluateKFold.py` refits ten models, five folds
per family, so that no held-out player influences the model predicting them.
If the fold fits already exist, rerun the scoring alone with
`python EvaluateKFold.py --skip-fits --plot --k-check`. Each held-out prediction
uses nothing from the player's target-format record: debut year, age at debut
and career-midpoint age all come from the format the player has already played.

`PaperFigures.py` draws the paper's five figures at their printed size and
needs the enriched traces from step 2 and the fold predictions from step 4.

The enriched bowling model uses the longer sampling run shown above to keep all
R-hat values below 1.01. Sampling uses one core for cross-platform reliability.

## Where the paper's numbers come from

| Paper item | Script | File |
|---|---|---|
| Tables 1 and 2, country effects (94% HDI) | `PaperFigures.py` | `outputs/mixed_effects/enriched_effects_pct.csv` |
| Table 3, in-sample rows | `EvaluateBatting.py`, `EvaluateBowling.py` | `outputs/mixed_effects/batting_model_evaluation.csv`, `bowling_model_evaluation.csv` |
| Table 3, held-out rows | `EvaluateKFold.py` | `outputs/mixed_effects/lopo_kfold_evaluation.csv` |
| Player rankings | `FitBatting.py`, `FitBowling.py` | `outputs/mixed_effects/batting_player_rankings.csv`, `bowling_player_rankings.csv` |
| Results in each of the five folds | `EvaluateKFold.py` | `outputs/mixed_effects/lopo_kfold_by_fold.csv` |
| k-NN with k = 5, 20 or 30 | `EvaluateKFold.py --k-check` | `outputs/mixed_effects/knn_k_sensitivity.csv` |
| Figures 1 to 5 | `PaperFigures.py` | `outputs/paper_figures/`, named as in the paper (Figures 3 and 4 as PDF) |

`ResidualExposure.py` defines the exposure-weighted residual likelihood shared
by all four models. `TransferMethods.py` holds the transfer and scoring
functions used by `EvaluateKFold.py`; neither is run directly.

`PredictBatting.py` and `PredictBowling.py` project a single player into a
format they have not played. Neither is needed to reproduce the paper.

## Data sources

Career-format statistics, country, debut year and all player details come from ESPNcricinfo.
