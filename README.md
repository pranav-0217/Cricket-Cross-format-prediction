# 🏏 Cricket Cross-Format Prediction

A Bayesian statistical analysis system that fairly compares cricket player performance across Test, ODI, and T20 formats using multivariate mixed-effects models.

## 🎯 Project Overview

- **Problem**: How to rank cricket players fairly across formats with different intensities?
- **Solution**: Bayesian mixed-effects model separates format effects from player ability
- **Key Finding**: T20 has +134% higher SR but -27% lower avg vs Test

## 📊 Features

- ✅ Multivariate Bayesian mixed-effects modeling (PyMC)
- ✅ Format transferability analysis (% changes between formats)  
- ✅ Player rankings across formats
- ✅ Interactive Streamlit dashboard (8 insight tabs)
- ✅ Model diagnostics & convergence validation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
# Data preparation
python prepare_joint_batting_data.py
python prepare_joint_bowling_data.py
python prepare_batting_data_enriched.py
python prepare_bowling_data_enriched.py

# Fit Bayesian models
python fit_joint_batting_pymc.py
python fit_joint_bowling_pymc.py

# Evaluate & extract insights
python evaluate_batting_model.py
python analyse_format_effects.py
python plot_model_diagnostics.py
```

### 3. View Dashboard
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

## 📁 Project Structure

```
cricket-cross-format/
├── *.py                    # 15 processing & modeling scripts
├── *.ipynb                 # 3 exploratory notebooks (EDA, cleaning, prep)
├── dashboard.py            # Interactive dashboard
├── Data/                   # Raw & processed data (git-ignored)
├── outputs/                # Model results & visualizations (git-ignored)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📊 Dashboard Sections

1. **Rankings** - Top 100 batting/bowling players
2. **Performance Analysis** - SR vs Avg by format
3. **Distribution Analysis** - Log-normality checks
4. **🔍 Key Findings** - 8 insight tabs:
   - Handedness Analysis
   - Playing Role Analysis
   - Bowling Style Analysis
   - Country Performance
   - Age & Experience
   - **Format Transferability** ⭐
   - Conversion Rates
   - Career Status
5. **Player Search** - Individual player lookup
6. **Model Diagnostics** - Bayesian validity checks

## 🔬 Model Details

**Multivariate Mixed-Effects Bayesian Model:**
```
y_ij = X_ij * β + Z_ij * ψ_i + ε_ij

y   = [log(avg), log(sr)]              [Outcomes]
X   = [format_ODI, format_T20]         [Fixed effects]
Z   = player indicator                 [Random effects]
ψ ~ MVN(0, Σ_u)                       [Player latent ability]
ε ~ MVN(0, Σ_eps)                     [Residuals]
```

## 📈 Key Insights

| Format | SR Change | Avg Change |
|--------|-----------|-----------|
| T20 vs Test | +134% | -27% |
| ODI vs Test | +42% | -9% |

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **Statistical Model**: PyMC (Bayesian inference)
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Streamlit
- **Model Diagnostics**: ArviZ

## 📚 Data Sources

Cricket statistics scraped from publicly available sources:
- Test, ODI, T20 career statistics
- Player metadata (age, country, role, handedness)
- Career span information

## 📝 Files Guide

| File | Purpose |
|------|---------|
| `prepare_joint_*.py` | Data pipeline |
| `fit_joint_*_pymc*.py` | Core Bayesian models |
| `evaluate_*.py` | Model validation |
| `analyse_format_effects.py` | Format insights |
| `predict_cross_format*.py` | Predictions |
| `plot_*.py` | Visualizations |
| `dashboard.py` | Interactive dashboard |
| `*.ipynb` | Exploratory analysis |

## 👤 Author

Pranav

## 📄 License

MIT
