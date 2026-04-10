# 🏏 Cricket Analysis Dashboard

A comprehensive Streamlit dashboard for visualizing cricket performance analytics, rankings, and Bayesian model diagnostics.

## 📋 Features

- **📊 Rankings**: View top batting/bowling rankings with multiple filtering options
- **📈 Performance Analysis**: Scatter plots showing relationships between metrics (SR vs Avg, Econ vs SR, etc.)
- **📉 Distribution Analysis**: Log-normality checks for model assumptions
- **🔍 Player Search**: Search and view individual player statistics
- **📋 Model Diagnostics**: Bayesian model convergence and validation metrics

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r dashboard_requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

## 📂 Directory Structure

The dashboard expects the following structure:
```
cricket all formats final/
├── dashboard.py                 # Main dashboard script
├── outputs/
│   ├── *.csv                   # Ranking files
│   ├── *_by_format.png         # Performance plots
│   ├── log_checks/             # Distribution analysis
│   └── diagnostics/            # Model diagnostics
```

## 🎯 Navigation

Use the sidebar to navigate between sections:
- **Rankings**: View player rankings across formats
- **Performance Analysis**: Explore metric relationships
- **Distribution Analysis**: Check log-normality assumptions
- **Player Search**: Look up individual players
- **Model Diagnostics**: Review Bayesian model validation

## 💡 Tips

- Use the search function on the Player Search tab to find specific players
- The Distribution Analysis section shows various log-transformation approaches
- Diagnostic plots show model convergence (Rhat < 1.01 is ideal)
- All plots are interactive (hover, zoom, pan)

## 📊 Data Sources

- Rankings: `top100_batting_rankings.csv`, `top100_bowling_rankings.csv`, etc.
- Performance plots: Generated from model output data
- Diagnostics: PyMC model diagnostic outputs
