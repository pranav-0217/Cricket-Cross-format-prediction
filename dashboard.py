import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(page_title="Cricket Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .title-main {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title-main'>🏏 Cricket Performance Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# Data loading
@st.cache_data
def load_data():
    output_dir = Path("outputs")
    data = {}

    # Load CSVs
    ranking_files = {
        "all_players": "all_players_ranking.csv",
        "top100_batting": "top100_batting_rankings.csv",
        "top100_bowling": "top100_bowling_rankings.csv",
        "batting_1000runs": "top_batting_rankings_1000runs.csv"
    }

    for key, filename in ranking_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)

    # Load images from log_checks
    log_checks_dir = output_dir / "log_checks"
    diagnostics_dir = output_dir / "diagnostics"

    data['log_checks_dir'] = log_checks_dir
    data['diagnostics_dir'] = diagnostics_dir

    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["📊 Rankings", "📈 Performance Analysis", "📉 Distribution Analysis", "🔍 Player Search", "📋 Model Diagnostics"]
)

# ============= RANKINGS PAGE =============
if page == "📊 Rankings":
    st.header("Player Rankings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏏 Batting Rankings")
        ranking_type = st.selectbox(
            "Select Ranking Type (Batting):",
            ["All Players", "Top 100", "1000+ Runs"]
        )

        if ranking_type == "All Players" and "all_players" in data:
            df = data["all_players"].head(50)
            st.dataframe(df, use_container_width=True)
        elif ranking_type == "Top 100" and "top100_batting" in data:
            df = data["top100_batting"]
            st.dataframe(df, use_container_width=True)
        elif ranking_type == "1000+ Runs" and "batting_1000runs" in data:
            df = data["batting_1000runs"]
            st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("🎯 Bowling Rankings")
        if "top100_bowling" in data:
            df = data["top100_bowling"]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Bowling rankings data not available")

# ============= PERFORMANCE ANALYSIS PAGE =============
elif page == "📈 Performance Analysis":
    st.header("Performance Metrics by Format")

    output_dir = Path("outputs")

    col1, col2 = st.columns(2)

    with col1:
        # SR vs Avg
        sr_avg_path = output_dir / "sr_vs_avg_by_format.png"
        if sr_avg_path.exists():
            st.subheader("Strike Rate vs Average (Batting)")
            st.image(str(sr_avg_path), use_container_width=True)

        # Bowling Econ vs Avg
        econ_avg_path = output_dir / "bowling_econ_vs_avg_by_format.png"
        if econ_avg_path.exists():
            st.subheader("Economy vs Average (Bowling)")
            st.image(str(econ_avg_path), use_container_width=True)

    with col2:
        # SR vs Econ
        sr_econ_path = output_dir / "bowling_sr_vs_econ_by_format.png"
        if sr_econ_path.exists():
            st.subheader("Strike Rate vs Economy (Bowling)")
            st.image(str(sr_econ_path), use_container_width=True)

# ============= DISTRIBUTION ANALYSIS PAGE =============
elif page == "📉 Distribution Analysis":
    st.header("Log-Normality Distribution Checks")
    st.info("These visualizations show whether log-transforming the metrics improves normality for the statistical model")

    log_checks_dir = data.get('log_checks_dir')

    if log_checks_dir and log_checks_dir.exists():
        metrics = ["batting_avg", "batting_sr", "bowling_econ", "bowling_sr"]

        for metric in metrics:
            st.subheader(f"{metric.replace('_', ' ').title()} - Log Normality Analysis")

            # Find related files
            metric_files = {}
            for file in log_checks_dir.glob(f"{metric}*"):
                if "hist" in file.name:
                    if "eps" in file.name:
                        eps_val = file.name.split("eps_")[1].split("_")[0]
                        key = f"hist_eps_{eps_val}"
                    else:
                        key = f"hist_{file.name.split('_')[-2]}"
                    metric_files[key] = file
                elif "qq" in file.name:
                    if "eps" in file.name:
                        eps_val = file.name.split("eps_")[1].split("_")[0]
                        key = f"qq_eps_{eps_val}"
                    else:
                        key = f"qq_{file.name.split('_')[-2]}"
                    metric_files[key] = file

            if metric_files:
                cols = st.columns(len(metric_files))
                for idx, (file_type, filepath) in enumerate(sorted(metric_files.items())):
                    with cols[idx]:
                        st.caption(file_type)
                        st.image(str(filepath), use_container_width=True)
            else:
                st.warning(f"No distribution checks found for {metric}")

            st.markdown("---")
    else:
        st.warning("Log checks directory not found")

# ============= PLAYER SEARCH PAGE =============
elif page == "🔍 Player Search":
    st.header("Player Performance Lookup")

    if "all_players" in data:
        df = data["all_players"]

        col1, col2 = st.columns(2)

        with col1:
            player_name = st.text_input("Search for a player:")

        with col2:
            metric = st.selectbox("Sort by:", df.columns[1:] if len(df.columns) > 1 else ["Name"])

        if player_name:
            filtered = df[df.iloc[:, 0].str.contains(player_name, case=False, na=False)]
            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True)

                # Display stats
                if len(filtered) == 1:
                    player_data = filtered.iloc[0]
                    cols = st.columns(len(player_data))
                    for idx, (col, val) in enumerate(player_data.items()):
                        with cols[idx]:
                            st.metric(col, val)
            else:
                st.warning("Player not found")
        else:
            # Show top players
            st.write("Top 10 Players:")
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Player data not available")

# ============= DIAGNOSTICS PAGE =============
elif page == "📋 Model Diagnostics":
    st.header("Model Diagnostics & Validation")
    st.info("Bayesian model convergence and posterior validation checks")

    diagnostics_dir = data.get('diagnostics_dir')

    if diagnostics_dir and diagnostics_dir.exists():
        models = ["batting", "bowling"]

        for model in models:
            st.subheader(f"{model.title()} Model Diagnostics")

            diagnostic_types = ["trace", "rhat", "posteriors", "ppc", "player_corr"]
            cols = st.columns(len(diagnostic_types))

            for idx, diag_type in enumerate(diagnostic_types):
                filepath = diagnostics_dir / f"{model}_{diag_type}.png"
                if filepath.exists():
                    with cols[idx]:
                        st.caption(diag_type.upper())
                        st.image(str(filepath), use_container_width=True)

            st.markdown("---")
    else:
        st.warning("Diagnostics directory not found")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>Cricket Performance Analytics Dashboard | Mixed-Effects Bayesian Model</small>
    </div>
    """,
    unsafe_allow_html=True
)
