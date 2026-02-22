"""
AARHI War-Room Dashboard

Interactive Streamlit dashboard for visualizing Aadhaar resilience hotspots,
trends, and recommendations for UIDAI administrators.

This frontend is read-only and displays precomputed analytics.
No ML or computation logic is performed here.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import subprocess
import sys


# =============================================================================
# PROJECT ROOT & DATA PIPELINE
# =============================================================================

def get_project_root() -> Path:
    """Dynamically detect the project root directory."""
    # Start from the current file's directory and go up to find the project root
    current_file = Path(__file__).resolve()
    # frontend/app.py -> go up one level to reach project root
    return current_file.parent.parent


def get_required_data_files() -> list:
    """Return list of required processed data files."""
    return [
        "hotspot_clusters.csv",
        "recommendations.csv",
        "ari_scored_districts.csv",
        "district_merged_metrics.csv"
    ]


def check_data_files_exist(project_root: Path) -> tuple:
    """
    Check if all required processed data files exist.
    
    Returns:
        tuple: (all_exist: bool, missing_files: list)
    """
    processed_dir = project_root / "data" / "processed"
    required_files = get_required_data_files()
    missing_files = []
    
    for filename in required_files:
        filepath = processed_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files


def run_data_pipeline(project_root: Path) -> tuple:
    """
    Run the data pipeline scripts in order.
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    pipeline_scripts = [
        project_root / "engine" / "data_prep.py",
        project_root / "engine" / "ari.py",
        project_root / "engine" / "clustering.py",
        project_root / "engine" / "recommendations.py"
    ]
    
    for script in pipeline_scripts:
        if not script.exists():
            return False, f"Pipeline script not found: {script.name}"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Script {script.name} failed:\n{e.stderr or e.stdout}"
            return False, error_msg
        except Exception as e:
            return False, f"Error running {script.name}: {str(e)}"
    
    return True, None


def initialize_data_pipeline():
    """
    Check for required data files and run pipeline if needed.
    This runs only once per session using session state.
    """
    # Skip if already initialized this session
    if st.session_state.get("_pipeline_initialized", False):
        return
    
    project_root = get_project_root()
    files_exist, missing_files = check_data_files_exist(project_root)
    
    if files_exist:
        st.session_state["_pipeline_initialized"] = True
        return
    
    # Show pipeline execution UI
    st.info(f"🔍 Missing data files detected: {', '.join(missing_files)}")
    
    with st.status("Running data pipeline...", expanded=True) as status:
        st.write("📦 Preparing data...")
        st.write("Running: engine/data_prep.py")
        st.write("Running: engine/ari.py")
        st.write("Running: engine/clustering.py")
        st.write("Running: engine/recommendations.py")
        
        success, error_msg = run_data_pipeline(project_root)
        
        if success:
            status.update(label="Data pipeline completed successfully.", state="complete", expanded=False)
            st.session_state["_pipeline_initialized"] = True
            st.rerun()
        else:
            status.update(label="Data pipeline failed.", state="error", expanded=True)
            st.error(f"❌ Pipeline Error:\n{error_msg}")
            st.stop()


# =============================================================================
# THEME CONFIGURATION
# =============================================================================

LIGHT_THEME = {
    "base": "light",
    "primaryColor": "#1F4ED8",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F5F7FB",
    "textColor": "#0F172A",
    "plotly_template": "plotly_white"
}

DARK_THEME = {
    "base": "dark",
    "primaryColor": "#60A5FA",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#1E222A",
    "textColor": "#FAFAFA",
    "plotly_template": "plotly_dark"
}


def init_theme():
    """Initialize theme in session state."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"


def get_current_theme() -> dict:
    """Get the current theme configuration."""
    if st.session_state.get("theme", "Light") == "Dark":
        return DARK_THEME
    return LIGHT_THEME


def apply_theme():
    """Apply theme settings using Streamlit's internal config."""
    theme_config = get_current_theme()
    
    # Apply theme via internal config API
    try:
        st._config.set_option("theme.base", theme_config["base"])
        st._config.set_option("theme.primaryColor", theme_config["primaryColor"])
        st._config.set_option("theme.backgroundColor", theme_config["backgroundColor"])
        st._config.set_option("theme.secondaryBackgroundColor", theme_config["secondaryBackgroundColor"])
        st._config.set_option("theme.textColor", theme_config["textColor"])
    except Exception:
        # Fallback if internal API is not available
        pass


# Initialize theme before page config
init_theme()
apply_theme()


# =============================================================================
# GLOBAL COLOR PALETTE - High-contrast government-style colors
# =============================================================================

COLOR_MAP = {
    "Red": "#D32F2F",      # High Stress - Vibrant Red
    "Yellow": "#F9A825",   # Moderate Stress - Amber/Gold
    "Green": "#2E7D32"     # Stable - Deep Green
}

# =============================================================================
# INDIAN STATE CENTROIDS (Latitude, Longitude)
# =============================================================================

STATE_CENTROIDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.1809, 73.0169),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Lakshadweep": (10.5667, 72.6417),
    "Puducherry": (11.9416, 79.8083),
    # Common variations and aliases
    "Andaman & Nicobar Islands": (11.7401, 92.6586),
    "Andaman And Nicobar Islands": (11.7401, 92.6586),
    "Dadra & Nagar Haveli": (20.1809, 73.0169),
    "Daman & Diu": (20.4283, 72.8397),
    "Jammu & Kashmir": (33.7782, 76.5762),
    "NCT of Delhi": (28.7041, 77.1025),
    "Nct Of Delhi": (28.7041, 77.1025),
}


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AARHI War-Room Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_hotspot_clusters() -> pd.DataFrame:
    """Load hotspot cluster data."""
    filepath = Path("data/processed/hotspot_clusters.csv")
    return pd.read_csv(filepath)


@st.cache_data
def load_ari_scores() -> pd.DataFrame:
    """Load ARI-scored district data."""
    filepath = Path("data/processed/ari_scored_districts.csv")
    return pd.read_csv(filepath)


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    """Load recommendations data."""
    filepath = Path("data/processed/recommendations.csv")
    return pd.read_csv(filepath)


@st.cache_data
def load_merged_metrics() -> pd.DataFrame:
    """Load merged district metrics with date-level data."""
    filepath = Path("data/processed/district_merged_metrics.csv")
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_data() -> dict:
    """Load all required datasets."""
    return {
        "hotspots": load_hotspot_clusters(),
        "ari_scores": load_ari_scores(),
        "recommendations": load_recommendations(),
        "metrics": load_merged_metrics()
    }


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def get_state_list(df: pd.DataFrame) -> list:
    """Get sorted list of unique states with 'All States' option."""
    states = sorted(df["state"].unique().tolist())
    return ["All States"] + states


def filter_by_state(df: pd.DataFrame, selected_state: str) -> pd.DataFrame:
    """Filter DataFrame by selected state."""
    if selected_state == "All States":
        return df
    return df[df["state"] == selected_state].copy()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def aggregate_state_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate district-level data to state level.
    
    Computes:
    - Average ARI score per state
    - Dominant stress level (mode of hotspot_label)
    - District counts per stress level
    
    Args:
        df: DataFrame with district-level hotspot data
        
    Returns:
        DataFrame aggregated at state level
    """
    # Aggregate metrics by state
    state_agg = df.groupby("state").agg(
        avg_ari=("ari_score", "mean"),
        total_districts=("district", "count"),
        red_districts=("hotspot_label", lambda x: (x == "Red").sum()),
        yellow_districts=("hotspot_label", lambda x: (x == "Yellow").sum()),
        green_districts=("hotspot_label", lambda x: (x == "Green").sum())
    ).reset_index()
    
    # Determine dominant stress level (mode)
    def get_dominant_label(row):
        counts = {
            "Red": row["red_districts"],
            "Yellow": row["yellow_districts"],
            "Green": row["green_districts"]
        }
        # If there are any Red districts, prioritize Red
        if counts["Red"] > 0:
            return "Red"
        # If there are any Yellow districts, prioritize Yellow
        elif counts["Yellow"] > 0:
            return "Yellow"
        else:
            return "Green"
    
    state_agg["dominant_stress"] = state_agg.apply(get_dominant_label, axis=1)
    
    return state_agg


def create_state_scatter_map(df: pd.DataFrame) -> go.Figure:
    """
    Create a Mapbox scatter map of India showing state-level resilience.
    
    Each marker represents one state, positioned at its centroid.
    - Color indicates dominant stress level (Red/Yellow/Green)
    - Size indicates average ARI score (larger = higher ARI = more stable)
    
    Args:
        df: DataFrame with district-level hotspot data
        
    Returns:
        Plotly Figure with Mapbox scatter map
    """
    # Aggregate to state level
    state_data = aggregate_state_level(df)
    
    # Add latitude and longitude from centroids lookup
    state_data["lat"] = state_data["state"].apply(
        lambda s: STATE_CENTROIDS.get(s, (20.5937, 78.9629))[0]
    )
    state_data["lon"] = state_data["state"].apply(
        lambda s: STATE_CENTROIDS.get(s, (20.5937, 78.9629))[1]
    )
    
    # Create stress level labels for display
    stress_labels = {
        "Red": "High Stress",
        "Yellow": "Moderate",
        "Green": "Stable"
    }
    state_data["stress_label"] = state_data["dominant_stress"].map(stress_labels)
    
    # Marker size based on average ARI score (scaled for visibility)
    # Higher ARI = larger marker (more stable areas are more prominent)
    state_data["marker_size"] = state_data["avg_ari"].clip(lower=10)
    
    # Create scatter mapbox
    fig = px.scatter_mapbox(
        state_data,
        lat="lat",
        lon="lon",
        color="dominant_stress",
        color_discrete_map=COLOR_MAP,
        size="marker_size",
        size_max=35,
        hover_name="state",
        hover_data={
            "lat": False,
            "lon": False,
            "marker_size": False,
            "dominant_stress": False,
            "stress_label": True,
            "avg_ari": ":.1f",
            "total_districts": True
        },
        labels={
            "stress_label": "Status",
            "avg_ari": "Avg ARI Score",
            "total_districts": "Total Districts"
        },
        category_orders={"dominant_stress": ["Red", "Yellow", "Green"]}
    )
    
    # Update layout for India-centered view (MANDATORY settings)
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=22.5, lon=78.9),
            zoom=3.7
        ),
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(
            title="Stress Level",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Clean hover template showing key info
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Status: %{customdata[0]}<br>" +
            "Avg ARI: %{customdata[1]:.1f}<br>" +
            "Districts: %{customdata[2]}" +
            "<extra></extra>"
        )
    )
    
    return fig


def create_state_choropleth(df: pd.DataFrame) -> go.Figure:
    """
    Create a state-level choropleth map of India showing resilience patterns.
    
    Colors states by dominant stress level:
    - Red: High Stress (has districts with high stress indicators)
    - Yellow: Moderate Stress
    - Green: Stable
    
    Args:
        df: DataFrame with district-level hotspot data
        
    Returns:
        Plotly Figure with India choropleth map
    """
    # Aggregate to state level
    state_data = aggregate_state_level(df)
    
    # Use global color palette for stress levels
    color_map = COLOR_MAP
    
    # Map colors to states
    state_data["color"] = state_data["dominant_stress"].map(color_map)
    
    # Create the choropleth using a bar chart as geographic visualization
    # Since Plotly's choropleth requires specific geo data for Indian states,
    # we'll create a treemap visualization that effectively shows state-level data
    
    # Sort by stress priority then by ARI
    priority_map = {"Red": 0, "Yellow": 1, "Green": 2}
    state_data["_priority"] = state_data["dominant_stress"].map(priority_map)
    state_data = state_data.sort_values(["_priority", "avg_ari"], ascending=[True, True])
    
    # Create a treemap for better geographic-style visualization
    fig = px.treemap(
        state_data,
        path=["dominant_stress", "state"],
        values="total_districts",
        color="dominant_stress",
        color_discrete_map=color_map,
        hover_data={
            "avg_ari": ":.2f",
            "total_districts": True,
            "red_districts": True,
            "yellow_districts": True,
            "green_districts": True
        },
        custom_data=["avg_ari", "total_districts", "red_districts", "yellow_districts", "green_districts"]
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Average ARI: %{customdata[0]:.1f}<br>" +
            "Total Districts: %{customdata[1]}<br>" +
            "High Stress: %{customdata[2]} | Moderate: %{customdata[3]} | Stable: %{customdata[4]}" +
            "<extra></extra>"
        )
    )
    
    fig.update_layout(
        title="State-Level Aadhaar Resilience Map",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        template=get_current_theme()["plotly_template"]
    )
    
    return fig


def create_state_bar_map(df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart showing state-level ARI scores.
    
    This provides a clean, sortable view of state resilience levels
    as an alternative to geographic mapping.
    
    Args:
        df: DataFrame with district-level hotspot data
        
    Returns:
        Plotly Figure with state-level bar chart
    """
    # Aggregate to state level
    state_data = aggregate_state_level(df)
    
    # Sort by average ARI (ascending - high stress first)
    state_data = state_data.sort_values("avg_ari", ascending=True)
    
    # Use global color palette
    colors = state_data["dominant_stress"].map(COLOR_MAP).tolist()
    
    fig = go.Figure(data=[
        go.Bar(
            y=state_data["state"],
            x=state_data["avg_ari"],
            orientation="h",
            marker_color=colors,
            text=state_data["avg_ari"].round(1),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Average ARI: %{x:.1f}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title="State-Level Average ARI Scores",
        xaxis_title="Average ARI Score",
        yaxis_title="State",
        height=max(400, len(state_data) * 25),
        xaxis={"range": [0, 105]},
        margin=dict(l=20, r=60, t=50, b=20),
        yaxis={"categoryorder": "total ascending"},
        template=get_current_theme()["plotly_template"]
    )
    
    # Add threshold lines with matching palette colors
    fig.add_vline(x=70, line_dash="dash", line_color=COLOR_MAP["Green"],
                  annotation_text="Stable (70)")
    fig.add_vline(x=40, line_dash="dash", line_color=COLOR_MAP["Yellow"],
                  annotation_text="Moderate (40)")
    
    return fig


def create_ari_bar_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """
    Create a bar chart comparing districts by ARI score.
    
    Sorted from high stress (low ARI) to stable (high ARI).
    """
    # Sort by ARI score ascending (high stress first)
    df_sorted = df.sort_values("ari_score", ascending=True).head(top_n)
    
    # Use global color palette
    colors = df_sorted["hotspot_label"].map(COLOR_MAP).tolist()
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted["district"] + " (" + df_sorted["state"].str[:3] + ")",
            y=df_sorted["ari_score"],
            marker_color=colors,
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "ARI Score: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=f"District ARI Scores (Top {top_n} by Stress Level)",
        xaxis_title="District",
        yaxis_title="ARI Score (0-100)",
        height=400,
        xaxis={"tickangle": 45},
        yaxis={"range": [0, 105]},
        margin=dict(l=20, r=20, t=50, b=100),
        template=get_current_theme()["plotly_template"]
    )
    
    # Add reference lines for categories with matching palette colors
    fig.add_hline(y=70, line_dash="dash", line_color=COLOR_MAP["Green"], 
                  annotation_text="Stable Threshold (70)")
    fig.add_hline(y=40, line_dash="dash", line_color=COLOR_MAP["Yellow"],
                  annotation_text="Moderate Threshold (40)")
    
    return fig


def create_trend_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing trends over time.
    
    Shows total enrolments, biometric updates, and demographic updates.
    """
    # Aggregate by date
    daily_metrics = df.groupby("date").agg({
        "total_enrolments": "sum",
        "total_biometric_updates": "sum",
        "total_demographic_updates": "sum"
    }).reset_index()
    
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=daily_metrics["date"],
        y=daily_metrics["total_enrolments"],
        mode="lines+markers",
        name="Enrolments",
        line=dict(color="#3498db", width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_metrics["date"],
        y=daily_metrics["total_biometric_updates"],
        mode="lines+markers",
        name="Biometric Updates",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_metrics["date"],
        y=daily_metrics["total_demographic_updates"],
        mode="lines+markers",
        name="Demographic Updates",
        line=dict(color="#f39c12", width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Operational Activity Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode="x unified",
        template=get_current_theme()["plotly_template"]
    )
    
    return fig


def create_summary_metrics(df: pd.DataFrame) -> dict:
    """Calculate summary metrics for display."""
    total_districts = len(df)
    red_count = (df["hotspot_label"] == "Red").sum()
    yellow_count = (df["hotspot_label"] == "Yellow").sum()
    green_count = (df["hotspot_label"] == "Green").sum()
    avg_ari = df["ari_score"].mean()
    
    return {
        "total_districts": total_districts,
        "red_count": red_count,
        "yellow_count": yellow_count,
        "green_count": green_count,
        "avg_ari": avg_ari
    }


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_header():
    """Display dashboard header with title and subtitle."""
    st.title("🛡️ AARHI War-Room Dashboard")
    st.caption("Real-time monitoring of district-level Aadhaar operational resilience across India")


def display_sidebar_filters(data: dict) -> str:
    """Display sidebar filters and return selected state."""
    st.sidebar.header("🔍 Filters")
    
    states = get_state_list(data["hotspots"])
    selected_state = st.sidebar.selectbox(
        "Select State",
        options=states,
        index=0,
        help="Filter all views by state"
    )
    
    st.sidebar.divider()
    
    # Theme toggle with icons
    st.sidebar.markdown("### 🎨 Theme")
    theme_options = ["🌞 Light", "🌙 Dark"]
    current_theme = st.session_state.get("theme", "Light")
    current_theme_index = 0 if current_theme == "Light" else 1
    
    selected_theme_display = st.sidebar.radio(
        "Select Theme",
        options=theme_options,
        index=current_theme_index,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Extract theme name from selection (remove emoji)
    selected_theme = "Light" if "Light" in selected_theme_display else "Dark"
    
    # Handle theme change
    if selected_theme != st.session_state.get("theme", "Light"):
        st.session_state.theme = selected_theme
        apply_theme()
        st.rerun()
    
    # Visual theme indicator badge
    if current_theme == "Light":
        st.sidebar.caption("🌞 Light Mode Active")
    else:
        st.sidebar.caption("🌙 Dark Mode Active")
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Legend")
    st.sidebar.markdown("""
    - 🔴 **Red**: High Stress
    - 🟡 **Yellow**: Moderate Stress
    - 🟢 **Green**: Stable
    """)
    
    st.sidebar.divider()
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This dashboard displays precomputed 
    resilience indicators. Update intensity 
    is used as a proxy for operational stress.
    """)
    
    return selected_state


def display_summary_cards(metrics: dict):
    """Display summary metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Districts",
            value=f"{metrics['total_districts']:,}"
        )
    
    with col2:
        st.metric(
            label="🔴 High Stress",
            value=f"{metrics['red_count']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🟡 Moderate Stress",
            value=f"{metrics['yellow_count']:,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🟢 Stable",
            value=f"{metrics['green_count']:,}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="Avg. ARI Score",
            value=f"{metrics['avg_ari']:.1f}"
        )


def display_map_section(df: pd.DataFrame):
    """Display the state-level resilience map section in a bordered card."""
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📊 Treemap View", "📈 Bar Chart View"])
    
    with tab1:
        fig_map = create_state_scatter_map(df)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        fig_treemap = create_state_choropleth(df)
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    with tab3:
        fig_bar = create_state_bar_map(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.caption("**Color** = Stress Level (🔴 High | 🟡 Moderate | 🟢 Stable) | **Size** = Average ARI Score")


def display_bar_chart_section(df: pd.DataFrame):
    """Display the ARI comparison bar chart section in a bordered card."""
    # Slider for number of districts to show
    n_districts = st.slider(
        "Districts to display",
        min_value=10,
        max_value=min(100, len(df)),
        value=min(30, len(df)),
        step=5,
        label_visibility="collapsed"
    )
    
    fig = create_ari_bar_chart(df, top_n=n_districts)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower ARI scores indicate higher operational stress intensity.")


def display_trend_section(df: pd.DataFrame):
    """Display the trend line chart section in a bordered card."""
    fig = create_trend_line_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shows enrolment and update volumes over the observation period.")


def render_stress_badge(level: str) -> str:
    """
    Render a stress level as a professional pill-style HTML badge.
    
    Args:
        level: Stress level string ("Red", "Yellow", or "Green")
        
    Returns:
        HTML string for styled badge pill
    """
    # Common base styles for all badges
    base_style = (
        "display:inline-block;"
        "padding:4px 10px;"
        "border-radius:999px;"
        "font-weight:600;"
        "font-size:12px;"
    )
    
    # Badge-specific styles (background, text color, border)
    badges = {
        "Red": (
            f"<span style='{base_style}"
            "background:#FDECEA;"
            "color:#B71C1C;"
            "border:1px solid #F5C6CB;'>"
            "HIGH</span>"
        ),
        "Yellow": (
            f"<span style='{base_style}"
            "background:#FFF4E5;"
            "color:#8A6D1D;"
            "border:1px solid #FFE0A3;'>"
            "MODERATE</span>"
        ),
        "Green": (
            f"<span style='{base_style}"
            "background:#E8F5E9;"
            "color:#1B5E20;"
            "border:1px solid #C8E6C9;'>"
            "STABLE</span>"
        ),
    }
    return badges.get(level, level)


def display_recommendations_section(df: pd.DataFrame):
    """Display the recommendations table section in a bordered card."""
    # Sort by hotspot priority
    priority_map = {"Red": 0, "Yellow": 1, "Green": 2}
    df_sorted = df.copy()
    df_sorted["_priority"] = df_sorted["hotspot_label"].map(priority_map)
    df_sorted = df_sorted.sort_values("_priority").drop(columns=["_priority"])
    
    # Create styled stress level badges
    df_display = df_sorted.copy()
    df_display["Stress Level"] = df_display["hotspot_label"].apply(render_stress_badge)
    
    # Prepare display dataframe with renamed columns
    df_display = df_display.rename(columns={
        "state": "State",
        "district": "District",
        "recommendation": "Recommendation",
        "basis": "Basis"
    })
    
    # Select and order columns for display
    display_cols = ["State", "District", "Stress Level", "Recommendation", "Basis"]
    df_display = df_display[display_cols]
    
    # Render table with HTML badges
    table_html = df_display.to_html(escape=False, index=False, classes="recommendations-table")
    
    # Add custom CSS for table styling
    table_css = """
    <style>
        .recommendations-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .recommendations-table th {
            background-color: #1F4ED8;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }
        .recommendations-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }
        .recommendations-table tr:hover {
            background-color: rgba(31, 78, 216, 0.05);
        }
    </style>
    """
    
    # Create scrollable container
    st.markdown(table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div style="max-height: 350px; overflow-y: auto;">{table_html}</div>',
        unsafe_allow_html=True
    )
    st.caption("Sorted by priority: High Stress districts first.")
    
    # Download button (use original data without HTML)
    csv = df_sorted.to_csv(index=False)
    st.download_button(
        label="📥 Download Recommendations (CSV)",
        data=csv,
        file_name="aarhi_recommendations.csv",
        mime="text/csv"
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    try:
        # Load all data
        data = load_all_data()
        
        # Display sidebar and get filter selection
        selected_state = display_sidebar_filters(data)
        
        # Apply state filter to all datasets
        filtered_hotspots = filter_by_state(data["hotspots"], selected_state)
        filtered_ari = filter_by_state(data["ari_scores"], selected_state)
        filtered_recommendations = filter_by_state(data["recommendations"], selected_state)
        filtered_metrics = filter_by_state(data["metrics"], selected_state)
        
        # =====================================================================
        # ROW 1: Header + KPI Summary
        # =====================================================================
        display_header()
        
        # Display current filter status inline
        if selected_state != "All States":
            st.info(f"📍 Showing data for: **{selected_state}**")
        
        # Summary metrics (KPI cards)
        summary = create_summary_metrics(filtered_hotspots)
        display_summary_cards(summary)
        
        st.divider()
        
        # =====================================================================
        # ROW 2: Geographic Overview (Hero Section)
        # =====================================================================
        with st.container(border=True):
            st.subheader("🗺️ Geographic Overview")
            display_map_section(filtered_hotspots)
        
        st.divider()
        
        # =====================================================================
        # ROW 3: Analytics (Side-by-Side)
        # =====================================================================
        col_left, col_right = st.columns(2)
        
        with col_left:
            with st.container(border=True):
                st.subheader("📊 District Comparison")
                display_bar_chart_section(filtered_hotspots)
        
        with col_right:
            with st.container(border=True):
                st.subheader("📈 System Trends")
                display_trend_section(filtered_metrics)
        
        st.divider()
        
        # =====================================================================
        # ROW 4: Actionable Insights
        # =====================================================================
        with st.container(border=True):
            st.subheader("📋 Action Recommendations")
            display_recommendations_section(filtered_recommendations)
        
        # Footer
        st.divider()
        st.caption(
            "**AARHI Dashboard** | Built for UIDAI Hackathon 2026 | "
            "Data represents observed operational patterns, not causal assessments."
        )
        
    except FileNotFoundError as e:
        st.error(f"""
        **Data files not found!**
        
        Please ensure the data pipeline has been run first:
        1. Run `python engine/data_prep.py`
        2. Run `python engine/ari.py`
        3. Run `python engine/clustering.py`
        4. Run `python engine/recommendations.py`
        
        Error details: {e}
        """)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    initialize_data_pipeline()
    main()
