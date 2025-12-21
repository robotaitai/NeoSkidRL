"""
Streamlit Reward Dashboard for NeoSkidRL

Visualizes reward terms, allows weight tuning, and shows training/eval results.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import altair as alt
import yaml
from streamlit_autorefresh import st_autorefresh


# ============================================================================
# Data Loading
# ============================================================================

def load_episodes_jsonl(path: Path, run_id_filter: str | None = None, last_n: int | None = None) -> pd.DataFrame:
    """Load episodes from JSONL file."""
    if not path.exists():
        return pd.DataFrame()
    
    records = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Filter by run_id if specified
    if run_id_filter and "run_id" in df.columns:
        df = df[df["run_id"].str.contains(run_id_filter, case=False, na=False)]
    
    # Take last N episodes
    if last_n and len(df) > last_n:
        df = df.tail(last_n)
    
    return df


def expand_reward_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Expand reward_terms_sum dict column into separate columns."""
    if df.empty or "reward_terms_sum" not in df.columns:
        return df
    
    # Extract reward terms into separate columns
    terms_df = pd.json_normalize(df["reward_terms_sum"])
    terms_df.columns = [f"term_{col}" for col in terms_df.columns]
    
    # Combine with original dataframe
    result = pd.concat([df.reset_index(drop=True), terms_df], axis=1)
    return result


def expand_reward_contrib(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Expand reward contribution dict column into separate columns."""
    if df.empty or column not in df.columns:
        return df
    contrib_df = pd.json_normalize(df[column])
    contrib_df.columns = [f"{prefix}{col}" for col in contrib_df.columns]
    result = pd.concat([df.reset_index(drop=True), contrib_df], axis=1)
    return result


def load_config_yaml(path: Path) -> Dict[str, Any]:
    """Load config YAML file."""
    if not path.exists():
        return {}
    
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def save_config_yaml(path: Path, config: Dict[str, Any]) -> None:
    """Save config YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_reward_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract reward weights from config."""
    reward_cfg = config.get("reward", {})
    
    # Try new format first (reward.weights)
    weights = reward_cfg.get("weights")
    if weights:
        return {k: float(v) for k, v in weights.items()}
    
    # Fall back to legacy format (w_* keys directly in reward)
    legacy_keys = {
        "w_progress": "progress",
        "w_time": "time",
        "w_smooth": "smooth",
        "w_heading": "heading",
        "w_velocity": "velocity",
        "w_collision": "collision",
        "w_goal_bonus": "goal_bonus",
    }
    
    weights = {}
    for legacy_key, term_name in legacy_keys.items():
        if legacy_key in reward_cfg:
            weights[term_name] = float(reward_cfg[legacy_key])
    
    return weights


def set_reward_weights(config: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """Update reward weights in config (supports both formats)."""
    if "reward" not in config:
        config["reward"] = {}
    
    reward_cfg = config["reward"]
    
    # If config already uses new format, update it
    if "weights" in reward_cfg:
        reward_cfg["weights"] = weights
    else:
        # Use legacy format
        legacy_mapping = {
            "progress": "w_progress",
            "time": "w_time",
            "smooth": "w_smooth",
            "heading": "w_heading",
            "velocity": "w_velocity",
            "collision": "w_collision",
            "goal_bonus": "w_goal_bonus",
        }
        for term_name, weight in weights.items():
            legacy_key = legacy_mapping.get(term_name, f"w_{term_name}")
            reward_cfg[legacy_key] = float(weight)
    
    return config


def find_latest_video(video_dir: Path) -> Path | None:
    """Find the most recent MP4 video file."""
    video_dir = video_dir.resolve()  # Ensure absolute path
    if not video_dir.exists():
        return None
    
    videos = list(video_dir.rglob("*.mp4"))
    if not videos:
        return None
    
    # Sort by modification time, newest first
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0].resolve()  # Return absolute path


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate metrics from episode dataframe."""
    if df.empty:
        return {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "stuck_rate": 0.0,
            "avg_return": 0.0,
            "avg_ep_len": 0.0,
            "total_episodes": 0,
        }
    
    return {
        "success_rate": df["success"].mean() * 100 if "success" in df.columns else 0.0,
        "collision_rate": df["collision"].mean() * 100 if "collision" in df.columns else 0.0,
        "stuck_rate": df["stuck"].mean() * 100 if "stuck" in df.columns else 0.0,
        "avg_return": df["ep_return"].mean() if "ep_return" in df.columns else 0.0,
        "avg_ep_len": df["ep_len"].mean() if "ep_len" in df.columns else 0.0,
        "total_episodes": len(df),
    }


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar(config_files: List[Path]) -> tuple:
    """Render sidebar with config selection and weight sliders."""
    st.sidebar.title("⚙️ Configuration")
    
    # Config file selection - resolve to absolute then convert to relative for display
    cwd = Path.cwd()
    config_options = []
    for p in config_files:
        try:
            # Resolve to absolute path first
            abs_path = p.resolve()
            # Try to make it relative to cwd
            config_options.append(str(abs_path.relative_to(cwd)))
        except ValueError:
            # If path is outside cwd, just use the path as-is
            config_options.append(str(p))
    selected_config = st.sidebar.selectbox(
        "Config File",
        config_options,
        index=0 if config_options else None,
    )
    
    # Find the original Path object corresponding to the selected option
    config_path = None
    if selected_config:
        # Match selected string back to original Path
        for i, opt in enumerate(config_options):
            if opt == selected_config:
                config_path = config_files[i].resolve()
                break
    config = load_config_yaml(config_path) if config_path else {}
    weights = get_reward_weights(config)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Reward Weights")
    
    # Reward term descriptions
    st.sidebar.markdown("""
    **Reward Terms Explained:**
    - **Progress**: Distance moved toward goal (positive = encourage)
    - **Heading**: Turn toward the goal direction (positive = encourage alignment)
    - **Velocity**: Reward for moving distance regardless of direction
    - **Time**: Penalty per timestep (negative = encourage speed)
    - **Smooth**: Penalty for action changes (negative = encourage smoothness)
    - **Collision**: Penalty when hitting obstacles (negative = discourage)
    - **Goal Bonus**: Bonus when goal reached (positive = encourage success)
    - **Stuck**: Penalty when robot gets stuck (negative = discourage)
    - **Clearance**: Penalty for being too close to obstacles (negative = keep distance)
    """)
    
    st.sidebar.markdown("---")
    
    # Default weight ranges
    weight_ranges = {
        "progress": (0.0, 20.0, 10.0),
        "heading": (0.0, 5.0, 2.0),
        "velocity": (0.0, 10.0, 5.0),
        "time": (-1.0, 0.0, -0.01),
        "smooth": (-1.0, 0.0, -0.05),
        "collision": (-100.0, 0.0, -75.0),
        "goal_bonus": (0.0, 100.0, 75.0),
        "stuck": (-50.0, 0.0, -25.0),
        "clearance": (-5.0, 0.0, -0.5),
    }
    
    # Reward term help text
    weight_help = {
        "progress": "Reward for distance moved toward goal. Higher = more aggressive toward goal. Recommended: 5-20",
        "heading": "Reward for turning toward goal direction. Small shaping term. Recommended: 1-3",
        "velocity": "Reward for moving distance per step, regardless of direction. Use small to moderate values: 1-5",
        "time": "Time penalty per step. More negative = faster episodes. Keep tiny early on: -0.005 to -0.01",
        "smooth": "Penalty for action changes. More negative = smoother motions. Polish term: -0.01 to -0.2",
        "collision": "Penalty when colliding. More negative = stronger avoidance. Recommended: -50 to -75",
        "goal_bonus": "Bonus for reaching goal. Higher = more motivated to complete. Recommended: 50-100",
        "stuck": "Penalty for getting stuck. More negative = stronger motivation to keep moving. Recommended: -20 to -30",
        "clearance": "Penalty for being too close to obstacles. More negative = keeps more distance. Recommended: -0.3 to -1.0",
    }
    
    new_weights = {}
    for term_name, (min_val, max_val, default_val) in weight_ranges.items():
        current_val = weights.get(term_name, default_val)
        new_weights[term_name] = st.sidebar.slider(
            f"w_{term_name}",
            min_value=min_val,
            max_value=max_val,
            value=float(current_val),
            step=0.01,
            help=weight_help[term_name],
        )
    
    st.sidebar.markdown("---")
    
    # Save buttons
    col1, col2 = st.sidebar.columns(2)
    save_clicked = col1.button("💾 Save", use_container_width=True)
    save_as_clicked = col2.button("📝 Save As", use_container_width=True)
    
    if save_clicked and config_path:
        config = set_reward_weights(config, new_weights)
        save_config_yaml(config_path, config)
        st.sidebar.success(f"Saved to {config_path.name}")
    
    if save_as_clicked and config_path:
        new_name = st.sidebar.text_input("New config name:", value=f"{config_path.stem}_modified.yml")
        if new_name:
            new_path = config_path.parent / new_name
            config = set_reward_weights(config, new_weights)
            save_config_yaml(new_path, config)
            st.sidebar.success(f"Saved as {new_path.name}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    run_id_filter = st.sidebar.text_input("Run ID filter:", value="", help="Filter episodes by run_id")
    last_n = st.sidebar.number_input("Last N episodes:", min_value=10, max_value=10000, value=100, step=10)
    
    st.sidebar.markdown("---")
    refresh_clicked = st.sidebar.button("🔄 Refresh Now", use_container_width=True)
    
    return config_path, new_weights, run_id_filter, last_n, refresh_clicked


def render_metric_cards(metrics: Dict[str, float]) -> None:
    """Render big metric cards."""
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
    
    with col2:
        st.metric("Collision Rate", f"{metrics['collision_rate']:.1f}%")
    
    with col3:
        st.metric("Stuck Rate", f"{metrics['stuck_rate']:.1f}%")
    
    with col4:
        st.metric("Avg Return", f"{metrics['avg_return']:.2f}")
    
    with col5:
        st.metric("Avg Ep Length", f"{metrics['avg_ep_len']:.1f}")
    
    st.caption(f"Based on {metrics['total_episodes']} episodes")


def render_return_chart(df: pd.DataFrame) -> None:
    """Render episode return line chart."""
    if df.empty or "ep_return" not in df.columns:
        st.info("No episode data available yet.")
        return
    
    st.subheader("📈 Episode Returns Over Time")
    st.caption("Higher returns = better performance. Look for upward trends as training progresses.")
    
    chart_data = df[["episode_idx", "ep_return"]].copy()
    
    chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X("episode_idx:Q", title="Episode"),
        y=alt.Y("ep_return:Q", title="Return"),
        tooltip=["episode_idx", "ep_return"],
    ).properties(
        height=300,
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def compute_reward_percentages(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Compute average percentage contribution of each reward term."""
    if df.empty:
        return pd.DataFrame()

    if "reward_contrib_abs_sum" in df.columns:
        df_expanded = expand_reward_contrib(df, "reward_contrib_abs_sum", "abs_")
        term_cols = [col for col in df_expanded.columns if col.startswith("abs_")]
        if not term_cols:
            return pd.DataFrame()
        contributions = {}
        for term_col in term_cols:
            term_name = term_col.replace("abs_", "")
            contributions[term_name] = float(df_expanded[term_col].sum())
    else:
        if "reward_terms_sum" not in df.columns:
            return pd.DataFrame()

        df_expanded = expand_reward_terms(df)
        term_cols = [col for col in df_expanded.columns if col.startswith("term_")]

        if not term_cols:
            return pd.DataFrame()

        # Compute weighted contributions for all episodes
        contributions = {}
        for term_col in term_cols:
            term_name = term_col.replace("term_", "")
            weight = weights.get(term_name, 0.0)
            # Sum across all episodes, then take absolute value for percentage calc
            total_contribution = (df_expanded[term_col] * weight).sum()
            contributions[term_name] = total_contribution
    
    # Convert to dataframe
    contrib_df = pd.DataFrame([
        {"term": term, "contribution": abs(val)}
        for term, val in contributions.items()
    ])
    
    # Calculate percentages
    total = contrib_df["contribution"].sum()
    if total > 0:
        contrib_df["percentage"] = (contrib_df["contribution"] / total * 100).round(1)
    else:
        contrib_df["percentage"] = 0.0
    
    return contrib_df


def render_reward_percentages_pie(df: pd.DataFrame, weights: Dict[str, float]) -> None:
    """Render pie chart showing overall reward term percentages."""
    st.subheader("🥧 Reward Term Breakdown")
    st.caption("Overall percentage contribution of each reward term across all episodes (weighted, absolute values).")
    
    contrib_df = compute_reward_percentages(df, weights)
    
    if contrib_df.empty:
        st.info("No reward term data available yet.")
        return
    
    # Create pie chart
    chart = alt.Chart(contrib_df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("percentage:Q", stack=True),
        color=alt.Color("term:N", title="Reward Term", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("term:N", title="Term"),
            alt.Tooltip("percentage:Q", title="Percentage", format=".1f"),
            alt.Tooltip("contribution:Q", title="Total Contribution", format=".2f"),
        ],
    ).properties(
        height=300,
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Show data table below
    with st.expander("📊 Detailed Breakdown"):
        display_df = contrib_df.copy()
        display_df["percentage"] = display_df["percentage"].apply(lambda x: f"{x:.1f}%")
        display_df["contribution"] = display_df["contribution"].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_reward_terms_chart(df: pd.DataFrame, weights: Dict[str, float]) -> None:
    """Render stacked area chart of reward term contributions."""
    if df.empty or ("reward_terms_sum" not in df.columns and "reward_contrib_sum" not in df.columns):
        st.info("No reward terms data available yet.")
        return
    
    st.subheader("🎨 Reward Term Contributions Over Time")
    st.caption("Shows how each reward term (weighted) contributes to total return over episodes. Adjust weights to balance behavior.")
    
    if "reward_contrib_sum" in df.columns:
        df_expanded = expand_reward_contrib(df, "reward_contrib_sum", "contrib_")
        term_cols = [col for col in df_expanded.columns if col.startswith("contrib_")]
        weight_lookup = None
    else:
        # Expand reward terms
        df_expanded = expand_reward_terms(df)
        term_cols = [col for col in df_expanded.columns if col.startswith("term_")]
        weight_lookup = weights
    
    if not term_cols:
        st.info("No reward terms found in episode data.")
        return
    
    # Compute weighted contributions
    chart_data = []
    for idx, row in df_expanded.iterrows():
        episode_idx = row.get("episode_idx", idx)
        for term_col in term_cols:
            if weight_lookup is None:
                term_name = term_col.replace("contrib_", "")
                contribution = row.get(term_col, 0.0)
            else:
                term_name = term_col.replace("term_", "")
                term_sum = row.get(term_col, 0.0)
                weight = weight_lookup.get(term_name, 0.0)
                contribution = term_sum * weight
            
            chart_data.append({
                "episode_idx": episode_idx,
                "term": term_name,
                "contribution": contribution,
            })
    
    if not chart_data:
        st.info("No reward term contributions to display.")
        return
    
    chart_df = pd.DataFrame(chart_data)
    
    chart = alt.Chart(chart_df).mark_area().encode(
        x=alt.X("episode_idx:Q", title="Episode"),
        y=alt.Y("contribution:Q", title="Weighted Contribution", stack="zero"),
        color=alt.Color("term:N", title="Reward Term"),
        tooltip=["episode_idx", "term", "contribution"],
    ).properties(
        height=300,
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)


def render_video_section(video_dir: Path) -> None:
    """Render latest evaluation video."""
    st.subheader("🎬 Latest Evaluation Video")
    
    latest_video = find_latest_video(video_dir)
    
    if latest_video:
        st.video(str(latest_video))
        
        # Try to display as relative path, fall back to absolute if outside cwd
        try:
            rel_path = latest_video.relative_to(Path.cwd().resolve())
            st.caption(f"Video: {rel_path}")
        except ValueError:
            st.caption(f"Video: {latest_video}")
        
        st.caption(f"Modified: {time.ctime(latest_video.stat().st_mtime)}")
    else:
        st.info("No evaluation videos found yet. Videos will appear here after running evaluation.")


# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(
        page_title="NeoSkidRL Reward Dashboard",
        page_icon="🚗",
        layout="wide",
    )
    
    st.title("🚗 NeoSkidRL Reward Dashboard")
    st.markdown("Real-time visualization of training metrics and reward tuning")
    
    # Overview expander
    with st.expander("ℹ️ How to Use This Dashboard", expanded=False):
        st.markdown("""
        **Purpose**: Monitor training progress and tune reward weights to improve robot behavior.
        
        **Workflow**:
        1. Start training with `visual_train.py` - episodes automatically log here
        2. Watch metrics update in real-time (auto-refreshes every 2 seconds)
        3. Adjust reward weights using sliders (sidebar) to balance behavior:
           - Low success? → Increase `w_progress` or `w_goal_bonus`
           - Too many collisions? → Make `w_collision` more negative
           - Jerky motion? → Make `w_smooth` more negative
           - Too slow? → Make `w_time` more negative
        4. Save modified weights and restart training
        5. Compare results in the charts
        
        **Charts**:
        - **Returns**: Shows overall episode performance (higher = better)
        - **Term Contributions**: Shows which reward terms dominate (helps debug reward shaping)
        
        **Video**: Latest evaluation video appears automatically after running eval.
        """)
    
    # Auto-refresh every 2 seconds
    st_autorefresh(interval=2000, key="dashboard_refresh")
    
    # Find available config files - resolve to absolute paths
    config_dir = Path("config").resolve()
    src_config_dir = Path("src/neoskidrl/config").resolve()
    
    config_files = []
    if config_dir.exists():
        config_files.extend([p.resolve() for p in config_dir.glob("*.yml")])
    if src_config_dir.exists():
        config_files.extend([p.resolve() for p in src_config_dir.glob("*.yml")])
    
    if not config_files:
        st.error("No config files found! Please create a config YAML file.")
        return
    
    # Render sidebar
    config_path, weights, run_id_filter, last_n, refresh_clicked = render_sidebar(config_files)
    
    # Load episode data
    episodes_path = Path("runs/metrics/episodes.jsonl")
    df = load_episodes_jsonl(episodes_path, run_id_filter=run_id_filter, last_n=last_n)
    
    # Compute metrics
    metrics = compute_metrics(df)
    
    # Render main content
    render_metric_cards(metrics)
    
    st.markdown("---")
    
    # First row: Return chart and Pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        render_return_chart(df)
    
    with col2:
        render_reward_percentages_pie(df, weights)
    
    st.markdown("---")
    
    # Second row: Reward terms over time (full width)
    render_reward_terms_chart(df, weights)
    
    st.markdown("---")
    
    render_video_section(Path("runs/eval_videos"))
    
    # Show raw data in expander
    with st.expander("📋 Raw Episode Data"):
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No episode data available yet. Start training to see data here.")


if __name__ == "__main__":
    main()
