import json
import math
import time
from pathlib import Path

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashplayershotlocations,
    shotchartdetail,
)
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as nba_teams
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = Path(__file__).parent / "data" / "cache"
BASELINE_DIR = Path(__file__).parent / "data" / "baselines"

NBAStatsHTTP.timeout = 90
NBA_MAX_RETRIES = 4
NBA_RETRY_DELAY = 2

# NBA tracking coords (tenths of a foot). API fields are LOC_X / LOC_Y (= SHOT_X / SHOT_Y).
COURT_X_MIN, COURT_X_MAX = -250, 250
COURT_Y_MIN, COURT_Y_MAX = -50, 422
DEFAULT_HEX_GRIDSIZE = 22
DEFAULT_MIN_HEX_SHOTS = 5


def _call_with_retry(fetch_fn, label: str):
    """Retry NBA API calls — cloud requests often need longer timeouts."""
    last_err = None
    for attempt in range(NBA_MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as err:
            last_err = err
            if attempt < NBA_MAX_RETRIES - 1:
                time.sleep(NBA_RETRY_DELAY * (attempt + 1))
    raise last_err


def _zone_for_court_point(x_ft: float, y_ft: float) -> str:
    """Approximate NBA SHOT_ZONE_BASIC from court coordinates (feet)."""
    dist = math.hypot(x_ft, 47 - y_ft)
    if y_ft < 0:
        return "Backcourt"
    if dist <= 4:
        return "Restricted Area"
    if abs(x_ft) <= 8 and y_ft <= 19:
        return "In The Paint (Non-RA)"
    if dist >= 23.75 and y_ft <= 14:
        return "Left Corner 3" if x_ft < 0 else "Right Corner 3"
    if dist >= 23.75:
        return "Above the Break 3"
    return "Mid-Range"


def get_player_shot_vectors(season='2025-26'):
    """
    Fetches raw shot counts across court zones for all players 
    and transforms them into normalized percentage vectors.
    """
    print(f"Fetching shot location data for the {season} season...")
    
    # This specific endpoint breaks down shot attempts by zone across the whole league
    def _fetch():
        return leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
            season=season,
            distance_range='By Zone',
        )

    raw_data = _call_with_retry(_fetch, "league shot vectors")
    
    # The NBA API returns multi-index headers for this endpoint. Let's clean it.
    df = raw_data.get_data_frames()[0]
    
    # Flatten the column names (NBA API uses a 2-tier header here)
    # Tier 0 is player info or zone name. Tier 1 is FGM, FGA, FG_PCT.
    clean_cols = []
    for col in df.columns:
        tier0 = str(col[0])
        if not tier0 or tier0.startswith('Unnamed'):
            clean_cols.append(col[1]) # Keep PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION
        else:
            clean_cols.append(f"{col[0]}_{col[1]}") # e.g., "Restricted Area_FGA"
    df.columns = clean_cols

    # We want to look strictly at Shot Selection (Frequency), which uses Attempts (FGA)
    fga_cols = [c for c in df.columns if c.endswith('_FGA')]
    player_info_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
    
    # Filter down to just the columns we need
    vector_df = df[player_info_cols + fga_cols].copy()
    
    # Filter out benchwarmers who haven't taken enough shots to have a stable profile
    total_fga = vector_df[fga_cols].sum(axis=1)
    vector_df = vector_df[total_fga >= 150].reset_index(drop=True)
    
    # CRUCIAL STEP: Normalize rows so they represent percentages (0.0 to 1.0) instead of raw counts.
    # This ensures a superstar taking 20 shots a game matches a bench player taking 4 shots a game if their styles are identical.
    row_sums = vector_df[fga_cols].sum(axis=1)
    for col in fga_cols:
        vector_df[col] = vector_df[col] / row_sums
        
    return vector_df, fga_cols

def find_similar_players(target_name, vector_df, fga_cols, top_n=5):
    """
    Calculates Cosine Similarity matrices and returns the closest matches.
    """
    # Locate our target player
    target_player = vector_df[vector_df['PLAYER_NAME'].str.upper() == target_name.upper()]
    
    if target_player.empty:
        return f"Player '{target_name}' not found or didn't meet the 150 shot threshold."
    
    # Extract feature matrices for similarity calculation
    target_vector = target_player[fga_cols].values
    all_vectors = vector_df[fga_cols].values
    
    # Compute Cosine Similarity between our target and everyone else in the dataset
    similarity_scores = cosine_similarity(target_vector, all_vectors)[0]
    
    # Append scores to a temporary dataframe to sort them
    results_df = vector_df.copy()
    results_df['SIM_SCORE'] = similarity_scores
    
    # Sort descending, exclude the target player themselves (score will be 1.0)
    matched_players = results_df[results_df['PLAYER_NAME'].str.upper() != target_name.upper()]
    matched_players = matched_players.sort_values(by='SIM_SCORE', ascending=False)
    
    return matched_players[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SIM_SCORE'] + fga_cols].head(top_n)


def _zone_labels(fga_cols):
    return [col.replace("_FGA", "") for col in fga_cols]


def create_shot_profile_comparison_chart(
    player_a_name,
    player_a_row,
    player_b_name,
    player_b_row,
    fga_cols,
    chart_type="both",
):
    """
    Compare two players' normalized zone shot shares with a radar and/or grouped bar chart.

    chart_type: "radar", "bar", or "both" (side-by-side subplots).
    Returns a Plotly Figure ready for st.plotly_chart().
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    zones = _zone_labels(fga_cols)
    a_pct = (player_a_row[fga_cols].values * 100).tolist()
    b_pct = (player_b_row[fga_cols].values * 100).tolist()
    y_max = max(max(a_pct), max(b_pct)) * 1.15

    radar_traces = [
        go.Scatterpolar(
            r=a_pct + [a_pct[0]],
            theta=zones + [zones[0]],
            name=player_a_name,
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.25)",
            line=dict(color="rgb(31, 119, 180)", width=2),
        ),
        go.Scatterpolar(
            r=b_pct + [b_pct[0]],
            theta=zones + [zones[0]],
            name=player_b_name,
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.25)",
            line=dict(color="rgb(255, 127, 14)", width=2),
        ),
    ]
    bar_traces = [
        go.Bar(name=player_a_name, x=zones, y=a_pct, marker_color="rgb(31, 119, 180)"),
        go.Bar(name=player_b_name, x=zones, y=b_pct, marker_color="rgb(255, 127, 14)"),
    ]

    if chart_type == "radar":
        fig = go.Figure(data=radar_traces)
        fig.update_layout(
            polar=dict(radialaxis=dict(ticksuffix="%", range=[0, y_max])),
            title="Shot zone profile — radar comparison",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=520,
            margin=dict(t=80, b=40),
        )
    elif chart_type == "bar":
        fig = go.Figure(data=bar_traces)
        fig.update_layout(
            barmode="group",
            title="Shot zone profile — grouped bar comparison",
            xaxis_title="Court zone",
            yaxis_title="Share of attempts (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=520,
            margin=dict(t=80, b=80),
        )
        fig.update_xaxes(tickangle=-35)
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "polar"}, {"type": "xy"}]],
            subplot_titles=("Radar comparison", "Grouped bar comparison"),
        )
        for trace in radar_traces:
            fig.add_trace(trace, row=1, col=1)
        for trace in bar_traces:
            fig.add_trace(trace, row=1, col=2)
        fig.update_layout(
            barmode="group",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
            margin=dict(t=100, b=80),
        )
        fig.update_polars(radialaxis=dict(ticksuffix="%", range=[0, y_max]))
        fig.update_xaxes(tickangle=-35, row=1, col=2)
        fig.update_yaxes(title_text="Share of attempts (%)", row=1, col=2)

    return fig


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _court_extent_feet() -> list[float]:
    return [
        COURT_X_MIN / 10.0,
        COURT_X_MAX / 10.0,
        COURT_Y_MIN / 10.0,
        COURT_Y_MAX / 10.0,
    ]


def get_active_players(season: str = "2024-25") -> pd.DataFrame:
    """Return all active NBA players for a season (not filtered by FGA)."""
    _ensure_cache_dir()
    cache_path = CACHE_DIR / f"active_players_{season.replace('-', '_')}.json"

    if cache_path.exists():
        return pd.DataFrame(json.loads(cache_path.read_text()))

    def _fetch():
        endpoint = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00",
            season=season,
        )
        return endpoint.get_data_frames()[0]

    players_df = _call_with_retry(_fetch, "active players")
    players_df = players_df[players_df["ROSTERSTATUS"] == 1].copy()
    players_df = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ABBREVIATION"]]
    players_df.columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
    cache_path.write_text(players_df.to_json(orient="records"))

    return players_df


def fetch_player_shot_chart(player_id: int, season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch shot-level tracking data for one player.

    Returns LOC_X / LOC_Y (NBA's shot coordinates), SHOT_MADE_FLAG, and zone labels.
    """
    _ensure_cache_dir()
    cache_path = CACHE_DIR / f"player_shots_{player_id}_{season.replace('-', '_')}.json"

    if cache_path.exists():
        return pd.DataFrame(json.loads(cache_path.read_text()))

    def _fetch():
        endpoint = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable=season,
            context_measure_simple="FGA",
            league_id="00",
        )
        return endpoint.get_data_frames()[0]

    shots_df = _call_with_retry(_fetch, f"player shots {player_id}")
    cache_path.write_text(shots_df.to_json(orient="records"))
    return shots_df


def fetch_team_shot_chart(team_id: int, season: str = "2024-25") -> pd.DataFrame:
    """Fetch all shot attempts for one team (used to build league hex baselines)."""
    _ensure_cache_dir()
    cache_path = CACHE_DIR / f"team_shots_{team_id}_{season.replace('-', '_')}.json"

    if cache_path.exists():
        return pd.DataFrame(json.loads(cache_path.read_text()))

    endpoint = shotchartdetail.ShotChartDetail(
        player_id=0,
        team_id=team_id,
        season_nullable=season,
        context_measure_simple="FGA",
        league_id="00",
    )
    shots_df = endpoint.get_data_frames()[0]
    cache_path.write_text(shots_df.to_json(orient="records"))
    return shots_df


def get_league_shot_chart(season: str = "2024-25") -> pd.DataFrame:
    """Combine cached team shot charts into a league-wide shot dataset."""
    _ensure_cache_dir()
    cache_path = CACHE_DIR / f"league_shots_{season.replace('-', '_')}.json"

    if cache_path.exists():
        return pd.DataFrame(json.loads(cache_path.read_text()))

    frames = []
    for team in nba_teams.get_teams():
        frames.append(fetch_team_shot_chart(team["id"], season=season))

    league_df = pd.concat(frames, ignore_index=True)
    cache_path.write_text(league_df.to_json(orient="records"))
    return league_df


def get_league_zone_fg_pct(season: str = "2024-25") -> pd.DataFrame:
    """League-wide FG% by court zone from LeagueDashPlayerShotLocations."""
    def _fetch():
        return leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
            season=season,
            distance_range="By Zone",
        )

    raw_data = _call_with_retry(_fetch, "league zone fg")
    df = raw_data.get_data_frames()[0]

    clean_cols = []
    for col in df.columns:
        tier0 = str(col[0])
        if not tier0 or tier0.startswith("Unnamed"):
            clean_cols.append(col[1])
        else:
            clean_cols.append(f"{col[0]}_{col[1]}")
    df.columns = clean_cols

    zone_rows = []
    for col in [c for c in df.columns if c.endswith("_FGA")]:
        zone = col.replace("_FGA", "")
        fgm_col = f"{zone}_FGM"
        if fgm_col not in df.columns:
            continue
        fga = df[col].sum()
        fgm = df[fgm_col].sum()
        zone_rows.append(
            {
                "zone": zone,
                "fga": fga,
                "fgm": fgm,
                "fg_pct": fgm / fga if fga else np.nan,
            }
        )

    return pd.DataFrame(zone_rows).sort_values("fga", ascending=False)


def _assign_hex_keys(
    x_tenths: np.ndarray,
    y_tenths: np.ndarray,
    gridsize: int,
    extent_ft: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each shot to the center of its hex cell on the court grid."""
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    x_ft = np.asarray(x_tenths, dtype=float) / 10.0
    y_ft = np.asarray(y_tenths, dtype=float) / 10.0

    fig, ax = plt.subplots(figsize=(1, 1))
    hb = ax.hexbin(x_ft, y_ft, gridsize=gridsize, extent=extent_ft, mincnt=1)
    centers = hb.get_offsets()
    plt.close(fig)

    _, idx = cKDTree(centers).query(np.column_stack([x_ft, y_ft]))
    hex_x = centers[idx, 0]
    hex_y = centers[idx, 1]
    hex_keys = hex_x.round(3).astype(str) + "_" + hex_y.round(3).astype(str)
    return hex_keys, hex_x, hex_y


def aggregate_hex_fg_stats(
    shots_df: pd.DataFrame,
    gridsize: int = DEFAULT_HEX_GRIDSIZE,
) -> pd.DataFrame:
    """Aggregate makes, attempts, and FG% for each hex cell."""
    if shots_df.empty:
        return pd.DataFrame(columns=["hex_key", "x", "y", "fga", "fgm", "fg_pct"])

    extent_ft = _court_extent_feet()
    hex_keys, hex_x, hex_y = _assign_hex_keys(
        shots_df["LOC_X"].to_numpy(),
        shots_df["LOC_Y"].to_numpy(),
        gridsize,
        extent_ft,
    )

    grouped = (
        pd.DataFrame(
            {
                "hex_key": hex_keys,
                "x": hex_x,
                "y": hex_y,
                "fgm": shots_df["SHOT_MADE_FLAG"].to_numpy(),
            }
        )
        .groupby("hex_key", as_index=False)
        .agg(fga=("fgm", "count"), fgm=("fgm", "sum"), x=("x", "first"), y=("y", "first"))
    )
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"]
    return grouped


def _league_hex_from_zones(season: str, gridsize: int) -> pd.DataFrame:
    """Lightweight league hex baseline from zone FG% (one API call)."""
    import matplotlib.pyplot as plt

    zone_fg = get_league_zone_fg_pct(season).set_index("zone")["fg_pct"].to_dict()
    extent_ft = _court_extent_feet()

    x_centers = np.linspace(extent_ft[0], extent_ft[1], gridsize * 2)
    y_centers = np.linspace(extent_ft[2], extent_ft[3], gridsize)
    xx, yy = np.meshgrid(x_centers, y_centers)
    x_flat, y_flat = xx.ravel(), yy.ravel()

    fig, ax = plt.subplots(figsize=(1, 1))
    hb = ax.hexbin(x_flat, y_flat, gridsize=gridsize, extent=extent_ft, mincnt=1)
    centers = hb.get_offsets()
    plt.close(fig)

    rows = []
    for x_val, y_val in centers:
        zone = _zone_for_court_point(float(x_val), float(y_val))
        rows.append(
            {
                "hex_key": f"{round(x_val, 3)}_{round(y_val, 3)}",
                "x": x_val,
                "y": y_val,
                "fga": 1,
                "fgm": zone_fg.get(zone, 0.45),
                "fg_pct": zone_fg.get(zone, 0.45),
            }
        )
    return pd.DataFrame(rows)


def get_league_hex_baseline(
    season: str = "2024-25",
    gridsize: int = DEFAULT_HEX_GRIDSIZE,
) -> pd.DataFrame:
    """Load precomputed league hex FG% or build from zone averages."""
    bundled = BASELINE_DIR / f"league_hex_{season.replace('-', '_')}_gs{gridsize}.json"
    if bundled.exists():
        return pd.DataFrame(json.loads(bundled.read_text()))

    return _league_hex_from_zones(season, gridsize)


def compute_zone_efficiency_vs_league(
    player_shots: pd.DataFrame,
    league_zones: pd.DataFrame,
) -> pd.DataFrame:
    """Compare player FG% by SHOT_ZONE_BASIC against league zone baselines."""
    if player_shots.empty:
        return pd.DataFrame()

    player_zones = (
        player_shots.groupby("SHOT_ZONE_BASIC", as_index=False)
        .agg(fga=("SHOT_MADE_FLAG", "count"), fgm=("SHOT_MADE_FLAG", "sum"))
        .assign(fg_pct=lambda d: d["fgm"] / d["fga"])
    )
    merged = player_zones.merge(
        league_zones.rename(columns={"zone": "SHOT_ZONE_BASIC", "fg_pct": "league_fg_pct"}),
        on="SHOT_ZONE_BASIC",
        how="left",
    )
    merged["fg_delta"] = merged["fg_pct"] - merged["league_fg_pct"]
    return merged.sort_values("fga", ascending=False)


def draw_court(ax, line_color: str = "white", lw: float = 1.4) -> None:
    """Draw a half-court overlay in feet (LOC_X/10, LOC_Y/10)."""
    from matplotlib.patches import Arc, Circle, Rectangle

    ax.set_facecolor("#1a1a2e")
    ax.plot([-25, 25], [0, 0], color=line_color, lw=lw, zorder=1)
    ax.plot([-25, -25, 25, 25], [0, 47, 47, 0], color=line_color, lw=lw, zorder=1)
    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, edgecolor=line_color, lw=lw, zorder=1))
    ax.add_patch(Circle((0, 47), radius=4, fill=False, edgecolor=line_color, lw=lw, zorder=1))
    ax.add_patch(
        Arc((0, 47), width=47.5, height=47.5, angle=0, theta1=22, theta2=158, edgecolor=line_color, lw=lw, zorder=1)
    )
    ax.add_patch(
        Arc((0, 47), width=23.75, height=23.75, angle=0, theta1=22, theta2=158, edgecolor=line_color, lw=lw, zorder=1)
    )
    ax.set_xlim(-26, 26)
    ax.set_ylim(-2, 48)
    ax.set_aspect("equal")
    ax.axis("off")


def create_efficiency_shot_chart(
    player_shots: pd.DataFrame,
    league_hex: pd.DataFrame,
    player_name: str,
    season: str,
    gridsize: int = DEFAULT_HEX_GRIDSIZE,
    min_hex_shots: int = DEFAULT_MIN_HEX_SHOTS,
):
    """
    Plot a half-court hex map colored by player FG% minus league-average FG%
    in each hex cell. Returns (matplotlib Figure, hex comparison DataFrame).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="dark")

    player_hex = aggregate_hex_fg_stats(player_shots, gridsize=gridsize)
    comparison = player_hex.merge(
        league_hex[["hex_key", "fg_pct", "fga"]].rename(
            columns={"fg_pct": "league_fg_pct", "fga": "league_fga"}
        ),
        on="hex_key",
        how="inner",
    )
    comparison = comparison[comparison["fga"] >= min_hex_shots].copy()
    comparison["fg_delta"] = comparison["fg_pct"] - comparison["league_fg_pct"]

    fig, ax = plt.subplots(figsize=(10, 9), facecolor="#0f0f1a")
    draw_court(ax)

    x_ft = player_shots["LOC_X"].to_numpy(dtype=float) / 10.0
    y_ft = player_shots["LOC_Y"].to_numpy(dtype=float) / 10.0

    if not comparison.empty:
        delta_lookup = comparison.set_index("hex_key")["fg_delta"]
        hex_keys, _, _ = _assign_hex_keys(
            player_shots["LOC_X"].to_numpy(),
            player_shots["LOC_Y"].to_numpy(),
            gridsize,
            _court_extent_feet(),
        )
        shot_delta = pd.Series(hex_keys).map(delta_lookup).to_numpy()

        hb = ax.hexbin(
            x_ft,
            y_ft,
            C=shot_delta,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            extent=_court_extent_feet(),
            mincnt=min_hex_shots,
            cmap="RdYlGn",
            vmin=-0.12,
            vmax=0.12,
            alpha=0.92,
            linewidths=0.35,
            edgecolors="#555555",
            zorder=2,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("FG% vs league", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.scatter(x_ft, y_ft, c="white", s=4, alpha=0.07, zorder=3)
    ax.set_title(f"{player_name}, {season}", color="white", fontsize=14, pad=10)
    fig.tight_layout()
    return fig, comparison

# Quick local test execution
if __name__ == "__main__":
    df, fga_features = get_player_shot_vectors('2024-25')
    
    # Let's test a known unique shot profile (e.g., Stephen Curry)
    test_player = "Stephen Curry"
    matches = find_similar_players(test_player, df, fga_features)
    
    print(f"\n--- Top Playstyle Matches for {test_player} ---")
    print(matches.to_string(index=False))
