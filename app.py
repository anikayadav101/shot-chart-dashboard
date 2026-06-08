import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_pipeline import (
    compute_zone_efficiency_vs_league,
    create_efficiency_shot_chart,
    fetch_player_shot_chart,
    find_similar_players,
    get_active_players,
    get_league_hex_baseline,
    get_league_zone_fg_pct,
    get_player_shot_vectors,
)

st.set_page_config(page_title="Shot Analytics", layout="wide")

st.title("Shot Analytics")

SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23"]


@st.cache_data(ttl=3600, show_spinner="Loading…")
def load_player_vectors(season: str) -> tuple[pd.DataFrame, list[str]]:
    return get_player_shot_vectors(season)


@st.cache_data(ttl=3600, show_spinner="Loading…")
def load_active_players(season: str) -> pd.DataFrame:
    return get_active_players(season)


@st.cache_data(ttl=3600, show_spinner="Loading…")
def load_player_shots(player_id: int, season: str) -> pd.DataFrame:
    return fetch_player_shot_chart(player_id, season)


@st.cache_data(ttl=86400, show_spinner="Loading league baseline…")
def load_league_hex(season: str, gridsize: int) -> pd.DataFrame:
    return get_league_hex_baseline(season, gridsize=gridsize)


@st.cache_data(ttl=3600, show_spinner="Loading…")
def load_league_zones(season: str) -> pd.DataFrame:
    return get_league_zone_fg_pct(season)


season = st.sidebar.selectbox("Season", SEASONS, index=1)

tab_match, tab_chart = st.tabs(["Playstyle", "Shot chart"])

with tab_match:
    vector_df, fga_cols = load_player_vectors(season)

    match_options = sorted(vector_df["PLAYER_NAME"].unique().tolist())
    selected_match_player = st.selectbox(
        "Player",
        match_options,
        key="match_player",
    )

    player_row = vector_df.loc[vector_df["PLAYER_NAME"] == selected_match_player].iloc[0]
    st.write(f"{selected_match_player} · {player_row['TEAM_ABBREVIATION']}")

    matches = find_similar_players(
        selected_match_player, vector_df, fga_cols, top_n=5
    )

    st.subheader("Similar players")
    if isinstance(matches, str):
        st.warning(matches)
    else:
        results_table = matches[["PLAYER_NAME", "TEAM_ABBREVIATION", "SIM_SCORE"]].copy()
        results_table.columns = ["Player", "Team", "Similarity"]
        results_table.index = range(1, len(results_table) + 1)
        st.dataframe(
            results_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Similarity": st.column_config.NumberColumn(format="%.4f"),
            },
        )

with tab_chart:
    gridsize = st.sidebar.slider("Hex grid size", 16, 30, 22, key="hex_grid")
    min_hex_shots = st.sidebar.slider("Min FGA per hex", 3, 12, 5, key="min_hex")

    active_df = load_active_players(season)
    chart_options = sorted(active_df["PLAYER_NAME"].unique().tolist())
    selected_chart_player = st.selectbox(
        "Player",
        chart_options,
        key="chart_player",
    )

    chart_row = active_df.loc[active_df["PLAYER_NAME"] == selected_chart_player].iloc[0]
    player_id = int(chart_row["PLAYER_ID"])

    col_map, col_zones = st.columns([1.6, 1])

    player_shots = load_player_shots(player_id, season)
    league_hex = load_league_hex(season, gridsize)
    league_zones = load_league_zones(season)

    if player_shots.empty:
        st.warning("No shot data for this player and season.")
    else:
        total_fga = len(player_shots)
        total_fgm = int(player_shots["SHOT_MADE_FLAG"].sum())
        overall_fg = total_fgm / total_fga if total_fga else 0

        metric_cols = st.columns(4)
        metric_cols[0].metric("FGA", f"{total_fga:,}")
        metric_cols[1].metric("FGM", f"{total_fgm:,}")
        metric_cols[2].metric("FG%", f"{overall_fg:.1%}")
        metric_cols[3].metric("Team", chart_row["TEAM_ABBREVIATION"])

        fig, hex_comparison = create_efficiency_shot_chart(
            player_shots=player_shots,
            league_hex=league_hex,
            player_name=selected_chart_player,
            season=season,
            gridsize=gridsize,
            min_hex_shots=min_hex_shots,
        )

        with col_map:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_zones:
            st.subheader("By zone")
            zone_table = compute_zone_efficiency_vs_league(player_shots, league_zones)
            if not zone_table.empty:
                display_zones = zone_table[
                    ["SHOT_ZONE_BASIC", "fga", "fg_pct", "league_fg_pct", "fg_delta"]
                ].copy()
                display_zones.columns = ["Zone", "FGA", "FG%", "League FG%", "Diff"]
                st.dataframe(
                    display_zones,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "FG%": st.column_config.NumberColumn(format="%.1%"),
                        "League FG%": st.column_config.NumberColumn(format="%.1%"),
                        "Diff": st.column_config.NumberColumn(format="+%.1%"),
                    },
                )

            if not hex_comparison.empty:
                st.subheader("Hex outliers")
                hot = hex_comparison.nlargest(
                    3, "fg_delta"
                )[["x", "y", "fga", "fg_pct", "league_fg_pct", "fg_delta"]]
                cold = hex_comparison.nsmallest(
                    3, "fg_delta"
                )[["x", "y", "fga", "fg_pct", "league_fg_pct", "fg_delta"]]
                st.dataframe(hot, use_container_width=True, hide_index=True)
                st.dataframe(cold, use_container_width=True, hide_index=True)
