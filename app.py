
import os
from datetime import date
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MLB K Prop Scanner", page_icon="⚾", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: .8rem; padding-bottom: 2rem;}
.stButton > button {width: 100%; min-height: 3.2rem; font-weight: 750; font-size: 1.05rem; border-radius: 14px;}
div[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 14px;}
h1 {font-size: 2rem !important;}
</style>
""", unsafe_allow_html=True)

MLB_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"

TEAM_NAME_MAP = {
    "Athletics": "Oakland Athletics",
    "A's": "Oakland Athletics",
}

def get_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=900)
def get_schedule(day_string):
    data = get_json(
        f"{MLB_BASE}/schedule",
        params={
            "sportId": 1,
            "date": day_string,
            "hydrate": "probablePitcher,team",
        },
    )
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            away_prob = g["teams"]["away"].get("probablePitcher", {})
            home_prob = g["teams"]["home"].get("probablePitcher", {})
            games.append({
                "gamePk": g.get("gamePk"),
                "status": g.get("status", {}).get("detailedState", ""),
                "gameDate": g.get("gameDate", ""),
                "away_team": away.get("name", ""),
                "away_team_id": away.get("id"),
                "away_pitcher": away_prob.get("fullName", ""),
                "away_pitcher_id": away_prob.get("id"),
                "home_team": home.get("name", ""),
                "home_team_id": home.get("id"),
                "home_pitcher": home_prob.get("fullName", ""),
                "home_pitcher_id": home_prob.get("id"),
            })
    return games

@st.cache_data(ttl=3600)
def get_pitcher_season_stats(player_id, season):
    if not player_id:
        return None
    data = get_json(
        f"{MLB_BASE}/people/{player_id}/stats",
        params={"stats": "season", "group": "pitching", "season": season},
    )
    stats = {}
    try:
        stats = data["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError, TypeError):
        return None

    so = float(stats.get("strikeOuts", 0) or 0)
    bf = float(stats.get("battersFaced", 0) or 0)
    gs = float(stats.get("gamesStarted", 0) or 0)

    k_rate = so / bf if bf > 0 else None
    expected_bf = bf / gs if gs > 0 else None

    return {
        "strikeouts": so,
        "batters_faced": bf,
        "games_started": gs,
        "k_rate": k_rate,
        "expected_bf": expected_bf,
        "innings": stats.get("inningsPitched", ""),
        "era": stats.get("era", ""),
    }

@st.cache_data(ttl=3600)
def get_team_hitting_stats(team_id, season):
    if not team_id:
        return None
    data = get_json(
        f"{MLB_BASE}/teams/{team_id}/stats",
        params={"stats": "season", "group": "hitting", "season": season},
    )
    try:
        stats = data["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError, TypeError):
        return None

    so = float(stats.get("strikeOuts", 0) or 0)
    pa = float(stats.get("plateAppearances", 0) or 0)
    k_rate = so / pa if pa > 0 else None

    return {
        "strikeouts": so,
        "plate_appearances": pa,
        "k_rate": k_rate,
    }

def american_to_implied(odds):
    odds = int(odds)
    if odds == 0:
        return 0.5
    return 100 / (odds + 100) if odds > 0 else (-odds) / ((-odds) + 100)

def profit_per_unit(odds):
    odds = int(odds)
    return odds / 100 if odds > 0 else 100 / (-odds)

def expected_value(prob, odds):
    return prob * profit_per_unit(odds) - (1 - prob)

def simulate_k_prop(pitcher_k_rate, opponent_k_rate, expected_bf, line, sims=50000, seed=42):
    """
    Baseline Monte Carlo:
    matchup K% = geometric mean of pitcher K% and opponent team K%.
    BF varies around pitcher season BF/start.
    """
    matchup_k = float(np.sqrt(pitcher_k_rate * opponent_k_rate))
    expected_bf = float(np.clip(expected_bf, 12, 32))
    rng = np.random.default_rng(seed)

    # Normal BF distribution is easier to control than raw Poisson.
    bf = np.rint(rng.normal(expected_bf, 3.1, sims)).astype(int)
    bf = np.clip(bf, 10, 34)

    ks = rng.binomial(bf, np.clip(matchup_k, .05, .55))
    over = float(np.mean(ks > line))
    under = float(np.mean(ks < line))
    push = float(np.mean(ks == line))
    return {
        "projection": float(np.mean(ks)),
        "over_prob": over,
        "under_prob": under,
        "push_prob": push,
        "matchup_k_rate": matchup_k,
        "samples": ks,
    }

@st.cache_data(ttl=300)
def get_odds_events(api_key):
    return get_json(
        f"{ODDS_BASE}/sports/{SPORT_KEY}/events",
        params={"apiKey": api_key},
    )

@st.cache_data(ttl=300)
def get_event_pitcher_k_odds(api_key, event_id, bookmakers="draftkings,fanduel,betmgm,caesars"):
    return get_json(
        f"{ODDS_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "pitcher_strikeouts",
            "oddsFormat": "american",
            "bookmakers": bookmakers,
        },
    )

def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)

def odds_event_for_game(events, away, home):
    away = normalize_team_name(away).lower()
    home = normalize_team_name(home).lower()
    for ev in events:
        if ev.get("away_team", "").lower() == away and ev.get("home_team", "").lower() == home:
            return ev
    return None

def extract_pitcher_props(event_odds):
    rows = []
    for book in event_odds.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "pitcher_strikeouts":
                continue
            for out in market.get("outcomes", []):
                player = out.get("description", "")
                side = out.get("name", "")
                line = out.get("point")
                price = out.get("price")
                if not player or line is None or price is None or side not in ("Over", "Under"):
                    continue
                rows.append({
                    "Book": book.get("title", book.get("key", "")),
                    "Pitcher": player,
                    "Side": side,
                    "Line": float(line),
                    "Odds": int(price),
                })
    return pd.DataFrame(rows)

def build_pitcher_rows(games, season):
    rows = []
    for g in games:
        pairs = [
            (g["away_pitcher"], g["away_pitcher_id"], g["home_team"], g["home_team_id"], g),
            (g["home_pitcher"], g["home_pitcher_id"], g["away_team"], g["away_team_id"], g),
        ]
        for pitcher, pid, opponent, opp_id, game in pairs:
            if not pitcher or not pid:
                continue
            pstats = get_pitcher_season_stats(pid, season)
            tstats = get_team_hitting_stats(opp_id, season)
            if not pstats or not tstats or not pstats.get("k_rate") or not tstats.get("k_rate"):
                continue
            expected_bf = pstats.get("expected_bf") or 21.0
            rows.append({
                "Pitcher": pitcher,
                "Pitcher ID": pid,
                "Opponent": opponent,
                "Pitcher K%": pstats["k_rate"],
                "Opponent K%": tstats["k_rate"],
                "Expected BF": float(np.clip(expected_bf, 12, 32)),
                "GamePk": game["gamePk"],
                "Away Team": game["away_team"],
                "Home Team": game["home_team"],
            })
    return pd.DataFrame(rows)

def model_one_prop(row, line, odds, side="Over", sims=50000):
    sim = simulate_k_prop(
        row["Pitcher K%"],
        row["Opponent K%"],
        row["Expected BF"],
        line,
        sims=sims,
    )
    prob = sim["over_prob"] if side == "Over" else sim["under_prob"]
    implied = american_to_implied(odds)
    return {
        **row.to_dict(),
        "Side": side,
        "Line": line,
        "Odds": odds,
        "Projection": sim["projection"],
        "Model Probability": prob,
        "Implied Probability": implied,
        "Edge": prob - implied,
        "EV": expected_value(prob, odds),
        "Matchup K%": sim["matchup_k_rate"],
    }, sim

st.title("⚾ MLB K Prop Scanner")
st.caption("Load today's probable pitchers automatically, then simulate strikeout props.")

today = date.today()
selected_date = st.date_input("Game date", value=today)
season = selected_date.year

if "pitchers_df" not in st.session_state:
    st.session_state.pitchers_df = pd.DataFrame()

if st.button("1️⃣ Load Today's Pitchers", type="primary"):
    try:
        games = get_schedule(selected_date.isoformat())
        if not games:
            st.warning("No MLB games were returned for this date.")
        else:
            pitchers = build_pitcher_rows(games, season)
            st.session_state.pitchers_df = pitchers
            if pitchers.empty:
                st.warning("Games were found, but probable pitcher/stat data is not available yet.")
            else:
                st.success(f"Loaded {len(pitchers)} probable pitchers.")
    except Exception as e:
        st.error(f"Could not load MLB data: {e}")

pitchers_df = st.session_state.pitchers_df

if not pitchers_df.empty:
    show = pitchers_df.copy()
    show["Pitcher K%"] = (show["Pitcher K%"] * 100).round(1)
    show["Opponent K%"] = (show["Opponent K%"] * 100).round(1)
    show["Expected BF"] = show["Expected BF"].round(1)
    st.dataframe(
        show[["Pitcher", "Opponent", "Pitcher K%", "Opponent K%", "Expected BF"]],
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.subheader("Easy mode — one prop")
    choice = st.selectbox("Pitcher", pitchers_df["Pitcher"].tolist())
    row = pitchers_df[pitchers_df["Pitcher"] == choice].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        line = st.number_input("K line", min_value=0.5, max_value=15.5, value=4.5, step=1.0)
    with c2:
        odds = st.number_input("American odds", value=-110, step=5)

    side = st.segmented_control("Side", ["Over", "Under"], default="Over")
    sims = st.select_slider("Simulations", [10000, 25000, 50000, 100000], value=50000)

    if st.button("2️⃣ Simulate This Prop", type="primary"):
        result, sim = model_one_prop(row, line, int(odds), side, sims)

        p = result["Model Probability"]
        st.success(f'{result["Pitcher"]} {side} {line} Ks')

        a, b = st.columns(2)
        a.metric("Projection", f'{result["Projection"]:.2f} Ks')
        b.metric("Model probability", f"{p:.1%}")

        c, d = st.columns(2)
        c.metric("Edge", f'{result["Edge"]:+.1%}')
        d.metric("EV", f'{result["EV"]:+.1%}')

        if p >= .65:
            st.markdown("### 🟢 65%+ MODEL FLAG")
        elif p >= .60:
            st.markdown("### 🟡 60%+ MODEL FLAG")
        else:
            st.markdown("### ⚪ Below 60%")

        st.caption(
            f'Pitcher K% {row["Pitcher K%"]:.1%} • '
            f'Opponent K% {row["Opponent K%"]:.1%} • '
            f'Expected BF {row["Expected BF"]:.1f}'
        )

        dist = pd.Series(sim["samples"]).value_counts(normalize=True).sort_index() * 100
        st.bar_chart(pd.DataFrame({"Probability %": dist}))

    st.divider()
    st.subheader("🚀 Auto-scan sportsbook K props")
    st.write(
        "Optional: enter a The Odds API key and the app can load current pitcher strikeout "
        "lines/odds and rank them automatically."
    )

    default_key = os.getenv("ODDS_API_KEY", "")
    try:
        if "ODDS_API_KEY" in st.secrets:
            default_key = st.secrets["ODDS_API_KEY"]
    except Exception:
        pass

    api_key = st.text_input(
        "The Odds API key",
        value=default_key,
        type="password",
        help="You can store this later as a Streamlit secret so you don't type it every time.",
    )

    if st.button("3️⃣ Scan Today's K Props", type="primary", disabled=not bool(api_key)):
        try:
            events = get_odds_events(api_key)
            all_props = []

            # Group MLB rows by game so each event prop endpoint is queried only once.
            for (_, away, home), group in pitchers_df.groupby(["GamePk", "Away Team", "Home Team"]):
                event = odds_event_for_game(events, away, home)
                if not event:
                    continue
                event_odds = get_event_pitcher_k_odds(api_key, event["id"])
                props = extract_pitcher_props(event_odds)
                if props.empty:
                    continue

                for _, prop in props.iterrows():
                    matches = group[group["Pitcher"].str.lower() == str(prop["Pitcher"]).lower()]
                    if matches.empty:
                        continue
                    prow = matches.iloc[0]
                    modeled, _ = model_one_prop(
                        prow,
                        float(prop["Line"]),
                        int(prop["Odds"]),
                        str(prop["Side"]),
                        50000,
                    )
                    modeled["Book"] = prop["Book"]
                    all_props.append(modeled)

            if not all_props:
                st.warning(
                    "No matching strikeout props were returned. Props can appear closer to game time, "
                    "and availability varies by sportsbook."
                )
            else:
                result_df = pd.DataFrame(all_props)

                # Best available price for identical pitcher/side/line.
                result_df = result_df.sort_values("Odds", ascending=False)
                result_df = result_df.drop_duplicates(["Pitcher", "Side", "Line"], keep="first")
                result_df = result_df.sort_values(
                    ["Model Probability", "Edge", "EV"],
                    ascending=False,
                )

                display = result_df[
                    ["Pitcher", "Opponent", "Book", "Side", "Line", "Odds",
                     "Projection", "Model Probability", "Edge", "EV"]
                ].copy()

                for col in ["Model Probability", "Edge", "EV"]:
                    display[col] = (display[col] * 100).round(1).astype(str) + "%"
                display["Projection"] = display["Projection"].round(2)

                st.markdown("### Top modeled props")
                st.dataframe(display, hide_index=True, use_container_width=True)

                strong = result_df[
                    (result_df["Model Probability"] >= .65) &
                    (result_df["Edge"] > 0)
                ]
                if not strong.empty:
                    st.markdown("### 🟢 65%+ shortlist")
                    strong_disp = strong[
                        ["Pitcher", "Book", "Side", "Line", "Odds",
                         "Projection", "Model Probability", "Edge", "EV"]
                    ].copy()
                    for col in ["Model Probability", "Edge", "EV"]:
                        strong_disp[col] = (strong_disp[col] * 100).round(1).astype(str) + "%"
                    strong_disp["Projection"] = strong_disp["Projection"].round(2)
                    st.dataframe(strong_disp, hide_index=True, use_container_width=True)
                else:
                    st.info("No positive-edge 65%+ props were found in the returned lines.")

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scan results",
                    data=csv,
                    file_name=f"mlb_k_props_{selected_date.isoformat()}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not scan sportsbook props: {e}")

st.divider()
st.caption(
    "This is a projection model, not a guarantee. The current baseline uses season pitcher K%, "
    "opponent team K%, and pitcher batters-faced per start. Backtesting/calibration should be the next upgrade."
)
