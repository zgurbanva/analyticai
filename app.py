"""Analytica AI — traveler-focused Airbnb market explorer.
Run: python -m streamlit run app.py
Requires existing Inside Airbnb datasets and optional trained joblib pipelines.
"""
from __future__ import annotations
import html
import re
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Analytica AI | Travel Advisor", page_icon="✈️", layout="wide")
ROOT = Path(__file__).resolve().parent
MARKETS = {
    "Amsterdam": {"file": "listings.csv.gz", "model": "amsterdam.joblib", "currency": "€", "center": (52.37, 4.90)},
    "New York City": {"file": "nyc_listings.csv.gz", "model": "nyc.joblib", "currency": "$", "center": (40.73, -73.94)},
}
PAGES = ["Home Dashboard", "Plan My Trip", "Explore Stays", "Neighborhood Explorer", "Interactive Map", "AI Price Advisor", "AI Travel Assistant"]
FEATURES = ["accommodates", "bedrooms", "bathrooms", "minimum_nights", "number_of_reviews", "room_type"]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#eef2ff;--muted:#a4b2cf;--line:#293757;--violet:#aa91ff;--teal:#5ce3d0}
html,body,[class*="css"],button,input,textarea{font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 75% 38% at 85% -9%,rgba(100,69,185,.24),transparent 75%),radial-gradient(ellipse 45% 40% at 100% 55%,rgba(20,115,127,.08),transparent 75%),#090f1e;color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(175deg,#10182e,#0b1223 62%,#090f1d);border-right:1px solid #2a3550;box-shadow:12px 0 36px #04071344}
[data-testid="stSidebar"]>div:first-child{padding:1.35rem 1rem}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none}
[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:5px}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border:1px solid transparent;border-radius:11px;padding:9px 11px;transition:.2s;background:transparent}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#1a2744;border-color:#344264}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(100deg,#42336d88,#203c5a66);border-color:#7665bb;box-shadow:inset 3px 0 #ad9aff}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:13px;font-weight:650;color:#c1c9e0}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:white}
.block-container{max-width:1480px;padding:1.7rem 2.25rem 3rem}
h1,h2,h3,h4{font-family:Manrope,sans-serif!important;color:#f3f5ff!important;letter-spacing:-.035em}
h3{font-size:1.15rem!important;margin-top:1.2rem!important}
[data-testid="stCaptionContainer"]{color:#a2b2ce!important}
[data-testid="stMetric"]{background:linear-gradient(150deg,#1b2844,#121d33 85%);border:1px solid #35435f;border-radius:17px;padding:19px 20px;min-height:124px;box-shadow:0 12px 28px #03081633,inset 0 1px #ffffff0a}
[data-testid="stMetricLabel"] p{font-size:12px!important;font-weight:650;color:#a6b4d1!important}
[data-testid="stMetricValue"]{font-family:Manrope,sans-serif;color:#f6f4ff!important;font-weight:800;letter-spacing:-.04em}
[data-testid="stForm"]{background:linear-gradient(145deg,#16223b,#111a2e);border:1px solid #34445f;border-radius:19px;padding:20px}
[data-testid="stDataFrame"]{border:1px solid #33425f;border-radius:14px;overflow:hidden}
[data-testid="stPlotlyChart"]{background:linear-gradient(150deg,#151f35,#101a2d);border:1px solid #2b3a56;border-radius:18px;padding:12px 12px 5px;box-shadow:0 10px 25px #03071322}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button{border-radius:11px;min-height:43px;font-weight:700;border-color:#435375}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"]{background:linear-gradient(105deg,#8066eb,#6559c8);border:1px solid #a392ff;color:#fff;box-shadow:0 8px 22px #7058d344}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#a392ff;color:#fff}
[data-testid="stChatMessage"]{background:#151f35;border:1px solid #2c3a57;border-radius:15px;margin:10px 0;padding:12px}
[data-testid="stChatInput"]{border-radius:14px}
[data-testid="stAlert"]{border-radius:13px}
.brand{padding:13px 10px 18px;margin-bottom:10px;border-bottom:1px solid #293854}
.brand-mark{display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;border-radius:11px;margin-right:9px;background:linear-gradient(145deg,#7664db,#253f72);color:#fff;box-shadow:0 0 25px #7961da55}
.brand-title{font:800 22px Manrope,sans-serif;letter-spacing:-.065em;color:#fff}
.brand-title b{color:#b3a0ff}
.brand-sub{margin:9px 0 0 3px;color:#7f92b4;font-size:10px;letter-spacing:.14em;font-weight:800}
.sidebar-label{font-size:10px;color:#8193b5;letter-spacing:.16em;font-weight:800;margin:15px 8px 7px;text-transform:uppercase}
.side-note{margin-top:16px;border:1px solid #2d4260;background:linear-gradient(135deg,#1b2943,#112b35);border-radius:13px;padding:14px;color:#a9bad5;font-size:11px;line-height:1.7}
.side-note strong{color:#f2f5ff;font-size:12px}
.hero{position:relative;overflow:hidden;padding:34px 34px 31px;border:1px solid #455075;border-radius:22px;background:radial-gradient(circle at 95% 15%,#6b54b85e,transparent 30%),radial-gradient(circle at 70% 100%,#166c7952,transparent 40%),linear-gradient(115deg,#252b52,#151d36 72%,#142d3c);box-shadow:0 18px 45px #03071644,inset 0 1px #ffffff19;margin-bottom:25px}
.hero:after{content:'✧';position:absolute;right:6%;top:10%;font-size:145px;line-height:1;color:#ffffff0b;pointer-events:none}
.hero h1{font-size:clamp(27px,3vw,42px);font-weight:800;letter-spacing:-.055em;margin:9px 0 10px;position:relative;z-index:1}
.hero p{font-size:14px;line-height:1.7;color:#c1cbe1;margin:0;max-width:740px;position:relative;z-index:1}
.kicker{color:#c2b2ff;letter-spacing:.17em;font-size:10px;font-weight:800}
.hero-pills{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px;position:relative;z-index:1}
.hero-pill{border:1px solid #7880b566;background:#ffffff12;color:#d9ddff;border-radius:100px;padding:6px 12px;font-size:11px;font-weight:700}
.footer{border-top:1px solid #2a3752;margin-top:40px;padding:18px 0;color:#8396b8;font-size:11px}
@media(max-width:850px){.block-container{padding:1rem .85rem 2rem}.hero{padding:24px 19px}.hero:after{font-size:90px}}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading Airbnb listings…")
def load_market(city):
    path = ROOT / "data" / "raw" / MARKETS[city]["file"]
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    for col in ["price", "name", "room_type", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "listing_url", "bathrooms_text"]:
        if col not in df:
            df[col] = np.nan
    price_text = df["price"].astype(str).str.replace(r"[^0-9.,]", "", regex=True).str.replace(",", "", regex=False)
    df["price_num"] = pd.to_numeric(price_text, errors="coerce")
    for col in ["accommodates", "bedrooms", "bathrooms", "minimum_nights", "number_of_reviews", "review_scores_rating", "latitude", "longitude"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["bathrooms"].isna().all():
        df["bathrooms"] = pd.to_numeric(df["bathrooms_text"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    area_col = "neighbourhood_group_cleansed" if city == "New York City" else "neighbourhood_cleansed"
    df["area"] = df[area_col].fillna("Other").replace("", "Other").astype(str)
    df["room_type"] = df["room_type"].fillna("Unknown").astype(str)
    df["name"] = df["name"].fillna("Unnamed listing").astype(str)
    return df

@st.cache_resource
def load_model(path, modified):
    import joblib
    return joblib.load(path)

def money(value, city):
    return f"{MARKETS[city]['currency']}{value:,.0f}" if pd.notna(value) and np.isfinite(value) else "—"

def positive(df):
    return df.loc[df["price_num"].notna() & (df["price_num"] > 0)]

def grouped(df):
    return (df.groupby("area").agg(listings=("name", "size"), median_price=("price_num", "median"), avg_rating=("review_scores_rating", "mean")).reset_index().sort_values("listings", ascending=False))

def listing_view(df, city, limit=30):
    if df.empty:
        st.info("No matching listings with a valid advertised price. Try a different budget or room type.")
        return
    cols = ["name", "area", "room_type", "price_num", "accommodates", "minimum_nights", "review_scores_rating"]
    view = df[cols].head(limit).copy().rename(columns={"price_num": f"Price/night ({MARKETS[city]['currency']})", "minimum_nights": "Min. nights"})
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button("Download matching listings", df[cols].to_csv(index=False).encode("utf-8-sig"), "matching_stays.csv", "text/csv")
    st.caption("Historical listing snapshots only. These results do not confirm availability, current prices or reservations.")

def travel_matches(df, budget, guests, nights, room, area=None):
    out = positive(df)
    out = out.loc[(out["price_num"] <= budget) & (out["accommodates"] >= guests) & (out["minimum_nights"] <= nights)]
    if room != "Any room type":
        out = out.loc[out["room_type"] == room]
    if area and area != "Any area":
        out = out.loc[out["area"] == area]
    return out.sort_values(["price_num", "number_of_reviews"], ascending=[True, False])


# Chat parsing and evidence are independent of Streamlit UI, so they can be tested.
def detect_city(message):
    text = message.casefold()
    if re.search(r"\b(nyc|new york|manhattan|brooklyn|queens|bronx|nyu.york|nyu.yorka|nyu.yorkda|n[yü]u.york)\b", text):
        return "New York City"
    if re.search(r"\b(amsterdam|ams)\b", text):
        return "Amsterdam"
    return None


def empty_trip_preferences(market_city):
    return dict(city=market_city, nights=None, guests=None, bedrooms=None,
                budget_total=None, budget_night=None, budget_type=None,
                budget_amount=None, currency_mismatch=False,
                entire=False, hotel=False)


def extract_trip_request(message, market_city):
    """Extract only explicit information; do not assume an unqualified amount is nightly."""
    text = message.casefold().replace(",", "")
    result = empty_trip_preferences(market_city)
    nights_match = re.search(r"(\d{1,3})\s*(?:gecəlik|gecə|gecəyə|nights?)\b", text)
    if nights_match:
        result["nights"] = int(nights_match.group(1))
    friends = re.search(r"(?:mən\s+)?(\d+)\s+dostumla", text)
    guests_match = re.search(r"(\d+)\s*(?:nəfər|adam|guests?|people|travelers?)\b", text)
    english_friends = re.search(r"(\d+)\s+friends?\s+and\s+i\b", text)
    if friends or english_friends:
        result["guests"] = int((friends or english_friends).group(1)) + 1
    elif guests_match:
        result["guests"] = int(guests_match.group(1))
    bedrooms = re.search(r"(\d+)\s*(?:yataq otağ\w*|bedrooms?)\b", text)
    if bedrooms:
        result["bedrooms"] = int(bedrooms.group(1))

    amount_match = re.search(
        r"(?P<symbol>[$€])\s*(?P<prefix>\d+(?:\.\d+)?)\b"
        r"|(?P<suffix>\d+(?:\.\d+)?)\s*(?P<unit>usd|eur|dollars?|euros?|avro|€)\b", text
    )
    if amount_match:
        amount = float(amount_match.group("prefix") or amount_match.group("suffix"))
        result["budget_amount"] = amount
        currency = amount_match.group("symbol") or amount_match.group("unit")
        currency = "$" if currency in ("$", "usd", "dollar", "dollars") else "€"
        result["currency_mismatch"] = currency != MARKETS[market_city]["currency"]
        nightly = bool(re.search(r"(?:per\s+night|nightly(?:\s+budget)?|/night|gecəlik\s+büdcə)", text))
        total = bool(re.search(r"\b(?:total|entire\s+budget|whole\s+budget|overall\s+budget|ümumi|cəmi)\b", text))
        # The amount and duration must appear in the same phrase, not just anywhere in the question.
        between = text[amount_match.end():]
        for_nights = bool(re.match(r"\s+for\s+\d{1,3}\s+nights?\b", between))
        if nightly and (total or for_nights):
            result["budget_type"] = "ambiguous"
        elif nightly:
            result["budget_type"] = "nightly"
        elif total or for_nights:
            result["budget_type"] = "total"
        else:
            result["budget_type"] = "ambiguous"

    if re.search(r"(?:bütöv mənzil|bütöv ev|entire (?:home|apartment|place)|whole apartment)", text):
        result["entire"] = True
    if re.search(r"(?:hotel|otel|mehmanxana)", text):
        result["hotel"] = True
    return recalculate_budget(result)


def recalculate_budget(request):
    """Recompute nightly/total budgets after a later message supplies the nights."""
    request = request.copy()
    request["budget_night"] = None
    request["budget_total"] = None
    amount = request.get("budget_amount")
    if amount is None or request.get("currency_mismatch"):
        return request
    if request.get("budget_type") == "nightly":
        request["budget_night"] = amount
        if request.get("nights"):
            request["budget_total"] = amount * request["nights"]
    elif request.get("budget_type") == "total":
        request["budget_total"] = amount
        if request.get("nights"):
            request["budget_night"] = amount / request["nights"]
    return request


def merge_trip_preferences(previous, current):
    """Preserve constraints across messages, updating only explicitly supplied values."""
    if previous.get("city") != current["city"]:
        previous = empty_trip_preferences(current["city"])
    combined = previous.copy()
    for field in ("nights", "guests", "bedrooms"):
        if current[field] is not None:
            combined[field] = current[field]
    if current["budget_amount"] is not None:
        for field in ("budget_amount", "budget_type", "currency_mismatch"):
            combined[field] = current[field]
    if current["entire"]:
        combined["entire"], combined["hotel"] = True, False
    if current["hotel"]:
        combined["hotel"], combined["entire"] = True, False
    return recalculate_budget(combined)


def build_evidence(market_city, request):
    """Compute listing counts and prices in Pandas, never in the LLM."""
    data = load_market(market_city)
    if data is None:
        return "Dataset unavailable; no statistics can be reported.", pd.DataFrame()
    data = positive(data).copy()
    filters = []
    for field in ("guests", "bedrooms", "nights"):
        value = request.get(field)
        if value is None:
            continue
        col, operator = {"guests": ("accommodates", ">="),
                         "bedrooms": ("bedrooms", ">="),
                         "nights": ("minimum_nights", "<=")}[field]
        data = data.loc[data[col] >= value] if operator == ">=" else data.loc[data[col] <= value]
        filters.append(f"{col} {operator} {value}")
    if request.get("entire"):
        data = data.loc[data["room_type"] == "Entire home/apt"]
        filters.append("room_type = Entire home/apt")
    elif request.get("hotel"):
        data = data.loc[data["room_type"] == "Hotel room"]
        filters.append("room_type = Hotel room")
    if data.empty:
        return (f"CITY: {market_city}. FILTERS: {filters}. ZERO historical priced listings satisfy the "
                "property constraints. Do not invent matches or prices."), pd.DataFrame()
    summary = (data.groupby("area", dropna=False)["price_num"]
               .agg(listings="size", median_nightly="median", cheapest_nightly="min").reset_index())
    limit = request.get("budget_night")
    if limit is not None:
        counts = data.loc[data["price_num"] <= limit].groupby("area").size()
        summary["within_budget"] = summary["area"].map(counts).fillna(0).astype(int)
    else:
        summary["within_budget"] = pd.NA
    summary["estimated_stay"] = (summary["median_nightly"] * request["nights"]
                                 if request.get("nights") else np.nan)
    display = summary.sort_values(["median_nightly", "listings"]).head(20).copy()
    for col in ("median_nightly", "cheapest_nightly", "estimated_stay"):
        display[col] = display[col].round(2)
    evidence = (
        f"CITY: {market_city}; CURRENCY: {MARKETS[market_city]['currency']}; "
        f"EXPLICIT AND REMEMBERED CONSTRAINTS: {request}; APPLIED FILTERS: {filters}; "
        f"MATCHING PRICED LISTINGS BEFORE BUDGET: {len(data)}; "
        f"DISPLAYED AREAS: {len(display)} OF {len(summary)}. "
        "The table is a subset when there are more than 20 areas. "
        "within_budget counts listings satisfying property constraints AND nightly budget. "
        "median_nightly is BEFORE budget filtering; estimated_stay excludes fees and taxes. "
        "If budget_type is ambiguous or currency_mismatch is true, ask for clarification; do not "
        "make budget-specific claims. DATA TABLE (authoritative):\n" + display.to_csv(index=False)
    )
    return evidence, display

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-title"><span class="brand-mark">✦</span>Analytica <b>AI</b></div><div class="brand-sub">YOUR PERSONAL TRAVEL INTELLIGENCE</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Discover</div>', unsafe_allow_html=True)
    icons = ["⌂", "✈", "▤", "◎", "⌖", "✧", "◈"]
    page = st.radio("Explore", PAGES, format_func=lambda p: f"{icons[PAGES.index(p)]}   {p}", label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-label">Your destination</div>', unsafe_allow_html=True)
    city = st.selectbox("Destination", list(MARKETS))
    df = load_market(city)
    if df is None:
        st.error(f"Missing data/raw/{MARKETS[city]['file']}. Keep the dataset in the same project folder as app.py.")
        st.stop()
    st.markdown('<div class="side-note"><strong>✦ Travel intelligence</strong><br>Explore historical Airbnb listings and plan smarter. Availability and booking prices are not live.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="hero"><div class="kicker">✦ ANALYTICA AI &nbsp; / &nbsp; YOUR TRAVEL COMPANION</div><h1>{html.escape(page)}</h1><p>Discover your next stay in {html.escape(city)}. Compare neighborhoods, explore real listing data and plan your budget with confidence.</p><div class="hero-pills"><span class="hero-pill">✈ {html.escape(city)}</span><span class="hero-pill">◈ Data-powered insights</span><span class="hero-pill">✧ Smart travel planning</span></div></div>', unsafe_allow_html=True)
priced = positive(df)

if page == "Home Dashboard":
    a,b,c,d = st.columns(4)
    a.metric("Listings", f"{len(df):,}")
    b.metric("Listings with prices", f"{len(priced):,}")
    c.metric("Median nightly price", money(priced["price_num"].median(), city))
    d.metric("Neighborhoods / areas", df["area"].nunique())
    left,right = st.columns(2)
    with left:
        st.subheader("Room types")
        counts = df["room_type"].value_counts().reset_index()
        st.plotly_chart(px.pie(counts, names="room_type", values="count", hole=.65, color_discrete_sequence=["#9888ff", "#49d8c2", "#72a8ff", "#f4c179"]), width="stretch")
    with right:
        st.subheader("Advertised price distribution")
        if not priced.empty:
            cap = priced["price_num"].quantile(.98)
            st.plotly_chart(px.histogram(priced.loc[priced["price_num"] <= cap], x="price_num", nbins=35, labels={"price_num": f"Price/night ({MARKETS[city]['currency']})"}, color_discrete_sequence=["#49d8c2"]), width="stretch")
    st.info("Start with Plan My Trip to find historical listings that match your budget and group size.")

elif page == "Plan My Trip":
    st.subheader("Tell us about your trip")
    with st.form("trip_form"):
        a,b = st.columns(2)
        with a:
            arrival = st.date_input("Arrival date", min_value=date.today(), value=date.today()+timedelta(days=1))
            nights = st.number_input("Nights", min_value=1, max_value=90, value=3)
            guests = st.number_input("Travelers", min_value=1, max_value=30, value=2)
        with b:
            budget = st.number_input(f"Maximum price per night ({MARKETS[city]['currency']})", min_value=1, max_value=10000, value=150)
            room = st.selectbox("Room preference", ["Any room type"] + sorted(df["room_type"].unique().tolist()))
            area = st.selectbox("Preferred area", ["Any area"] + sorted(df["area"].unique().tolist()))
        submitted = st.form_submit_button("Find matching stays", type="primary")
    if submitted:
        matches = travel_matches(df, budget, guests, nights, room, area)
        st.session_state["trip"] = {"city":city,"arrival":str(arrival),"nights":int(nights),"guests":int(guests),"budget":int(budget),"room":room,"area":area}
        a,b,c = st.columns(3)
        a.metric("Historical matches", f"{len(matches):,}")
        b.metric("Maximum stay budget", money(budget*nights, city))
        c.metric("Median matching nightly price", money(matches["price_num"].median(), city))
        st.caption(f"Arrival: {arrival} · {nights} nights. Dates are for budget planning only; availability is not verified.")
        listing_view(matches, city, 40)

elif page == "Explore Stays":
    st.subheader("Find listings that fit your preferences")
    a,b,c = st.columns(3)
    with a:
        budget = st.slider(f"Max price/night ({MARKETS[city]['currency']})", 25, 1500, 200, 25)
    with b:
        guests = st.number_input("Travelers", 1, 30, 2)
    with c:
        room = st.selectbox("Room type", ["Any room type"]+sorted(df["room_type"].unique().tolist()))
    matches = travel_matches(df, budget, guests, 90, room)
    st.metric("Matching historical listings", f"{len(matches):,}")
    listing_view(matches, city, 100)

elif page == "Neighborhood Explorer":
    st.subheader("Compare neighborhoods by advertised prices")
    areas = grouped(priced)
    min_count = st.slider("Minimum priced listings per area", 1, 100, 10)
    areas = areas.loc[areas["listings"] >= min_count]
    if areas.empty:
        st.info("No areas meet this threshold.")
    else:
        st.dataframe(areas.round(1), width="stretch", hide_index=True)
        chart = areas.nlargest(15,"listings").sort_values("median_price")
        st.plotly_chart(px.bar(chart, x="median_price", y="area", orientation="h", color_discrete_sequence=["#9888ff"], labels={"median_price":f"Median nightly price ({MARKETS[city]['currency']})","area":"Area"}), width="stretch")
        st.caption("Cheaper does not necessarily mean safer, more convenient or available for your dates.")

elif page == "Interactive Map":
    st.subheader("Explore where listings are located")
    lat,lon = MARKETS[city]["center"]
    geo = df.dropna(subset=["latitude","longitude"])
    geo = geo.loc[geo["latitude"].between(lat-1.5,lat+1.5) & geo["longitude"].between(lon-1.5,lon+1.5)]
    if geo.empty:
        st.info("No valid coordinates are available.")
    else:
        n = st.slider("Map sample", 100, 5000, 1500, 100)
        sample = geo.sample(n=min(n,len(geo)), random_state=42)
        st.map(sample, latitude="latitude", longitude="longitude", size=6, color="#9888ff")
        st.caption(f"Showing {len(sample):,} historical listings. Markers do not indicate availability.")

elif page == "AI Price Advisor":
    st.subheader("Estimate a nightly listing price")
    path = ROOT / "models" / MARKETS[city]["model"]
    if not path.exists():
        st.warning("Saved prediction model not found. Add the original model file to the models directory.")
    else:
        try:
            model = load_model(str(path), path.stat().st_mtime_ns)
            with st.form("price_advisor"):
                a,b,c = st.columns(3)
                with a:
                    room = st.selectbox("Room type", sorted(df["room_type"].unique().tolist()))
                    guests = st.number_input("Accommodates", 1, 30, 2)
                with b:
                    bedrooms = st.number_input("Bedrooms", 0, 20, 1)
                    bathrooms = st.number_input("Bathrooms", 0.0, 20.0, 1.0, .5)
                with c:
                    nights = st.number_input("Minimum nights", 1, 365, 2)
                    reviews = st.number_input("Number of reviews", 0, 10000, 25)
                submit = st.form_submit_button("Predict price", type="primary")
            if submit:
                x = pd.DataFrame([{"accommodates":guests,"bedrooms":bedrooms,"bathrooms":bathrooms,"minimum_nights":nights,"number_of_reviews":reviews,"room_type":room}], columns=FEATURES)
                prediction = float(model.predict(x)[0])
                if np.isfinite(prediction) and prediction >= 0:
                    st.metric("Estimated advertised price/night", money(prediction, city))
                else:
                    st.warning("Model returned an invalid price estimate.")
            st.caption("This model uses six property features. It does not consider neighborhood, travel dates, or live availability.")
        except Exception as e:
            st.error(f"Could not use the saved model: {e}")

elif page == "AI Travel Assistant":
    st.subheader("Your personal AI travel advisor")
    st.caption("Conversational AI · Grounded in historical Inside Airbnb market data · No live availability")
    try:
        from openai import OpenAI
    except ImportError:
        st.error("API client package is missing. Run: pip install openai")
        st.stop()

    st.markdown("#### 🔑 Connect your own Groq account")
    st.caption("Bring your own API key (BYOK). The app does not save visitor keys to a file or database.")
    entered_key = st.text_input(
        "Your Groq API key", type="password", key="visitor_groq_api_key",
        placeholder="gsk_...", help="Create a key at https://console.groq.com/keys. Never paste it into public chats or GitHub.",
    )
    api_key = entered_key.strip()
    model_name = st.selectbox(
        "AI model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        help="Availability depends on your Groq account and its current limits.",
    )
    if not api_key:
        st.info("Enter your own Groq API key above to start chatting. It is held in this Streamlit session; clear the field to remove it from the widget state.")
        st.markdown("[Create a Groq API key](https://console.groq.com/keys)")
        st.stop()
    st.caption("Privacy: your prompt, relevant market statistics, and key are sent to Groq for inference. Use HTTPS when deploying publicly.")

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    if "ai_chat_city" not in st.session_state:
        st.session_state.ai_chat_city = city
    if "ai_trip_preferences" not in st.session_state:
        st.session_state.ai_trip_preferences = empty_trip_preferences(city)

    top1, top2 = st.columns([3, 1])
    with top1:
        st.caption(f"Current destination: **{st.session_state.ai_chat_city}** · You can also mention NYC or Amsterdam in your question.")
    with top2:
        if st.button("Clear conversation", width="stretch"):
            st.session_state.ai_chat_history = []
            st.session_state.ai_chat_city = city
            st.session_state.ai_trip_preferences = empty_trip_preferences(city)
            st.rerun()

    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask: I have $600 for 4 nights in NYC. Where should I stay?")
    if question:
        detected = detect_city(question)
        if detected:
            st.session_state.ai_chat_city = detected
        target_city = st.session_state.ai_chat_city
        previous = st.session_state.ai_trip_preferences
        # Avoid silently applying old budgets in a different currency after switching markets.
        if previous["city"] != target_city:
            previous = empty_trip_preferences(target_city)
        parsed = extract_trip_request(question, target_city)
        request = merge_trip_preferences(previous, parsed)
        st.session_state.ai_trip_preferences = request
        context, evidence_table = build_evidence(target_city, request)
        st.session_state.ai_chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        system_message = (
            "You are Analytica AI, a friendly, practical travel accommodation advisor. "
            "Respond in the user's language. Answer the actual question, including hotel-room requests. "
            "Use ONLY the supplied market statistics for factual claims about Airbnb listing prices, counts and areas. "
            "Use the precomputed Pandas evidence table as the ONLY source of numeric listing claims. "
            "Ask for missing nights, guests, or budget rather than assuming them. "
            "If budget_type is ambiguous or currency_mismatch is true, ask for clarification before making budget-specific recommendations. "
            "Do not invent individual listings, hotel inventory, neighborhood safety, transport details, real-time prices, "
            "availability, bookings, taxes or fees. If the user asks to book, explain that you cannot book or verify availability. "
            "Distinguish historical advertised nightly prices from actual trip costs. "
            "Use the city in the supplied market data, not an earlier city. "
            "The median price is calculated BEFORE applying budget; within_budget is a count of cheaper matches. "
            "The table may show only a subset of areas. Do not claim it is complete. "
            "A comparison table is rendered separately by the application. "
            "Keep responses clear, helpful and reasonably concise.\n\nVERIFIED PANDAS EVIDENCE:\n" + context
        )
        messages = [{"role": "system", "content": system_message}]
        messages.extend(st.session_state.ai_chat_history[-12:])
        try:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your trip…"):
                    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=30.0, max_retries=1)
                    result = client.chat.completions.create(
                        model=model_name, messages=messages, temperature=0.3, max_tokens=1500,
                    )
                    answer = result.choices[0].message.content or "I couldn't generate a response. Please try again."
                st.markdown(answer)
                if not evidence_table.empty:
                    st.markdown("**Historical area comparison (property-filtered, before budget)**")
                    st.dataframe(evidence_table, width="stretch", hide_index=True)
                    st.caption("within_budget counts listings under your nightly budget. Median prices include listings above budget. At most 20 areas are shown.")
                st.caption("Historical Inside Airbnb data only; availability and final booking costs are not verified.")
            st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status == 401:
                st.error("Groq rejected the API key (401). Check that it is valid and active.")
            elif status == 404:
                st.error("Model not available (404). Try another model above.")
            elif status == 429:
                st.error("Groq rate limit reached (429). Please wait and try again.")
            else:
                st.error(f"AI request failed ({type(error).__name__}). Check your connection and Groq account.")
            st.session_state.ai_chat_history.pop()

st.markdown('<div class="footer">✦ Analytica AI Global Pro &nbsp; · &nbsp; Inside Airbnb historical listings &nbsp; · &nbsp; Explore smarter, travel better. &nbsp; · &nbsp; No live availability or booking guarantees.</div>', unsafe_allow_html=True)
