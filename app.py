import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Analytica AI | Airbnb Intelligence", page_icon="🏙️", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}div[data-testid='stMetric']{background:#f3f6fb;padding:14px;border-radius:12px}h1{color:#15345a}</style>""", unsafe_allow_html=True)

DATA_PATH = "data/raw/listings.csv.gz"

@st.cache_data(show_spinner="Loading Airbnb data...")
def load_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    if "price" not in data.columns:
        raise ValueError("The dataset does not contain a 'price' column.")
    data["price_eur"] = pd.to_numeric(
        data["price"].astype(str).str.replace(r"[^\d.,]", "", regex=True)
        .str.replace(",", "", regex=False), errors="coerce"
    )
    for col in ["minimum_nights", "number_of_reviews", "availability_365", "accommodates", "bedrooms", "beds", "review_scores_rating"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "neighbourhood_cleansed" in data:
        data["neighborhood"] = data["neighbourhood_cleansed"].fillna("Unknown").astype(str)
    elif "neighbourhood" in data:
        data["neighborhood"] = data["neighbourhood"].fillna("Unknown").astype(str)
    else:
        data["neighborhood"] = "Unknown"
    data["room_type"] = data.get("room_type", pd.Series("Unknown", index=data.index)).fillna("Unknown").astype(str)
    data = data.dropna(subset=["price_eur"])
    data = data[data["price_eur"].between(10, 2000)].copy()  # documented outlier bounds
    return data

st.title("🏙️ Analytica AI")
st.caption("Amsterdam Airbnb Market Intelligence | Inside Airbnb listings snapshot: 16 September 2026")
try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load {DATA_PATH}: {exc}")
    st.info("Place the downloaded listings.csv.gz file in data/raw/ and restart the app.")
    st.stop()

st.sidebar.header("🔎 Explore the market")
rooms = st.sidebar.multiselect("Room type", sorted(df["room_type"].unique()), default=sorted(df["room_type"].unique()))
areas = st.sidebar.multiselect("Neighborhood", sorted(df["neighborhood"].unique()), default=[])
lo, hi = int(df["price_eur"].min()), int(df["price_eur"].max())
price_range = st.sidebar.slider("Nightly price (€)", lo, hi, (lo, min(500, hi)))
filtered = df[df["room_type"].isin(rooms) & df["price_eur"].between(*price_range)]
if areas:
    filtered = filtered[filtered["neighborhood"].isin(areas)]
st.sidebar.caption("Neighborhood: leave blank to include all. Price filters affect dashboard and assistant, not the ML training set.")
if filtered.empty:
    st.warning("No listings match these filters. Try broadening the price range or room types.")
    st.stop()

n_reviews = filtered["number_of_reviews"].sum() if "number_of_reviews" in filtered else 0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Listings", f"{len(filtered):,}")
m2.metric("Avg nightly price", f"€{filtered['price_eur'].mean():,.0f}")
m3.metric("Median nightly price", f"€{filtered['price_eur'].median():,.0f}")
m4.metric("Recorded reviews", f"{int(n_reviews):,}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Price prediction", "💬 Analytics assistant", "📄 Data & methodology"])
with tab1:
    left, right = st.columns(2)
    with left:
        counts = filtered.groupby("neighborhood", as_index=False).size().sort_values("size", ascending=False).head(12)
        fig = px.bar(counts, x="size", y="neighborhood", orientation="h", title="Top neighborhoods by listing count", labels={"size":"Listings", "neighborhood":"Neighborhood"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        medians = filtered.groupby("room_type", as_index=False)["price_eur"].median()
        st.plotly_chart(px.bar(medians, x="room_type", y="price_eur", title="Median price by room type", labels={"price_eur":"Median nightly price (€)", "room_type":"Room type"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(filtered, x="price_eur", nbins=40, title="Nightly price distribution", labels={"price_eur":"Nightly price (€)"}), use_container_width=True)
    with right:
        if {"latitude", "longitude"}.issubset(filtered.columns):
            geo = filtered.dropna(subset=["latitude", "longitude"])
            if len(geo):
                st.subheader("Listing locations")
                st.map(geo[["latitude", "longitude"]].sample(min(1500, len(geo)), random_state=42))
            else:
                st.info("No valid coordinates available.")
        else:
            st.info("Coordinates are not available in this dataset.")

with tab2:
    st.subheader("Predict nightly listing price")
    st.caption("A baseline Random Forest regression model. This is a price estimate, not a guarantee of rental income or occupancy.")
    numeric_candidates = ["accommodates", "bedrooms", "beds", "minimum_nights", "availability_365"]
    categorical_candidates = ["room_type", "neighborhood"]
    numeric = [c for c in numeric_candidates if c in df and df[c].notna().any()]
    categorical = [c for c in categorical_candidates if c in df]
    features = numeric + categorical

    @st.cache_resource(show_spinner="Training model; this may take a moment...")
    def train_model(_data, feature_names, numeric_names, categorical_names):
        sample = _data.sample(n=min(len(_data), 10000), random_state=42)
        X = sample[list(feature_names)]
        y = sample["price_eur"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preprocessing = ColumnTransformer([
            ("numbers", SimpleImputer(strategy="median"), list(numeric_names)),
            ("categories", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), list(categorical_names)),
        ])
        model = Pipeline([
            ("preprocess", preprocessing),
            ("regressor", RandomForestRegressor(n_estimators=75, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions), len(X_train), len(X_test)

    if len(df) < 30 or not numeric:
        st.warning("Insufficient data or numeric features for training.")
    else:
        model, mae, r2, ntrain, ntest = train_model(df, tuple(features), tuple(numeric), tuple(categorical))
        a, b, c = st.columns(3)
        a.metric("Test MAE", f"€{mae:,.1f}")
        b.metric("Test R²", f"{r2:.3f}")
        c.metric("Train / test rows", f"{ntrain:,} / {ntest:,}")
        st.caption("MAE is the average absolute prediction error in euros; lower is better. R² can be negative when a model performs worse than predicting the mean.")
        with st.form("predict"):
            values = {}
            col1, col2 = st.columns(2)
            for i, name in enumerate(numeric):
                with (col1 if i % 2 == 0 else col2):
                    median = float(df[name].median())
                    values[name] = st.number_input(name.replace("_", " ").title(), min_value=0.0, value=max(0.0, median), step=1.0)
            for i, name in enumerate(categorical):
                with (col1 if i % 2 == 0 else col2):
                    values[name] = st.selectbox(name.replace("_", " ").title(), sorted(df[name].dropna().unique()))
            submitted = st.form_submit_button("Estimate nightly price", type="primary")
        if submitted:
            estimate = float(model.predict(pd.DataFrame([values], columns=features))[0])
            st.success(f"Estimated nightly listing price: €{estimate:,.0f}")

with tab3:
    st.subheader("Ask a business question")
    st.caption("Transparent rule-based analytics assistant: calculates answers from the filtered dataset; no external LLM is used.")
    questions = [
        "What is the average nightly price?",
        "Which neighborhoods have the most listings?",
        "Which neighborhoods are most expensive?",
        "How do room types compare in price?",
        "What percentage of listings are available 365 days?",
    ]
    question = st.selectbox("Choose a question", questions)
    if st.button("Analyze", type="primary"):
        if question == questions[0]:
            st.info(f"The average nightly price is €{filtered['price_eur'].mean():,.2f} across {len(filtered):,} filtered listings. The median is €{filtered['price_eur'].median():,.2f}.")
        elif question == questions[1]:
            result = filtered["neighborhood"].value_counts().head(5)
            st.dataframe(result.rename("Listings"), use_container_width=True)
        elif question == questions[2]:
            agg = filtered.groupby("neighborhood")["price_eur"].agg(["median", "count"])
            agg = agg[agg["count"] >= 10].sort_values("median", ascending=False).head(5)
            st.caption("Neighborhoods with at least 10 matching listings; ranked by median nightly price.")
            st.dataframe(agg.rename(columns={"median":"Median price (€)", "count":"Listings"}), use_container_width=True)
        elif question == questions[3]:
            st.dataframe(filtered.groupby("room_type")["price_eur"].agg(["count", "median", "mean"]).round(2), use_container_width=True)
        else:
            if "availability_365" in filtered:
                valid = filtered["availability_365"].dropna()
                pct = (valid == 365).mean() * 100 if len(valid) else float("nan")
                st.info(f"{pct:.1f}% of listings with availability data show 365 available days. This does not measure occupancy or booked nights.")
            else:
                st.warning("Availability data is not present.")

with tab4:
    st.subheader("Data and methodology")
    st.write(f"**Source:** Inside Airbnb, Amsterdam listings, 16 September 2026. **Original rows:** {len(pd.read_csv(DATA_PATH, usecols=['price'])):,}. **Rows after price cleaning:** {len(df):,}.")
    st.write("**Cleaning:** Parse the price field to numeric euros; remove missing prices and prices outside €10–€2,000. These thresholds are project assumptions, not official Inside Airbnb rules.")
    st.write("**Model:** Random Forest regression with an 80/20 random train/test split (seed 42). Numeric missing values use median imputation; categorical missing values use most-frequent imputation and one-hot encoding. The model uses listing attributes and neighborhood, not review scores or post-outcome features.")
    st.write("**Limitations:** Observational listing data, not confirmed transactions. Asking prices may differ from actual paid prices. Single-city, single-date snapshot. Random splits may overestimate future performance. The assistant uses predefined questions, not generative AI.")
    st.dataframe(filtered.head(100), use_container_width=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "amsterdam_airbnb_filtered.csv", "text/csv")

st.caption("Portfolio demonstration • Analytica AI • Source: insideairbnb.com")
