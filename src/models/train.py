"""Train and evaluate Airbnb price prediction models."""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from evaluation import evaluate_model

DATASETS = {
    "Amsterdam": "data/raw/listings.csv.gz",
    "New York City": "data/raw/nyc_listings.csv.gz",
}

FEATURES = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "minimum_nights",
    "number_of_reviews",
    "room_type",
]


def train_city(city, path):
    df = pd.read_csv(path, low_memory=False)

    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$€,]", "", regex=True),
        errors="coerce",
    )

    for col in FEATURES:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=["price"])
    df = df[df["price"].between(10, 5000)].copy()

    X = df[FEATURES].copy()
    y = df["price"]

    numeric = FEATURES[:-1]
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        ), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), ["room_type"])
    ])

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ))
    ])
    model.fit(X_train, y_train)

    import joblib
    output = Path("models")
    output.mkdir(exist_ok=True)
    filename = "amsterdam.joblib" if city == "Amsterdam" else "nyc.joblib"
    joblib.dump(model, output / filename)
    print(f"Model saved: {output / filename}")

    print(f"\n{city}")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Baseline:", evaluate_model(y_test, baseline.predict(X_test)))
    print("Random Forest:", evaluate_model(y_test, model.predict(X_test)))


if __name__ == "__main__":
    for city, path in DATASETS.items():
        if Path(path).exists():
            train_city(city, path)
        else:
            print(f"Missing dataset: {path}")
