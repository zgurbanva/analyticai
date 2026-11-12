from pathlib import Path
import pandas as pd
import pytest

@pytest.mark.parametrize("city,filename", [
    ("Amsterdam", "listings.csv.gz"),
    ("New York City", "nyc_listings.csv.gz"),
])
def test_airbnb_dataset(city, filename):
    path = Path("data/raw") / filename
    assert path.exists(), f"{city} dataset is missing"

    df = pd.read_csv(path, compression="gzip", low_memory=False)

    assert not df.empty, f"{city} dataset is empty"
    assert "price" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "room_type" in df.columns
