
import pandas as pd

# Load the Amsterdam Airbnb dataset
df = pd.read_csv("data/raw/listings.csv.gz", compression="gzip")

# Display basic information
print("Welcome to Analytica AI!")
print("Total listings:", len(df))
print("Total columns:", len(df.columns))

# Display the first five rows
print(df.head())


# Explore the dataset columns
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum().sort_values(ascending=False).head(10))
