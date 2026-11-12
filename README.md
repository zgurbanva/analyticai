# Analytica AI — Airbnb Market Intelligence

## Project Overview

Analytica AI is an interactive data science application designed to analyze Airbnb rental markets and support data-driven decision-making.

The project uses real Airbnb listing data from Amsterdam, Netherlands, to explore property prices, neighborhood characteristics, availability, and rental market patterns.

## Dataset

- **Source:** Inside Airbnb
- **City:** Amsterdam, Netherlands
- **Dataset date:** September 16, 2026
- **Total listings:** 10,612
- **Original features:** 90

Dataset: https://insideairbnb.com/get-the-data/

## Technologies

- Python
- pandas and NumPy
- Streamlit
- Plotly
- scikit-learn
- Git and GitHub

## Main Features

1. **Interactive dashboard:** Explore Airbnb listing statistics and market patterns.
2. **Data visualization:** Compare prices and neighborhoods through charts.
3. **Machine learning:** Estimate nightly rental prices using property characteristics.
4. **Analytics assistant:** Answer predefined business questions using dataset calculations.
5. **Data exploration:** Investigate listing attributes and market differences.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the detailed Amsterdam `listings.csv.gz` dataset from Inside Airbnb and save it to `data/raw/listings.csv.gz`.

Launch the application:

```bash
python -m streamlit run app.py
```

## Project Objective

Demonstrate practical skills in data preparation, exploratory data analysis, interactive visualization, predictive modeling, and communicating business insights.

## Limitations

Price predictions are estimates, not guaranteed market prices. The analytics assistant provides predefined, data-driven responses rather than unrestricted generative AI.

## Future Improvements

- PostgreSQL integration
- Natural-language-to-SQL using an LLM
- Multi-city rental market comparisons
- Model optimization and deployment
