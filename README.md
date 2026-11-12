<div align="center">

# ✈️ Analytica AI Global Pro

### Airbnb Market Intelligence & AI-Powered Travel Planning

Explore historical Airbnb listings, compare destinations, estimate accommodation costs, and plan smarter trips with data-driven insights.

**Python · Streamlit · Pandas · Plotly · Scikit-learn · Groq API**

</div>

---

## Overview

**Analytica AI Global Pro** is an interactive travel-planning and Airbnb market analytics application built with Python and Streamlit. It combines exploratory data analysis, interactive visualizations, optional machine-learning price predictions, and a conversational AI travel assistant.

The application currently supports **Amsterdam, Netherlands**, and **New York City, United States**, using historical listing snapshots from [Inside Airbnb](https://insideairbnb.com/get-the-data/).

The goal is to help travelers understand accommodation price patterns and compare potential stays—not to replace a live booking service.

## Features

| Page | What it offers |
| --- | --- |
| **Home Dashboard** | Listing counts, median advertised nightly prices, room-type breakdowns, and price distributions. |
| **Plan My Trip** | Filter historical listings by destination, nightly budget, group size, minimum-stay requirement, room type, and area. |
| **Explore Stays** | Browse and export listings that match selected price, guest, and room-type preferences. |
| **Neighborhood Explorer** | Compare areas by listing volume, median advertised price, and average review rating. |
| **Interactive Map** | Explore a sample of historical listing locations on a map. |
| **AI Price Advisor** | Estimate nightly listing prices using locally saved machine-learning pipelines, when available. |
| **AI Travel Assistant** | Ask travel questions in natural language through Groq-hosted language models, with historical market data supplied as context. |

### AI Travel Assistant

- Supports conversational questions, including multilingual queries.
- Uses the **bring-your-own-key (BYOK)** approach: visitors enter their own Groq API key in the app.
- Allows selection from configured Groq model options, subject to account access and provider availability.
- Uses historical Airbnb data to provide market context; it **cannot confirm live availability, bookings, or final checkout prices**.
- Sends user prompts and relevant market context to Groq for inference.

> **Important:** AI-generated explanations should not be treated as independently verified listing statistics. For precise comparisons, verify that the requested filters have been applied to the underlying dataset. The app is not a reservation platform.

## Data

**Source:** [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/)

| Destination | Expected dataset path | Currency |
| --- | --- | --- |
| Amsterdam | `data/raw/listings.csv.gz` | EUR (€) |
| New York City | `data/raw/nyc_listings.csv.gz` | USD ($) |

The original Amsterdam analysis used a snapshot dated **September 16, 2026**, containing **10,612 listings** and **90 original features**. Counts and statistics displayed by the application depend on the dataset files installed locally or deployed with the app.

The application derives a numeric nightly price and uses fields such as room type, accommodates, bedrooms, minimum nights, neighborhood, reviews, and geographic coordinates when available.

## Technology Stack

- **Python** — application logic and data processing
- **Streamlit** — interactive web interface
- **Pandas and NumPy** — dataset preparation, filtering, and statistics
- **Plotly** — interactive charts
- **Scikit-learn / Joblib** — saved price-prediction pipelines
- **Groq API (OpenAI-compatible client)** — generative travel assistant
- **Git and GitHub** — version control and collaboration

## Project Structure

```text
analyticai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/
│       ├── listings.csv.gz
│       └── nyc_listings.csv.gz
└── models/
    ├── amsterdam.joblib
    └── nyc.joblib
```

The data files and trained model files must be supplied separately if they are not included in the repository. Model files are optional for the other pages, but required to enable predictions in **AI Price Advisor**.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the placeholder repository URL and directory with your actual GitHub details.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate` instead.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The dependencies must include the `openai` Python package for the Groq-compatible API client.

### 4. Add the datasets

Download the appropriate detailed `listings.csv.gz` files from [Inside Airbnb](https://insideairbnb.com/get-the-data/) and save them as:

```text
data/raw/listings.csv.gz
data/raw/nyc_listings.csv.gz
```

Ensure the files contain the expected city data. Add trained `.joblib` pipelines to `models/` if you want to use the AI Price Advisor.

### 5. Run the app

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal, typically `http://localhost:8501`.

### 6. Connect Groq (optional)

To use **AI Travel Assistant**:

1. Create an API key at [Groq Console](https://console.groq.com/keys).
2. Open **AI Travel Assistant** in the app.
3. Paste your key into the password-protected input field.
4. Select an available model and start chatting.

The other analytics pages do not require a Groq API key.

## Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push `app.py`, `requirements.txt`, and the necessary accessible datasets to a GitHub repository.
2. Confirm that the repository includes any model artifacts required for price predictions.
3. In Streamlit Community Cloud, create an app from the GitHub repository and select `app.py` as the entry point.
4. Deploy and test both destinations, charts, predictions, and the BYOK chatbot.

**Deployment note:** Large datasets or model artifacts may exceed repository or hosting limits. If necessary, use an appropriate external data/artifact delivery strategy. The app must be adapted to retrieve files from that location before deployment.

## Security & Privacy

- **Never commit Groq API keys**, `.env` files, or `.streamlit/secrets.toml` to GitHub.
- The app is designed to use the visitor's Groq API key rather than a shared owner key.
- The key is entered through a password field; this does not eliminate all deployment or provider-side privacy risks.
- Prompts and relevant market context are sent to Groq to generate responses.
- Use HTTPS in production and follow Groq's applicable data handling policies.

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
.streamlit/secrets.toml
__pycache__/
.DS_Store
```

## Limitations

- **Historical data:** Listing snapshots do not represent live inventory, confirmed availability, or real-time prices.
- **Incomplete trip costs:** Advertised nightly prices may exclude taxes, cleaning fees, service fees, and other charges.
- **Prediction uncertainty:** Machine-learning outputs are estimates, not guaranteed prices. The current model uses a limited set of listing features.
- **AI reliability:** Language-model answers can be incomplete or inaccurate; dataset calculations and user-selected filters should be checked independently.
- **City and currency differences:** Amsterdam and NYC prices are reported in different currencies. They should not be combined without an explicit exchange rate.
- **Data coverage:** Results depend on the completeness, quality, and snapshot dates of the supplied datasets.

## Future Roadmap

- More destinations and standardized cross-city comparisons
- Stronger constraint-aware travel recommendations and deterministic comparison tables
- Better evaluation and optimization of price-prediction models
- Improved handling of taxes, fees, and currency conversions when reliable data is available
- Additional deployment, testing, and accessibility improvements

## Data Attribution

Listing data is sourced from [Inside Airbnb](https://insideairbnb.com/). Refer to the source website for dataset documentation, attribution requirements, and usage conditions.

---

<div align="center">

**Analytica AI Global Pro** · Explore smarter. Travel better.

*An educational data science and travel analytics project.*

</div>
