# Coca-Cola vs. Pepsi Stock & Sentiment Analysis

**Project Period:** Fall 2025  
**Project Type:** Academic Team Project

## Overview

This project analyzes public sentiment toward Coca-Cola and Pepsi using social media data and compares sentiment trends with stock market performance. The project combines sentiment analysis, data visualization, and financial data analysis to explore patterns between public opinion and stock behavior.

## My Contributions

- Collected and prepared sentiment data from Reddit and Twitter
- Developed Python scripts to analyze sentiment for Coca-Cola and Pepsi
- Cleaned and organized social media data for analysis
- Created Streamlit dashboards to visualize sentiment and stock trends
- Collaborated with a team member responsible for stock analysis and prediction

## Technologies

Python · Pandas · NLTK/VADER · TextBlob · Matplotlib · Seaborn · Plotly · Streamlit · Jupyter Notebook

## Repository Structure

- `Andrew_work/data/` — local folder for source Reddit and Twitter datasets (raw social-media files are not distributed in the public portfolio)
- `Andrew_work/sentiment_script/` — sentiment-analysis and combined-data scripts
- `Andrew_work/dashboards/` — Coca-Cola and Pepsi Streamlit dashboards
- `Jason_work/` — stock analysis, prediction, correlation analysis, and final report

## Setup

Clone the repository, move into the project directory, and install the dependencies:

```bash
pip install -r requirements.txt
```

The sentiment scripts use NLTK's VADER lexicon and download it automatically when needed.

Raw Reddit and Twitter records are intentionally excluded from the public portfolio because they contain platform-specific post and author identifiers. To reproduce the sentiment pipeline, place appropriately formatted source datasets in `Andrew_work/data/` using the filenames referenced by the scripts.

## Running the Analysis

Run the scripts from the `Andrew_work/sentiment_script` directory in this order.

For Coca-Cola:

```bash
cd Andrew_work/sentiment_script
python coca_cola_sentiment_reddit.py
python coca_cola_sentiment_tweets.py
python coca_cola_sentiment_data.py
```

For Pepsi:

```bash
python pepsi_sentiment_reddit.py
python pepsi_sentiment_tweets.py
python pepsi_sentiment_data.py
```

The final two scripts create `coca_cola_sentiment_data.csv` and `pepsi_sentiment_data.csv`, which are used by the dashboards. The dashboards aggregate sentiment by date before comparing it with daily stock prices so that multiple social-media posts from the same day do not duplicate the stock observation.

## Running the Dashboards

From the repository root:

```bash
streamlit run Andrew_work/dashboards/coca_cola_dashboard.py
```

or

```bash
streamlit run Andrew_work/dashboards/pepsi_dashboard.py
```

## Data Sources

- Reddit
- Twitter
- Yahoo Finance

## Contributors

- **Andrew Ham** — Sentiment analysis, dashboards, and data preparation
- **Jason Jeong** — Stock analysis, predictions, and final report

## Notes

This repository contains an academic team project. Raw social-media records used in the original analysis are intentionally not distributed in the public portfolio. The repository preserves the analysis code and project workflow without publishing platform-specific post or author identifiers.

The dashboard presents descriptive and exploratory analysis. It does not provide investment advice or a validated stock-price forecasting model.

No API credentials are required. Re-running the sentiment pipeline requires local copies of the original-format social-media datasets because those raw records are intentionally excluded from this public portfolio version.
