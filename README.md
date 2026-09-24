# Coca-Cola vs. Pepsi Stock & Sentiment Analysis

**Project Period:** January 2026 – May 2026  
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

- `Andrew_work/data/` — source Reddit and Twitter datasets used for the academic analysis
- `Andrew_work/sentiment_script/` — sentiment-analysis and combined-data scripts
- `Andrew_work/dashboards/` — Coca-Cola and Pepsi Streamlit dashboards
- `Jason_work/` — stock analysis, prediction, correlation analysis, and final report

## Setup

Clone the repository, move into the project directory, and install the dependencies:

```bash
pip install -r requirements.txt
```

The sentiment scripts use NLTK's VADER lexicon and download it automatically when needed.

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

The final two scripts create `coca_cola_sentiment_data.csv` and `pepsi_sentiment_data.csv`, which are used by the dashboards.

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

This repository contains an academic team project. The included social-media records are public posts collected for academic analysis and may contain platform-specific post or author identifiers. They are included only to document the original project workflow.

The dashboard presents descriptive and exploratory analysis. It does not provide investment advice or a validated stock-price forecasting model.

No API credentials are required to run the included analysis on the stored datasets.
