import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide", page_title="Pepsi Stock Analysis Dashboard")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff; /* White Background */
        color: #000000; /* Black Font Color */
    }

    iframe[title="plotly"] {
        height: 400px !important;
    }

    .stMarkdown h1, h2, h3, h4, h5, h6 {
        color: #000000;
    }

    .metric {
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
    }

    .stMetric label, .stMetric div {
        color: #000000 !important;
    }

    text {
        fill: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_default_data():
    dashboard_dir = Path(__file__).resolve().parent
    repo_root = dashboard_dir.parents[1]
    try:
        pepsi_data = pd.read_csv(repo_root / "Jason_work" / "stock_analysis" / "pepsi.csv")
        sentiment_data = pd.read_csv(dashboard_dir.parent / "sentiment_script" / "pepsi_sentiment_data.csv")
        gross_profit_data = pd.read_csv(repo_root / "Jason_work" / "stock_analysis" / "gross profit(P).csv")
    except FileNotFoundError:
        st.error("Error: Required data files not found. Please check the repository structure.")
        st.stop()

    pepsi_data.rename(columns={"date": "Date", "Close/Last": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)
    sentiment_data.rename(columns={"date": "Date", "sentiment_score_textblob": "Sentiment Score"}, inplace=True)
    gross_profit_data.rename(columns={"Date (Quarter)": "Date", "Price (Billions)": "Gross Profit"}, inplace=True)

    pepsi_data['Date'] = pd.to_datetime(pepsi_data['Date'], errors='coerce')
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], errors='coerce')
    gross_profit_data['Date'] = pd.to_datetime(gross_profit_data['Date'], errors='coerce')

    for col in ['Close', 'Open', 'High', 'Low']:
        pepsi_data[col] = pepsi_data[col].str.replace('$', '', regex=False).astype(float)

    gross_profit_data['Gross Profit'] = gross_profit_data['Gross Profit'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    pepsi_data.dropna(subset=['Date', 'Close'], inplace=True)
    sentiment_data.dropna(subset=['Date', 'Sentiment Score'], inplace=True)

    return pepsi_data, sentiment_data, gross_profit_data

def generate_charts(pepsi_data, sentiment_data, gross_profit_data):
    charts = []

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=pepsi_data['Date'], y=pepsi_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig1.update_layout(title="Historical Stock Price", template="plotly_white", height=400, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig1)

    fig2 = px.histogram(sentiment_data, x="Sentiment Score", nbins=20, title="Sentiment Score Distribution", template="plotly_white", height=400, color_discrete_sequence=["blue"])
    fig2.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig2)

    fig3 = px.bar(
        gross_profit_data,
        x="Date",
        y="Gross Profit",
        title="Gross Profit of Pepsi Over Time",
        labels={"Date": "Quarter", "Gross Profit": "Gross Profit (in Billions)"},
        template="plotly_white",
        height=400,
        color_discrete_sequence=["blue"]
    )
    fig3.update_traces(text=gross_profit_data["Gross Profit"].map("{:.2f}".format), textposition="outside")
    fig3.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig3)

    pepsi_data['Quarter'] = pepsi_data['Date'].dt.to_period('Q')
    quarterly_growth = pepsi_data.groupby('Quarter')['Close'].mean().reset_index()
    quarterly_growth['Quarter'] = quarterly_growth['Quarter'].dt.to_timestamp()
    quarterly_growth['Growth (%)'] = quarterly_growth['Close'].pct_change() * 100

    fig4 = px.bar(
        quarterly_growth,
        x="Quarter",
        y="Growth (%)",
        title="Pepsi Growth Per Quarter",
        labels={"Quarter": "Quarter", "Growth (%)": "Growth (%)"},
        template="plotly_white",
        height=400,
        color="Growth (%)",
        color_continuous_scale=[
            [0.0, "red"],
            [0.5, "lightgrey"],
            [1.0, "green"]
        ]
    )
    fig4.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig4)

    pepsi_data['Month'] = pepsi_data['Date'].dt.to_period('M')
    monthly_avg_price = pepsi_data.groupby('Month')['Close'].mean().reset_index()
    monthly_avg_price['Month'] = monthly_avg_price['Month'].dt.to_timestamp()

    fig5 = px.line(
        monthly_avg_price,
        x="Month",
        y="Close",
        title="Monthly Average Price of Pepsi",
        labels={"Month": "Month"},
        template="plotly_white",
        height=400,
        line_shape='spline',
        color_discrete_sequence=["blue"]
    )
    fig5.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig5)

    sentiment_counts = sentiment_data['Sentiment Score'].apply(
        lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative')
    ).value_counts()

    fig6 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Analysis Breakdown",
        template="plotly_white",
        height=400,
        color=sentiment_counts.index,
        color_discrete_map={
            "Positive": "green",
            "Negative": "red",
            "Neutral": "lightgrey"
        }
    )
    fig6.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig6)

    fig7 = px.line(
        pepsi_data,
        x="Date",
        y="Volume",
        title="Volume Analysis",
        template="plotly_white",
        height=400,
        color_discrete_sequence=["blue"]
    )
    fig7.add_scatter(
        x=pepsi_data['Date'],
        y=pepsi_data['Volume'].rolling(7).mean(),
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='darkblue')
    )
    fig7.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig7)

    fig8 = px.scatter(
        pepsi_data,
        x="Volume",
        y="Close",
        title="Trading Volume vs Stock Price",
        labels={"Volume": "Volume", "Close": "Close Price"},
        template="plotly_white",
        height=400,
        color_discrete_sequence=["blue"]
    )
    fig8.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig8)

    merged_data = pd.merge(pepsi_data, sentiment_data, on="Date", how="inner")
    fig9 = px.scatter(
        merged_data,
        x="Sentiment Score",
        y="Close",
        title="Correlation Between Sentiment and Stock Price",
        labels={"Sentiment Score": "Sentiment Score", "Close": "Stock Price"},
        template="plotly_white",
        height=400,
        color_discrete_sequence=["blue"]
    )
    fig9.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#000000"))
    charts.append(fig9)

    return charts

pepsi_data, sentiment_data, gross_profit_data = load_default_data()
charts = generate_charts(pepsi_data, sentiment_data, gross_profit_data)

st.title("Pepsi Stock Analysis Dashboard")
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("📈 Highest Price", f"${pepsi_data['Close'].max():,.2f}")
col2.metric("📉 Lowest Price", f"${pepsi_data['Close'].min():,.2f}")
col3.metric("📊 Average Price", f"${pepsi_data['Close'].mean():,.2f}")

st.subheader("Additional Insights")
st.write(f"📅 Max Volume Date: {pepsi_data.loc[pepsi_data['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')}")
st.write(f"🧮 Average Sentiment Score: {sentiment_data['Sentiment Score'].mean():.2f}")

chart_columns = st.columns(3)
for i, chart in enumerate(charts):
    with chart_columns[i % 3]:
        st.plotly_chart(chart, use_container_width=True)

st.markdown("---")
st.write("Data Sources: Reddit, Twitter & Yahoo Finance")
st.write("Made by Andrew Ham & Jason Jeong")
