import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Coca Cola Stock Analysis Dashboard")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #2a2a2a; /* Dark Gray */
        color: #ffffff;
    }

    iframe[title="plotly"] {
        height: 400px !important;
    }

    .stMarkdown h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }

    .metric {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
    }

    .stMetric label, .stMetric div {
        color: #ffffff !important;
    }

    text {
        fill: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_default_data():
    try:
        coca_data = pd.read_csv("/Users/andrewham/Downloads/Cola.csv")
        sentiment_data = pd.read_csv("/Users/andrewham/Downloads/coca_cola_sentiment_data.csv")
        gross_profit_data = pd.read_csv("/Users/andrewham/Downloads/Gross Profit(KO).csv")
    except FileNotFoundError:
        st.error("Error: Required data files not found. Please check the file paths.")
        st.stop()

    coca_data.rename(columns={"date": "Date", "Close/Last": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)
    sentiment_data.rename(columns={"date": "Date", "sentiment_score_textblob": "Sentiment Score"}, inplace=True)
    gross_profit_data.rename(columns={"Date (Quarter)": "Date", "Price (Millions)": "Gross Profit"}, inplace=True)

    coca_data['Date'] = pd.to_datetime(coca_data['Date'], errors='coerce')
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], errors='coerce')
    gross_profit_data['Date'] = pd.to_datetime(gross_profit_data['Date'], errors='coerce')

    for col in ['Close', 'Open', 'High', 'Low']:
        coca_data[col] = coca_data[col].str.replace('$', '', regex=False).astype(float)

    gross_profit_data['Gross Profit'] = gross_profit_data['Gross Profit'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    coca_data.dropna(subset=['Date', 'Close'], inplace=True)
    sentiment_data.dropna(subset=['Date', 'Sentiment Score'], inplace=True)

    return coca_data, sentiment_data, gross_profit_data

def generate_forecast(coca_data):
    future_dates = pd.date_range(start=coca_data['Date'].max(), periods=365, freq='D')[1:]
    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": coca_data['Close'].iloc[-1] * (1 + 0.001 * pd.Series(range(len(future_dates))))
    })
    return forecast

def generate_charts(coca_data, sentiment_data, gross_profit_data):
    charts = []

    forecast = generate_forecast(coca_data)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=coca_data['Date'], y=coca_data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Forecast'], mode='lines', name='Forecast', line=dict(color='darkred')))
    fig1.update_layout(title="Stock Price Insights with Forecast", template="plotly_dark", height=400, plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig1)

    fig2 = px.histogram(sentiment_data, x="Sentiment Score", nbins=20, title="Sentiment Score Distribution", template="plotly_dark", height=400, color_discrete_sequence=["red"])
    fig2.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig2)

    fig3 = px.bar(
        gross_profit_data,
        x="Date",
        y="Gross Profit",
        title="Gross Profit of Coca Cola Over Time",
        labels={"Date": "Quarter", "Gross Profit": "Gross Profit (in Millions)"},
        template="plotly_dark",
        height=400,
        color_discrete_sequence=["red"]
    )
    fig3.update_traces(text=gross_profit_data["Gross Profit"].map("{:.2f}".format), textposition="outside")
    fig3.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig3)

    coca_data['Quarter'] = coca_data['Date'].dt.to_period('Q')
    quarterly_growth = coca_data.groupby('Quarter')['Close'].mean().reset_index()
    quarterly_growth['Quarter'] = quarterly_growth['Quarter'].dt.to_timestamp()
    quarterly_growth['Growth (%)'] = quarterly_growth['Close'].pct_change() * 100

    fig4 = px.bar(
        quarterly_growth,
        x="Quarter",
        y="Growth (%)",
        title="Coca Cola Growth Per Quarter",
        labels={"Quarter": "Quarter", "Growth (%)": "Growth (%)"},
        template="plotly_dark",
        height=400,
        color="Growth (%)",
        color_continuous_scale=[
            [0.0, "red"],
            [0.5, "lightgrey"],
            [1.0, "green"]
        ]
    )
    fig4.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig4)

    coca_data['Year'] = coca_data['Date'].dt.to_period('M')
    monthly_avg_price = coca_data.groupby('Year')['Close'].mean().reset_index()
    monthly_avg_price['Year'] = monthly_avg_price['Year'].dt.to_timestamp()

    fig5 = px.line(
        monthly_avg_price,
        x="Year",
        y="Close",
        title="Yearly Price of Coca Cola",
        labels={"Year": "Year"},
        template="plotly_dark",
        height=400,
        line_shape='spline',
        color_discrete_sequence=["red"]
    )
    fig5.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig5)

    sentiment_counts = sentiment_data['Sentiment Score'].apply(
        lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative')
    ).value_counts()

    fig6 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Analysis Breakdown",
        template="plotly_dark",
        height=400,
        color=sentiment_counts.index,
        color_discrete_map={
            "Positive": "green",
            "Negative": "red",
            "Neutral": "lightgrey"
        }
    )
    fig6.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig6)

    fig7 = px.line(
        coca_data,
        x="Date",
        y="Volume",
        title="Volume Analysis",
        template="plotly_dark",
        height=400,
        color_discrete_sequence=["red"]
    )
    fig7.add_scatter(
        x=coca_data['Date'],
        y=coca_data['Volume'].rolling(7).mean(),
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='darkred')
    )
    fig7.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig7)

    fig8 = px.scatter(
        coca_data,
        x="Volume",
        y="Close",
        title="Sentiment vs Volume Correlation",
        labels={"Volume": "Volume", "Close": "Close Price"},
        template="plotly_dark",
        height=400,
        color_discrete_sequence=["red"]
    )
    fig8.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig8)

    merged_data = pd.merge(coca_data, sentiment_data, on="Date", how="inner")
    fig9 = px.scatter(
        merged_data,
        x="Sentiment Score",
        y="Close",
        title="Correlation Between Sentiment and Stock Price",
        labels={"Sentiment Score": "Sentiment Score", "Close": "Stock Price"},
        template="plotly_dark",
        height=400,
        color_discrete_sequence=["red"]
    )
    fig9.update_layout(plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a", font=dict(color="#ffffff"))
    charts.append(fig9)

    return charts

coca_data, sentiment_data, gross_profit_data = load_default_data()
charts = generate_charts(coca_data, sentiment_data, gross_profit_data)

st.title("Coca Cola Stock Analysis Dashboard")
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("📈 Highest Price", f"${coca_data['Close'].max():,.2f}")
col2.metric("📉 Lowest Price", f"${coca_data['Close'].min():,.2f}")
col3.metric("📊 Average Price", f"${coca_data['Close'].mean():,.2f}")

st.subheader("Additional Insights")
st.write(f"📅 Max Volume Date: {coca_data.loc[coca_data['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')}")
st.write(f"🧮 Average Sentiment Score: {sentiment_data['Sentiment Score'].mean():.2f}")

chart_columns = st.columns(3)
for i, chart in enumerate(charts):
    with chart_columns[i % 3]:
        st.plotly_chart(chart, use_container_width=True)

st.markdown("---")
st.write("Data Source: Reddit, Twitter & Yahoo")
st.write("Made by Andrew Ham & Jason Jeong")
