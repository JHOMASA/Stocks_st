import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import cohere
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import time

# Initialize Cohere client
co = cohere.Client("YvexoWfYcfq9dxlWGt0EluWfYwfWwx5fbd6XJ4Aj")  # Replace with your Cohere API key

# Custom CSS for dark theme and glitchy buttons
custom_css = """
<style>
/* Dark theme for the entire app */
body {
  background-color: #1a1a1a;
  color: #ffffff;
  font-family: Arial, sans-serif;
}

/* Glitchy button styles for navigation */
.nav-button-wrapper {
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.2s ease;
  padding: 10px;
  margin: 10px 0;
}

.nav-spiderverse-button {
  position: relative;
  padding: 10px 20px;
  font-size: 18px;
  font-weight: 700;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  background: #fff;
  color: #000;
  text-transform: uppercase;
  letter-spacing: 2px;
  transform-style: preserve-3d;
  transition: all 0.15s ease;
  font-family: Arial, sans-serif;
  text-shadow:
    -1px -1px 0 #000,
    1px -1px 0 #000,
    -1px 1px 0 #000,
    1px 1px 0 #000;
}

.nav-glitch-text {
  position: relative;
  display: inline-block;
}

.nav-glitch-layers {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.nav-glitch-layer {
  position: absolute;
  content: attr(data-text);
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fff;
  border-radius: 25px;
  opacity: 0;
  transition: all 0.15s ease;
}

.nav-layer-1 {
  color: #0ff;
  transform-origin: center;
}

.nav-layer-2 {
  color: #f0f;
  transform-origin: center;
}

.nav-button-wrapper:hover .nav-layer-1 {
  opacity: 1;
  animation: glitchLayer1 0.4s steps(2) infinite;
}

.nav-button-wrapper:hover .nav-layer-2 {
  opacity: 1;
  animation: glitchLayer2 0.4s steps(2) infinite;
}

.nav-button-wrapper:hover .nav-spiderverse-button {
  animation: buttonGlitch 0.3s steps(2) infinite;
  box-shadow:
    0 0 20px rgba(255, 255, 255, 0.5),
    0 0 30px rgba(0, 255, 255, 0.5),
    0 0 40px rgba(255, 0, 255, 0.5);
}

.nav-noise {
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: repeating-radial-gradient(
    circle at 50% 50%,
    transparent 0,
    rgba(0, 0, 0, 0.1) 1px,
    transparent 2px
  );
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s;
  animation: noise 0.2s steps(2) infinite;
}

.nav-button-wrapper:hover .nav-noise {
  opacity: 1;
}

.nav-glitch-slice {
  position: absolute;
  width: 120%;
  height: 5px;
  background: #fff;
  opacity: 0;
  animation: slice 3s linear infinite;
}

@keyframes buttonGlitch {
  0% {
    transform: translate(0) scale(1.1);
  }
  25% {
    transform: translate(-10px, 5px) scale(1.15) skew(-5deg);
  }
  50% {
    transform: translate(10px, -5px) scale(1.1) skew(5deg);
  }
  75% {
    transform: translate(-15px, -5px) scale(1.05) skew(-3deg);
  }
  100% {
    transform: translate(0) scale(1.1);
  }
}

@keyframes glitchLayer1 {
  0% {
    transform: translate(-20px, -10px) scale(1.1) skew(-10deg);
    clip-path: polygon(0 20%, 100% 20%, 100% 50%, 0 50%);
  }
  25% {
    transform: translate(20px, 10px) scale(1.2) skew(10deg);
    clip-path: polygon(0 30%, 100% 30%, 100% 60%, 0 60%);
  }
  50% {
    transform: translate(-15px, 5px) scale(0.9) skew(-5deg);
    clip-path: polygon(0 10%, 100% 10%, 100% 40%, 0 40%);
  }
  75% {
    transform: translate(15px, -5px) scale(1.3) skew(5deg);
    clip-path: polygon(0 40%, 100% 40%, 100% 70%, 0 70%);
  }
  100% {
    transform: translate(-20px, -10px) scale(1.1) skew(-10deg);
    clip-path: polygon(0 20%, 100% 20%, 100% 50%, 0 50%);
  }
}

@keyframes glitchLayer2 {
  0% {
    transform: translate(20px, 10px) scale(1.1) skew(10deg);
    clip-path: polygon(0 50%, 100% 50%, 100% 80%, 0 80%);
  }
  25% {
    transform: translate(-20px, -10px) scale(1.2) skew(-10deg);
    clip-path: polygon(0 60%, 100% 60%, 100% 90%, 0 90%);
  }
  50% {
    transform: translate(15px, -5px) scale(0.9) skew(5deg);
    clip-path: polygon(0 40%, 100% 40%, 100% 70%, 0 70%);
  }
  75% {
    transform: translate(-15px, 5px) scale(1.3) skew(-5deg);
    clip-path: polygon(0 70%, 100% 70%, 100% 100%, 0 100%);
  }
  100% {
    transform: translate(20px, 10px) scale(1.1) skew(10deg);
    clip-path: polygon(0 50%, 100% 50%, 100% 80%, 0 80%);
  }
}

@keyframes noise {
  0% {
    transform: translate(0, 0);
  }
  10% {
    transform: translate(-5%, -5%);
  }
  20% {
    transform: translate(10%, 5%);
  }
  30% {
    transform: translate(-5%, 10%);
  }
  40% {
    transform: translate(15%, -5%);
  }
  50% {
    transform: translate(-10%, 15%);
  }
  60% {
    transform: translate(5%, -10%);
  }
  70% {
    transform: translate(-15%, 5%);
  }
  80% {
    transform: translate(10%, 10%);
  }
  90% {
    transform: translate(-5%, 15%);
  }
  100% {
    transform: translate(0, 0);
  }
}

@keyframes slice {
  0% {
    top: -10%;
    opacity: 0;
  }
  1% {
    opacity: 0.5;
  }
  3% {
    opacity: 0;
  }
  50% {
    top: 110%;
  }
  100% {
    top: 110%;
  }
}
</style>
"""

# HTML for the glitchy navigation buttons
def create_glitchy_button(text):
    return f"""
    <div class="nav-button-wrapper">
      <button class="nav-spiderverse-button">
        {text}
        <div class="nav-glitch-layers">
          <div class="nav-glitch-layer nav-layer-1" data-text="{text}">{text}</div>
          <div class="nav-glitch-layer nav-layer-2" data-text="{text}">{text}</div>
        </div>
        <div class="nav-noise"></div>
        <div class="nav-glitch-slice"></div>
      </button>
    </div>
    """

# Fetch stock data with caching
@st.cache(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1y")
        if stock_data.empty:
            st.error(f"No data found for ticker: {symbol}")
            return pd.DataFrame()
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Fetch news articles with caching
@st.cache(ttl=3600)  # Cache for 1 hour
def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=3f8e6bb1fb72490b835c800afcadd1aa"  # Replace with your NewsAPI key
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()["articles"]
            if not articles:
                st.warning("No news articles found.")
            return articles
        else:
            st.error(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Analyze sentiment of news articles using Cohere
def analyze_news_sentiment(articles):
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        try:
            sentiment = co.classify(text).classifications[0].prediction
            article["sentiment"] = sentiment
            sentiment_counts[sentiment] += 1
        except Exception as e:
            st.error(f"Error analyzing sentiment for article: {text}. Error: {e}")
            article["sentiment"] = "ERROR"
    return sentiment_counts

# Calculate risk metrics
def calculate_risk_metrics(stock_data):
    try:
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        max_drawdown = (stock_data['Close'] / stock_data['Close'].cummax() - 1).min()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
        var_95 = np.percentile(returns, 5)  # Value at Risk (95% confidence)
        return {
            "Volatility": f"{volatility:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "VaR (95%)": f"{var_95:.2%}"
        }
    except Exception as e:
        st.error(f"Error calculating risk metrics: {e}")
        return {}

# Streamlit App
def main():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("Stock Analysis Dashboard")
    st.markdown("---")

    # Sidebar for navigation with glitchy buttons
    with st.sidebar:
        st.title("Navigation")
        tabs = [
            "Stock Analysis", "Monte Carlo Simulation", "Financial Ratios",
            "News Sentiment", "Latest News", "Recommendations", "Predictions", "Chat"
        ]
        selected_tab = st.radio("Choose a section", tabs, label_visibility="collapsed")

        # Display glitchy buttons for navigation
        for tab in tabs:
            st.markdown(create_glitchy_button(tab), unsafe_allow_html=True)

    # Main content based on selected tab
    if selected_tab == "Stock Analysis":
        st.header("Stock Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Fetching stock data..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='#4CAF50')))
                    fig.update_layout(
                        title=f"Stock Price for {stock_ticker}",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        font=dict(color='#ffffff')
                    )
                    st.plotly_chart(fig)

    elif selected_tab == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Running Monte Carlo Simulation..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    simulations = monte_carlo_simulation(stock_data)
                    if simulations is not None:
                        fig = go.Figure()
                        for i in range(min(10, simulations.shape[1])):  # Plot first 10 simulations
                            fig.add_trace(go.Scatter(
                                x=np.arange(simulations.shape[0]),
                                y=simulations[:, i],
                                mode='lines',
                                name=f'Simulation {i+1}',
                                line=dict(color='#FF5722')
                            ))
                        fig.update_layout(
                            title="Monte Carlo Simulation",
                            xaxis_title="Days",
                            yaxis_title="Price",
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            font=dict(color='#ffffff')
                        )
                        st.plotly_chart(fig)

    elif selected_tab == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Calculating financial ratios..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    risk_metrics = calculate_risk_metrics(stock_data)
                    st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

    elif selected_tab == "News Sentiment":
        st.header("News Sentiment Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(stock_ticker)
                if articles:
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_counts = analyze_news_sentiment(articles)
                        if sentiment_counts:
                            fig = go.Figure(data=[go.Bar(
                                x=list(sentiment_counts.keys()),
                                y=list(sentiment_counts.values()),
                                marker_color=['#4CAF50', '#FF5722', '#3F51B5']  # Green for POSITIVE, Red for NEGATIVE, Blue for NEUTRAL
                            )])
                            fig.update_layout(
                                title="News Sentiment Analysis",
                                xaxis_title="Sentiment",
                                yaxis_title="Count",
                                plot_bgcolor='#1a1a1a',
                                paper_bgcolor='#1a1a1a',
                                font=dict(color='#ffffff')
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("No sentiment data available.")

    elif selected_tab == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(stock_ticker)
                if articles:
                    with st.spinner("Analyzing sentiment..."):
                        articles = analyze_news_sentiment(articles)  # Ensure sentiment is attached to articles
                        for article in articles[:5]:  # Display only the first 5 articles
                            st.subheader(article.get('title', 'No Title Available'))
                            st.write(article.get('description', 'No Description Available'))
                            sentiment = article.get('sentiment', 'N/A')
                            sentiment_color = {
                                "POSITIVE": "green",
                                "NEGATIVE": "red",
                                "NEUTRAL": "blue",
                                "ERROR": "gray"
                            }.get(sentiment, "gray")
                            st.write(f"Sentiment: <span style='color:{sentiment_color};'>{sentiment}</span>", unsafe_allow_html=True)

    elif selected_tab == "Recommendations":
        st.header("Recommendations")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        period = st.number_input("Enter Analysis Period (days)", value=30)
        if st.button("Submit"):
            with st.spinner("Generating recommendations..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    financial_ratios = calculate_risk_metrics(stock_data)
                    recommendations = generate_recommendations(stock_data, financial_ratios, period)
                    for recommendation in recommendations:
                        st.write(recommendation)

    elif selected_tab == "Predictions":
        st.header("Predictions")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        model_type = st.selectbox("Select Model", ["LSTM", "XGBoost", "ARIMA", "Prophet", "Random Forest", "Linear Regression", "Moving Average"])
        if st.button("Submit"):
            with st.spinner("Running predictions..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    try:
                        if model_type == "LSTM":
                            if len(stock_data) < 60:
                                st.error("Error: Insufficient data for LSTM (requires at least 60 days).")
                            else:
                                X, y, scaler = prepare_lstm_data(stock_data)
                                model, _ = train_lstm_model(stock_data)
                                predictions = predict_lstm(model, scaler, stock_data)

                        # Create a date range for the predictions
                        last_date = stock_data.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]  # Exclude the last date

                        # Ensure predictions and future_dates have the same length
                        if len(predictions) != len(future_dates):
                            st.error("Error: Predictions and future_dates length mismatch.")
                        else:
                            # Plot the graph
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data', line=dict(color='#4CAF50')))
                            fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Data', line=dict(color='#3F51B5')))
                            fig.update_layout(
                                title=f"Stock Price Predictions for {stock_ticker}",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                plot_bgcolor='#1a1a1a',
                                paper_bgcolor='#1a1a1a',
                                font=dict(color='#ffffff')
                            )
                            st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Error in predictions: {e}")

if __name__ == "__main__":
    main()
