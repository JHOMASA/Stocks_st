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

/* Glitchy button styles */
.button-wrapper {
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.2s ease;
  padding: 40px;
}

.spiderverse-button {
  position: relative;
  padding: 15px 30px;
  font-size: 24px;
  font-weight: 900;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  background: #fff;
  color: #000;
  text-transform: uppercase;
  letter-spacing: 3px;
  transform-style: preserve-3d;
  transition: all 0.15s ease;
  font-family: Arial, sans-serif;
  text-shadow:
    -1px -1px 0 #000,
    1px -1px 0 #000,
    -1px 1px 0 #000,
    1px 1px 0 #000;
}

.glitch-text {
  position: relative;
  display: inline-block;
}

.glitch-layers {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.glitch-layer {
  position: absolute;
  content: "CLICK ME";
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fff;
  border-radius: 50px;
  opacity: 0;
  transition: all 0.15s ease;
}

.layer-1 {
  color: #0ff;
  transform-origin: center;
}

.layer-2 {
  color: #f0f;
  transform-origin: center;
}

.button-wrapper:hover .layer-1 {
  opacity: 1;
  animation: glitchLayer1 0.4s steps(2) infinite;
}

.button-wrapper:hover .layer-2 {
  opacity: 1;
  animation: glitchLayer2 0.4s steps(2) infinite;
}

.button-wrapper:hover .spiderverse-button {
  animation: buttonGlitch 0.3s steps(2) infinite;
  box-shadow:
    0 0 20px rgba(255, 255, 255, 0.5),
    0 0 30px rgba(0, 255, 255, 0.5),
    0 0 40px rgba(255, 0, 255, 0.5);
}

.noise {
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

.button-wrapper:hover .noise {
  opacity: 1;
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

.glitch-slice {
  position: absolute;
  width: 120%;
  height: 5px;
  background: #fff;
  opacity: 0;
  animation: slice 3s linear infinite;
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

# HTML for the glitchy button
glitch_button_html = """
<div class="button-wrapper">
  <button class="spiderverse-button">
    CLICK ME
    <div class="glitch-layers">
      <div class="glitch-layer layer-1">CLICK ME</div>
      <div class="glitch-layer layer-2">CLICK ME</div>
    </div>
    <div class="noise"></div>
    <div class="glitch-slice"></div>
  </button>
</div>
"""

# Fetch stock data
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

# Fetch news articles
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

# Monte Carlo Simulation
def monte_carlo_simulation(stock_data, num_simulations=1000, days=252):
    try:
        if stock_data.empty:
            raise ValueError("No stock data available for simulation.")

        returns = stock_data['Close'].pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("Insufficient data to calculate returns.")

        mu = returns.mean()
        sigma = returns.std()
        simulations = np.zeros((days, num_simulations))
        S0 = stock_data['Close'].iloc[-1]  # Last observed price

        for i in range(num_simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            simulations[:, i] = S0 * (1 + daily_returns).cumprod()

        return simulations
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {e}")
        return None

# Generate recommendations
def generate_recommendations(stock_data, financial_ratios, period=30):
    recommendations = []

    # Analyze stock trend
    if len(stock_data) >= period:
        trend = "Upward" if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-period] else "Downward"
        if trend == "Upward":
            recommendations.append(f"The stock is in an upward trend over the last {period} days. Consider holding or buying more.")
        elif trend == "Downward":
            recommendations.append(f"The stock is in a downward trend over the last {period} days. Consider selling or setting stop-loss orders.")
    else:
        recommendations.append("Insufficient data to determine the stock trend.")

    # Analyze financial ratios
    benchmarks = {
        "Volatility": "15%",
        "Max Drawdown": "20%",
        "Sharpe Ratio": "1.0",
        "VaR (95%)": "5%"
    }
    for ratio, value in financial_ratios.items():
        if ratio in benchmarks:
            recommendations.append(f"{ratio}: {value} (Benchmark: {benchmarks[ratio]})")

    return recommendations

# Prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    return X, y, scaler

# Train LSTM model
def train_lstm_model(data):
    X, y, scaler = prepare_lstm_data(data)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10)
    return model, scaler

# Predict using LSTM
def predict_lstm(model, scaler, data, look_back=60):
    last_sequence = scaler.transform(data[['Close']].values[-look_back:])
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))  # Reshape for LSTM input
    predictions = []
    for _ in range(30):  # Predict next 30 days
        pred = model.predict(last_sequence)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[pred]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Streamlit App
def main():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("Stock Analysis Dashboard")
    st.markdown("---")

    # Glitchy button
    st.markdown(glitch_button_html, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        tab = st.radio(
            "Choose a section",
            ["Stock Analysis", "Monte Carlo Simulation", "Financial Ratios", "News Sentiment", "Latest News", "Recommendations", "Predictions", "Chat"]
        )

    if tab == "Stock Analysis":
        st.header("Stock Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
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

    elif tab == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
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

    elif tab == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                risk_metrics = calculate_risk_metrics(stock_data)
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

    elif tab == "News Sentiment":
        st.header("News Sentiment Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                sentiment_counts = analyze_news_sentiment(articles)
                fig = go.Figure(data=[go.Bar(
                    x=list(sentiment_counts.keys()),
                    y=list(sentiment_counts.values())
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

    elif tab == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                for article in articles[:5]:
                    st.subheader(article.get('title', 'No Title Available'))
                    st.write(article.get('description', 'No Description Available'))
                    st.write(f"Sentiment: {article.get('sentiment', 'N/A')}")

    elif tab == "Recommendations":
        st.header("Recommendations")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        period = st.number_input("Enter Analysis Period (days)", value=30)
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                financial_ratios = calculate_risk_metrics(stock_data)
                recommendations = generate_recommendations(stock_data, financial_ratios, period)
                for recommendation in recommendations:
                    st.write(recommendation)

    elif tab == "Predictions":
        st.header("Predictions")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        model_type = st.selectbox("Select Model", ["LSTM", "XGBoost", "ARIMA", "Prophet", "Random Forest", "Linear Regression", "Moving Average"])
        if st.button("Submit"):
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
