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
import plotly.graph_objects as go

# Initialize Cohere client
co = cohere.Client("YYvexoWfYcfq9dxlWGt0EluWfYwfWwx5fbd6XJ4Aj")  # Replace with your Cohere API key

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
        width: 200px;
    }
    .stSelectbox div {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
        width: 200px;
    }
    .stHeader {
        color: #4CAF50;
    }
    .stTab {
        background-color: #f9f9f9;
        color: #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    .stTab:hover {
        background-color: #4CAF50;
        color: white;
    }
    .stTab.selected {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1y")
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

# Chat with Cohere
def chat_with_cohere(prompt, context=None):
    try:
        if context:
            # Truncate the context to avoid exceeding token limits
            max_tokens = 3000  # Leave room for the prompt and response
            truncated_context = context[:max_tokens]
            prompt = f"{truncated_context}\n\nUser: {prompt}\nAssistant:"

        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            stop_sequences=["\n"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        st.error(f"Error in Cohere chat: {e}")
        return f"Sorry, I couldn't generate a response. Error: {str(e)}"

# Streamlit app
def main():
    st.title("Stock Analysis Dashboard")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Stock Analysis", "Monte Carlo Simulation", "Financial Ratios", "News Sentiment", "Latest News", "Recommendations", "Predictions"]
    choice = st.sidebar.radio("Choose a section", options)

    if choice == "Stock Analysis":
        st.header("Stock Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig.update_layout(title=f"Stock Price for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"Stock Analysis for {stock_ticker}:\n{stock_data.tail().to_string()}"
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

    elif choice == "Monte Carlo Simulation":
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
                            name=f'Simulation {i+1}'
                        ))
                    fig.update_layout(title="Monte Carlo Simulation", xaxis_title="Days", yaxis_title="Price")
                    st.plotly_chart(fig)

                    # Chat
                    st.header("Chat")
                    prompt = st.text_input("Ask me anything...")
                    if st.button("Send"):
                        context = f"Monte Carlo Simulation for {stock_ticker}:\n{stock_data.tail().to_string()}"
                        response = chat_with_cohere(prompt, context)
                        st.write(response)

    elif choice == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                risk_metrics = calculate_risk_metrics(stock_data)
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"Financial Ratios for {stock_ticker}:\n{risk_metrics}"
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

    elif choice == "News Sentiment":
        st.header("News Sentiment")
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
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"News Sentiment for {stock_ticker}:\n{sentiment_counts}"
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

    elif choice == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                for article in articles[:5]:  # Display top 5 articles
                    st.subheader(article.get('title', 'No Title Available'))
                    st.write(article.get('description', 'No Description Available'))
                    st.write(f"Sentiment: {article.get('sentiment', 'N/A')}")

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"Latest News for {stock_ticker}:\n{articles[:5]}"  # Display top 5 articles
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

    elif choice == "Recommendations":
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

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"Recommendations for {stock_ticker}:\n{recommendations}"
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

    elif choice == "Predictions":
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

                    elif model_type == "XGBoost":
                        model = train_xgboost_model(stock_data)
                        predictions = predict_xgboost(model, stock_data)

                    elif model_type == "ARIMA":
                        model = train_arima_model(stock_data)
                        predictions = predict_arima(model)

                    elif model_type == "Prophet":
                        model = train_prophet_model(stock_data)
                        predictions = predict_prophet(model)

                    elif model_type == "Random Forest":
                        model = train_random_forest_model(stock_data)
                        predictions = predict_random_forest(model, stock_data)

                    elif model_type == "Linear Regression":
                        model = train_linear_regression_model(stock_data)
                        predictions = predict_linear_regression(model, stock_data)

                    elif model_type == "Moving Average":
                        predictions = predict_moving_average(stock_data)

                    # Create a date range for the predictions
                    last_date = stock_data.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]  # Exclude the last date

                    # Ensure predictions and future_dates have the same length
                    if len(predictions) != len(future_dates):
                        st.error("Error: Predictions and future_dates length mismatch.")
                    else:
                        # Plot the graph
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))
                        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Data'))
                        fig.update_layout(title=f"Stock Price Predictions for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error in predictions: {e}")

                # Chat
                st.header("Chat")
                prompt = st.text_input("Ask me anything...")
                if st.button("Send"):
                    context = f"Predictions for {stock_ticker}:\nPredictions will be displayed here."
                    response = chat_with_cohere(prompt, context)
                    st.write(response)

if __name__ == "__main__":
    main()
