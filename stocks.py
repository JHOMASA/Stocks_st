import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob  # Fallback sentiment analysis
import plotly.graph_objects as go
import cohere

# Initialize Cohere client
co = cohere.Client("gpWuZqkXdfhfbYkjLlyRnc5x2rj0ml1IqfULfjt0")  # Replace with your Cohere API key

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

# Analyze sentiment of news articles using TextBlob
def analyze_news_sentiment(articles):
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Error": 0}
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        sentiment = "Error"  # Default value

        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "Positive"
            elif polarity < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        except Exception as e:
            st.warning(f"Sentiment analysis failed. Error: {e}")
            sentiment = "Error"

        article["sentiment"] = sentiment
        sentiment_counts[sentiment] += 1
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

# Chat interface
def chat_interface():
    st.markdown(
        """
        <style>
        .chatbox {
            width: 100%;
            height: 400px;
            max-height: 400px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 0 4px rgba(0,0,0,.14),0 4px 8px rgba(0,0,0,.28);
        }
        .chat-window {
            flex: auto;
            max-height: calc(100% - 60px);
            background: #2f323b;
            overflow: auto;
            padding: 10px;
        }
        .chat-input {
            flex: 0 0 auto;
            height: 60px;
            background: #40434e;
            border-top: 1px solid #2671ff;
            box-shadow: 0 0 4px rgba(0,0,0,.14),0 4px 8px rgba(0,0,0,.28);
            display: flex;
            align-items: center;
            padding: 0 10px;
        }
        .chat-input input {
            height: 40px;
            line-height: 40px;
            outline: 0 none;
            border: none;
            width: calc(100% - 60px);
            color: white;
            text-indent: 10px;
            font-size: 12pt;
            padding: 0;
            background: #40434e;
        }
        .chat-input button {
            float: right;
            outline: 0 none;
            border: none;
            background: rgba(255,255,255,.25);
            height: 40px;
            width: 40px;
            border-radius: 50%;
            padding: 2px 0 0 0;
            margin: 10px;
            transition: all 0.15s ease-in-out;
        }
        .msg-container {
            position: relative;
            display: inline-block;
            width: 100%;
            margin: 0 0 10px 0;
            padding: 0;
        }
        .msg-box {
            display: flex;
            background: #5b5e6c;
            padding: 10px 10px 0 10px;
            border-radius: 0 6px 6px 0;
            max-width: 80%;
            width: auto;
            float: left;
            box-shadow: 0 0 2px rgba(0,0,0,.12),0 2px 4px rgba(0,0,0,.24);
        }
        .msg-self .msg-box {
            border-radius: 6px 0 0 6px;
            background: #2671ff;
            float: right;
        }
        .msg {
            display: inline-block;
            font-size: 11pt;
            line-height: 13pt;
            color: rgba(255,255,255,.7);
            margin: 0 0 4px 0;
        }
        .timestamp {
            color: rgba(0,0,0,.38);
            font-size: 8pt;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chatbox">
            <div class="chat-window" id="chat-window">
                <div class="msg-container msg-remote">
                    <div class="msg-box">
                        <div class="flr">
                            <div class="messages">
                                <p class="msg">Hello! How can I assist you today?</p>
                            </div>
                            <span class="timestamp"><span class="username">Bot</span>&bull;<span class="posttime">Now</span></span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="chat-input" placeholder="Type a message" />
                <button onclick="sendMessage()">
                    <svg style="width:24px;height:24px" viewBox="0 0 24 24"><path fill="rgba(0,0,0,.38)" d="M17,12L12,17V14H8V10H12V7L17,12M21,16.5C21,16.88 20.79,17.21 20.47,17.38L12.57,21.82C12.41,21.94 12.21,22 12,22C11.79,22 11.59,21.94 11.43,21.82L3.53,17.38C3.21,17.21 3,16.88 3,16.5V7.5C3,7.12 3.21,6.79 3.53,6.62L11.43,2.18C11.59,2.06 11.79,2 12,2C12.21,2 12.41,2.06 12.57,2.18L20.47,6.62C20.79,6.79 21,7.12 21,7.5V16.5M12,4.15L5,8.09V15.91L12,19.85L19,15.91V8.09L12,4.15Z" /></svg>
                </button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Handle chat input
    user_input = st.text_input("Type a message", key="chat_input", on_change=send_message)

# Send message to Cohere and display response
def send_message():
    user_input = st.session_state.chat_input
    if user_input:
        # Generate response using Cohere
        response = co.generate(
            model="command",
            prompt=user_input,
            max_tokens=100,
            temperature=0.7,
        )
        st.session_state.chat_history.append({"user": user_input, "bot": response.generations[0].text})

# Streamlit app
def main():
    st.title("Stock Analysis Chatbot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
                st.write("### Stock Data")
                st.write(stock_data)

                # Plot stock data
                st.write("### Stock Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig.update_layout(title=f"Stock Price for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

    elif choice == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                simulations = monte_carlo_simulation(stock_data)
                if simulations is not None:
                    st.write("### Monte Carlo Simulation Results")
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

    elif choice == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                risk_metrics = calculate_risk_metrics(stock_data)
                st.write("### Financial Ratios")
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

    elif choice == "News Sentiment":
        st.header("News Sentiment Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                sentiment_counts = analyze_news_sentiment(articles)
                st.write("### Sentiment Summary")
                st.write(f"Positive: {sentiment_counts['Positive']}")
                st.write(f"Negative: {sentiment_counts['Negative']}")
                st.write(f"Neutral: {sentiment_counts['Neutral']}")
                st.write(f"Errors: {sentiment_counts['Error']}")

                # Plot sentiment distribution
                st.write("### Sentiment Distribution")
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
            else:
                st.warning("No news articles found for this stock ticker.")

    elif choice == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                st.write("### Top 5 News Articles")
                for article in articles[:5]:  # Display top 5 articles
                    st.write(f"**Title:** {article.get('title', 'No Title Available')}")
                    st.write(f"**Description:** {article.get('description', 'No Description Available')}")
                    st.write(f"**Source:** {article.get('source', {}).get('name', 'N/A')}")
                    st.write(f"**Published At:** {article.get('publishedAt', 'N/A')}")
                    st.write("---")
            else:
                st.warning("No news articles found for this stock ticker.")

    elif choice == "Recommendations":
        st.header("Recommendations")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        period = st.number_input("Enter Analysis Period (days)", value=30)
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                financial_ratios = calculate_risk_metrics(stock_data)
                recommendations = generate_recommendations(stock_data, financial_ratios, period)
                st.write("### Recommendations")
                for recommendation in recommendations:
                    st.write(recommendation)

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
                            model, scaler = train_lstm_model(stock_data)
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

    # Chat interface
    chat_interface()

if __name__ == "__main__":
    main()

