import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import cohere

# Initialize Cohere client
co = cohere.Client("gpWuZqkXdfhfbYkjLlyRnc5x2rj0ml1IqfULfjt0")  # Replace with your Cohere API key

# Cache expensive operations
@st.cache_data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1y")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["articles"]
        else:
            st.error(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to generate responses using Cohere
def generate_response(prompt):
    response = co.generate(
        model="command",  # Use Cohere's Command model
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
    )
    return response.generations[0].text

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to add messages to chat history
def add_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

# Function to display chat history
def display_chat():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**Bot:** {chat['message']}")

# Chat Interface
def chat_interface(choice):
    chat_html = f"""
    <style>
        body {{
            height: 100%;
            width: 100%;
            margin: 0;
            padding: 0;
            font-family: sans-serif;
            background: #1c1e22;
        }}
        .chatbox {{
            width: 100%;
            height: 500px;
            max-height: 500px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 0 4px rgba(0,0,0,.14),0 4px 8px rgba(0,0,0,.28);
        }}
        .chat-window {{
            flex: auto;
            max-height: calc(100% - 60px);
            background: #2f323b;
            overflow: auto;
            padding: 10px;
        }}
        .chat-input {{
            flex: 0 0 auto;
            height: 60px;
            background: #40434e;
            border-top: 1px solid #2671ff;
            box-shadow: 0 0 4px rgba(0,0,0,.14),0 4px 8px rgba(0,0,0,.28);
        }}
        .chat-input input {{
            height: 59px;
            line-height: 60px;
            outline: 0 none;
            border: none;
            width: calc(100% - 60px);
            color: white;
            text-indent: 10px;
            font-size: 12pt;
            padding: 0;
            background: #40434e;
        }}
        .chat-input button {{
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
        }}
        .msg-container {{
            position: relative;
            display: inline-block;
            width: 100%;
            margin: 0 0 10px 0;
            padding: 0;
            animation: fadeIn 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .msg-box {{
            display: flex;
            background: #5b5e6c;
            padding: 10px 10px 0 10px;
            border-radius: 0 6px 6px 0;
            max-width: 80%;
            width: auto;
            float: left;
            box-shadow: 0 0 2px rgba(0,0,0,.12),0 2px 4px rgba(0,0,0,.24);
        }}
        .user-img {{
            display: inline-block;
            border-radius: 50%;
            height: 40px;
            width: 40px;
            background: #2671ff;
            margin: 0 10px 10px 0;
        }}
        .flr {{
            flex: 1 0 auto;
            display: flex;
            flex-direction: column;
            width: calc(100% - 50px);
        }}
        .messages {{
            flex: 1 0 auto;
        }}
        .msg {{
            display: inline-block;
            font-size: 11pt;
            line-height: 13pt;
            color: rgba(255,255,255,.7);
            margin: 0 0 4px 0;
        }}
        .msg:first-of-type {{
            margin-top: 8px;
        }}
        .timestamp {{
            color: rgba(0,0,0,.38);
            font-size: 8pt;
            margin-bottom: 10px;
        }}
        .username {{
            margin-right: 3px;
        }}
        .posttime {{
            margin-left: 3px;
        }}
        .msg-self .msg-box {{
            border-radius: 6px 0 0 6px;
            background: #2671ff;
            float: right;
        }}
        .msg-self .user-img {{
            margin: 0 0 10px 10px;
        }}
        .msg-self .msg {{
            text-align: right;
        }}
        .msg-self .timestamp {{
            text-align: right;
        }}
    </style>
    <div class="chatbox">
        <div class="chat-window" id="chat-window">
            <div class="msg-container msg-remote">
                <div class="msg-box">
                    <img class="user-img" src="https://gravatar.com/avatar/00034587632094500000000000000000?d=retro" />
                    <div class="flr">
                        <div class="messages">
                            <p class="msg">Hello! You selected <strong>{choice}</strong>. How can I assist you?</p>
                        </div>
                        <span class="timestamp"><span class="username">Bot</span>&bull;<span class="posttime">Now</span></span>
                    </div>
                </div>
            </div>
        </div>
        <form class="chat-input" onsubmit="sendMessage(event, '{choice}')">
            <input type="text" id="chat-input" autocomplete="on" placeholder="Type a message" />
            <button type="submit">
                <svg style="width:24px;height:24px" viewBox="0 0 24 24"><path fill="rgba(0,0,0,.38)" d="M17,12L12,17V14H8V10H12V7L17,12M21,16.5C21,16.88 20.79,17.21 20.47,17.38L12.57,21.82C12.41,21.94 12.21,22 12,22C11.79,22 11.59,21.94 11.43,21.82L3.53,17.38C3.21,17.21 3,16.88 3,16.5V7.5C3,7.12 3.21,6.79 3.53,6.62L11.43,2.18C11.59,2.06 11.79,2 12,2C12.21,2 12.41,2.06 12.57,2.18L20.47,6.62C20.79,6.79 21,7.12 21,7.5V16.5M12,4.15L5,8.09V15.91L12,19.85L19,15.91V8.09L12,4.15Z" /></svg>
            </button>
        </form>
    </div>
    <script>
        function sendMessage(event, choice) {{
            event.preventDefault();
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (message === '') return;

            // Add user message to chat window
            const chatWindow = document.getElementById('chat-window');
            const userMessage = `
                <div class="msg-container msg-self">
                    <div class="msg-box">
                        <div class="flr">
                            <div class="messages">
                                <p class="msg">${{message}}</p>
                            </div>
                            <span class="timestamp"><span class="username">You</span>&bull;<span class="posttime">Now</span></span>
                        </div>
                        <img class="user-img" src="https://gravatar.com/avatar/56234674574535734573000000000001?d=retro" />
                    </div>
                </div>
            `;
            chatWindow.insertAdjacentHTML('beforeend', userMessage);

            // Clear input
            input.value = '';

            // Auto-scroll to the latest message
            chatWindow.scrollTop = chatWindow.scrollHeight;

            // Send message to backend using AJAX
            fetch('/process_message', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    message: message,
                    choice: choice
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                // Add bot response to chat window
                const botMessage = `
                    <div class="msg-container msg-remote">
                        <div class="msg-box">
                            <img class="user-img" src="https://gravatar.com/avatar/00034587632094500000000000000000?d=retro" />
                            <div class="flr">
                                <div class="messages">
                                    <p class="msg">${{data.response}}</p>
                                </div>
                                <span class="timestamp"><span class="username">Bot</span>&bull;<span class="posttime">Now</span></span>
                            </div>
                        </div>
                    </div>
                `;
                chatWindow.insertAdjacentHTML('beforeend', botMessage);
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }})
            .catch(error => {{
                console.error('Error:', error);
            }});
        }}
    </script>
    """
    st.components.v1.html(chat_html, height=600)

# Main App
def main():
    st.title("Stock Analysis Dashboard")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Stock Analysis", "Monte Carlo Simulation", "Financial Ratios", "News Sentiment", "Latest News", "Recommendations", "Predictions"]
    choice = st.sidebar.radio("Choose a section", options)

    # Chat Interface
    chat_interface(choice)

    # NewsAPI Key Input
    newsapi_key = st.sidebar.text_input("Enter your NewsAPI Key", type="password")

    # Stock Ticker Input
    stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")

    # Handle each section
    if choice == "Stock Analysis":
        st.header("Stock Analysis")
        if st.button("Submit"):
            with st.spinner("Fetching stock data..."):
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

                    # Generate a summary of the stock data using Cohere
                    prompt = f"""
                    Analyze the stock data for {stock_ticker}.
                    The stock price over the last year is as follows: {stock_data['Close'].tolist()}.
                    Provide a summary of the stock's performance.
                    """
                    summary = generate_response(prompt)
                    st.write("### Stock Performance Summary")
                    st.write(summary)

    elif choice == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        if st.button("Submit"):
            with st.spinner("Running Monte Carlo Simulation..."):
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

                        # Generate a summary of the Monte Carlo simulation using Cohere
                        prompt = f"""
                        Analyze the Monte Carlo simulation results for {stock_ticker}.
                        The simulations show the following price paths: {simulations[:, :10].tolist()}.
                        Provide a summary of the simulation results.
                        """
                        summary = generate_response(prompt)
                        st.write("### Simulation Summary")
                        st.write(summary)

    elif choice == "Financial Ratios":
        st.header("Financial Ratios")
        if st.button("Submit"):
            with st.spinner("Calculating financial ratios..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    risk_metrics = calculate_risk_metrics(stock_data)
                    st.write("### Financial Ratios")
                    st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

                    # Generate a summary of the financial ratios using Cohere
                    prompt = f"""
                    Analyze the financial ratios for {stock_ticker}.
                    The ratios are as follows: {risk_metrics}.
                    Provide a summary of the financial health of the stock.
                    """
                    summary = generate_response(prompt)
                    st.write("### Financial Ratios Summary")
                    st.write(summary)

    elif choice == "News Sentiment":
        st.header("News Sentiment Analysis")
        if st.button("Submit"):
            if not newsapi_key:
                st.error("Please enter your NewsAPI key in the sidebar.")
            else:
                with st.spinner("Fetching and analyzing news..."):
                    articles = fetch_news(stock_ticker, newsapi_key)
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

                        # Generate a summary of the news sentiment using Cohere
                        prompt = f"""
                        Analyze the news sentiment for {stock_ticker}.
                        The sentiment counts are as follows: {sentiment_counts}.
                        Provide a summary of the news sentiment.
                        """
                        summary = generate_response(prompt)
                        st.write("### News Sentiment Summary")
                        st.write(summary)
                    else:
                        st.warning("No news articles found for this stock ticker.")

    elif choice == "Latest News":
        st.header("Latest News")
        if st.button("Submit"):
            if not newsapi_key:
                st.error("Please enter your NewsAPI key in the sidebar.")
            else:
                with st.spinner("Fetching latest news..."):
                    articles = fetch_news(stock_ticker, newsapi_key)
                    if articles:
                        st.write("### Top 5 News Articles")
                        for article in articles[:5]:  # Display top 5 articles
                            st.write(f"**Title:** {article.get('title', 'No Title Available')}")
                            st.write(f"**Description:** {article.get('description', 'No Description Available')}")
                            st.write(f"**Source:** {article.get('source', {}).get('name', 'N/A')}")
                            st.write(f"**Published At:** {article.get('publishedAt', 'N/A')}")
                            st.write("---")

                        # Generate a summary of the latest news using Cohere
                        prompt = f"""
                        Summarize the latest news for {stock_ticker}.
                        The top news articles are: {articles[:5]}.
                        Provide a summary of the latest news.
                        """
                        summary = generate_response(prompt)
                        st.write("### Latest News Summary")
                        st.write(summary)
                    else:
                        st.warning("No news articles found for this stock ticker.")

    elif choice == "Recommendations":
        st.header("Recommendations")
        period = st.number_input("Enter Analysis Period (days)", value=30)
        if st.button("Submit"):
            with st.spinner("Generating recommendations..."):
                stock_data = fetch_stock_data(stock_ticker)
                if not stock_data.empty:
                    financial_ratios = calculate_risk_metrics(stock_data)
                    recommendations = generate_recommendations(stock_data, financial_ratios, period)
                    st.write("### Recommendations")
                    for recommendation in recommendations:
                        st.write(recommendation)

                    # Generate a summary of the recommendations using Cohere
                    prompt = f"""
                    Analyze the recommendations for {stock_ticker}.
                    The recommendations are: {recommendations}.
                    Provide a summary of the recommendations.
                    """
                    summary = generate_response(prompt)
                    st.write("### Recommendations Summary")
                    st.write(summary)

    elif choice == "Predictions":
        st.header("Predictions")
        model_type = st.selectbox("Select Model", ["LSTM", "XGBoost", "ARIMA", "Prophet", "Random Forest", "Linear Regression", "Moving Average"])
        if st.button("Submit"):
            with st.spinner("Fetching data and making predictions..."):
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

                            # Generate a summary of the predictions using Cohere
                            prompt = f"""
                            Analyze the predictions for {stock_ticker}.
                            The predicted prices are: {predictions}.
                            Provide a summary of the predictions.
                            """
                            summary = generate_response(prompt)
                            st.write("### Predictions Summary")
                            st.write(summary)

                    except Exception as e:
                        st.error(f"Error in predictions: {e}")

if __name__ == "__main__":
    main()

