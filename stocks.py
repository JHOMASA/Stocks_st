import streamlit as st
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
import plotly.graph_objects as go
from textblob import TextBlob  # Fallback sentiment analysis

# Initialize Cohere client
co = cohere.Client("gpWuZqkXdfhfbYkjLlyRnc5x2rj0ml1IqfULfjt0")  # Replace with your valid API key

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

# Analyze sentiment of news articles
def analyze_news_sentiment(articles):
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0, "ERROR": 0}
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        sentiment = "ERROR"  # Default value

        try:
            # Use Cohere API for sentiment analysis
            response = co.classify(
                model="your-fine-tuned-model-id",  # Replace with your fine-tuned model ID
                inputs=[text],
                examples=[
                    {"text": "This is great news!", "label": "POSITIVE"},
                    {"text": "This is terrible news!", "label": "NEGATIVE"},
                    {"text": "This is neutral news.", "label": "NEUTRAL"}
                ]
            )
            sentiment = response.classifications[0].prediction
            time.sleep(1.5)  # Add a delay to stay within rate limits
        except Exception as e:
            st.warning(f"Cohere API failed. Error: {e}")
            # Fallback to TextBlob for sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "POSITIVE"
            elif polarity < 0:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"

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

# Train XGBoost model
def train_xgboost_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Predict using XGBoost
def predict_xgboost(model, data):
    last_return = data['Close'].pct_change().iloc[-1]
    predictions = []
    for _ in range(30):  # Predict next 30 days
        pred = model.predict(np.array([[last_return]]))
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Train ARIMA model
def train_arima_model(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))  # (p, d, q) parameters
    model_fit = model.fit()
    return model_fit

# Predict using ARIMA
def predict_arima(model, steps=30):
    predictions = model.forecast(steps=steps)
    return predictions

# Train Prophet model
def train_prophet_model(data):
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone
    model = Prophet()
    model.fit(df)
    return model

# Predict using Prophet
def predict_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast['yhat'][-periods:].values

# Train Random Forest model
def train_random_forest_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# Predict using Random Forest
def predict_random_forest(model, data, steps=30):
    predictions = []
    last_return = data['Close'].pct_change().iloc[-1]
    for _ in range(steps):
        pred = model.predict([[last_return]])
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Train Linear Regression model
def train_linear_regression_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict using Linear Regression
def predict_linear_regression(model, data, steps=30):
    predictions = []
    last_return = data['Close'].pct_change().iloc[-1]
    for _ in range(steps):
        pred = model.predict([[last_return]])
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Predict using Moving Average
def predict_moving_average(data, window=30):
    predictions = data['Close'].rolling(window=window).mean().iloc[-30:].values
    return predictions

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

    elif choice == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                risk_metrics = calculate_risk_metrics(stock_data)
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))

    elif choice == "News Sentiment":
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
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)

    elif choice == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(stock_ticker)
                if articles:
                    try:
                        sentiment_counts = analyze_news_sentiment(articles)
                        st.subheader("Sentiment Summary")
                        st.write(f"Positive: {sentiment_counts['POSITIVE']}")
                        st.write(f"Negative: {sentiment_counts['NEGATIVE']}")
                        st.write(f"Neutral: {sentiment_counts['NEUTRAL']}")
                        st.write(f"Errors: {sentiment_counts['ERROR']}")

                        st.subheader("Top 5 News Articles")
                        for article in articles[:5]:  # Display top 5 articles
                            st.write(f"**Title:** {article.get('title', 'No Title Available')}")
                            st.write(f"**Description:** {article.get('description', 'No Description Available')}")
                            st.write(f"**Sentiment:** {article.get('sentiment', 'N/A')}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error processing articles: {e}")
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

if __name__ == "__main__":
    main()

