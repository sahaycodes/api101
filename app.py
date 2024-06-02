import yfinance as yf
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

def get_data(assets):
    df = pd.DataFrame()
    yf.pdr_override()
    for stock in assets:
        df[stock] = web.get_data_yahoo(stock)['Adj Close']
    return df

def monte_carlo_simulation(df, num_portfolios=5000, risk_free_rate=0.02):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(df.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        results[0, i] = portfolio_return * 100
        results[1, i] = portfolio_std_dev
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe Ratio

    results[3] = np.arange(1, num_portfolios + 1)
    sim_df = pd.DataFrame(results.T, columns=['Returns', 'Volatility', 'Sharpe Ratio', 'Portfolio'])
    sim_df['Weights'] = weights_record

    return sim_df

def optimize_portfolio(df, tickers_string, starting_amount=100):
    # Monte Carlo Simulation
    num_of_portfolios = 10000
    sim_df = monte_carlo_simulation(df, num_of_portfolios)

    # Selecting the portfolio with the highest Sharpe Ratio
    max_sharpe_ratio = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]
    optimal_weights = max_sharpe_ratio['Weights']

    # Creating DataFrame for weights
    weights_df = pd.DataFrame(optimal_weights, index=df.columns, columns=['Weights'])

    # Expected annual return, volatility, and Sharpe ratio
    mean_returns = df.pct_change().mean()
    cov_matrix = df.pct_change().cov() * 252
    annual_return = np.sum(mean_returns * optimal_weights) * 252
    port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (annual_return - 0.02) / port_volatility
    plot_sim_df(sim_df)

    return weights_df, annual_return, port_volatility, sharpe_ratio

def plot_chart(df, title):
    fig = px.line(df, title=title)
    fig.show()

def plot_sim_df(sim_df):
    fig = px.scatter(sim_df, x='Volatility', y='Returns', color='Sharpe Ratio',title='MonteCar Sim: Ret vs Volatility',labels={'Volatility': 'Volatility', 'Returns': 'Returns'})
    fig.show()

def gaussian_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def calculate_probabilities(user_point, centroids):
    distances = np.array([euclidean_distance(user_point, centroid) for centroid in centroids])
    std = np.std(distances)
    probabilities = np.array([gaussian_pdf(dist, 0, std) for dist in distances])
    normalized_probabilities = probabilities / np.sum(probabilities)
    return normalized_probabilities

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

# Input Data
lifestyle_risk = 1  # Change as needed
expected_annual_roi = 25  # Change as needed
principal_amount = 100000  # Change as needed

info_data = [
    {"Symbol": "AAPL", "Annualized ROI": 17.624598, "Volatility": 0.027930},
    {"Symbol": "AMZN", "Annualized ROI": 30.874074, "Volatility": 0.035528},
    {"Symbol": "ARKK", "Annualized ROI": 7.464424, "Volatility": 0.023803},
    {"Symbol": "BABA", "Annualized ROI": -1.149219, "Volatility": 0.026288},
    {"Symbol": "BTC-USD", "Annualized ROI": 57.557349, "Volatility": 0.036763},
    {"Symbol": "ELF", "Annualized ROI": 21.738374, "Volatility": 0.031783},
    {"Symbol": "ETH-USD", "Annualized ROI": 35.917395, "Volatility": 0.046763},
    {"Symbol": "GC=F", "Annualized ROI": 8.950140, "Volatility": 0.010886},
    {"Symbol": "GOOGL", "Annualized ROI": 22.442833, "Volatility": 0.019344},
    {"Symbol": "GSK", "Annualized ROI": 10.011961, "Volatility": 0.017046},
    {"Symbol": "ITC.NS", "Annualized ROI": 16.263788, "Volatility": 0.020341},
    {"Symbol": "JNJ", "Annualized ROI": 10.925850, "Volatility": 0.014432},
    {"Symbol": "MSFT", "Annualized ROI": 24.020376, "Volatility": 0.021158},
    {"Symbol": "NFLX", "Annualized ROI": 31.412082, "Volatility": 0.035304},
    {"Symbol": "NVDA", "Annualized ROI": 34.711791, "Volatility": 0.037871},
    {"Symbol": "PLD", "Annualized ROI": 5.721513, "Volatility": 0.023028},
    {"Symbol": "QCOM", "Annualized ROI": 18.908380, "Volatility": 0.030630},
    {"Symbol": "SQ", "Annualized ROI": 17.814773, "Volatility": 0.036852},
    {"Symbol": "TCEHY", "Annualized ROI": 17.299087, "Volatility": 0.023695},
    {"Symbol": "TSLA", "Annualized ROI": 37.054781, "Volatility": 0.035839},
    {"Symbol": "XOM", "Annualized ROI": 7.051585, "Volatility": 0.014561}
]

info = pd.DataFrame(info_data)
model_data = info[['Annualized ROI', 'Volatility']]
kmeans = KMeans(n_clusters=3, random_state=4)
kmeans.fit(model_data)
clusters = kmeans.predict(model_data)
info['Cluster'] = clusters
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(model_data, clusters)
centroids = kmeans.cluster_centers_

if lifestyle_risk == 0:
    expected_volatility = centroids[0][1]
elif lifestyle_risk == 1:
    expected_volatility = centroids[2][1]
elif lifestyle_risk == 2:
    expected_volatility = centroids[1][1]
else:
    raise ValueError("Invalid lifestyle risk value")

model_input = np.array([[expected_annual_roi, expected_volatility]])
predicted_cluster = knn.predict(model_input)

probabilities = calculate_probabilities(model_input, centroids)
nearest_centroid_index = np.argmax(probabilities)
weighted_amounts = principal_amount * probabilities
weights_df = pd.DataFrame({'Weight': weighted_amounts})

clusters_data = {
    "Symbols": [
        "ARKK, BABA, GC=F, GSK, JNJ, PLD, XOM",
        "AMZN, BTC-USD, ETH-USD, NFLX, NVDA, TSLA",
        "AAPL, ELF, GOOGL, ITC.NS, MSFT, QCOM, SQ, TCEHY"
    ]
}

clusters_df = pd.DataFrame(clusters_data)
clusters_df['Weights'] = weights_df['Weight']

results = []
for index, row in clusters_df.iterrows():
    ticker = str(row['Symbols'])
    ticker = remove_spaces(ticker)
    assets = ticker.split(',')
    df = get_data(assets)
    starting_amount = row['Weights']
    weights_df, annual_return, port_volatility, sharpe_ratio = optimize_portfolio(df, ticker, starting_amount=starting_amount)
    results.append({
        "Symbols": row['Symbols'],
        "Weights": weights_df.to_dict(),
        "Annual Return": annual_return,
        "Volatility": port_volatility,
        "Sharpe Ratio": sharpe_ratio
    })
   

print("Results:")
print(results)
print("Clusters:")
print(clusters_df.to_dict(orient='records'))
