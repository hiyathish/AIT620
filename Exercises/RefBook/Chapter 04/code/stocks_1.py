import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
# from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
import yfinance as yf

# Input file containing company symbols
input_file = 'company_symbol_mapping.json'

# Load the company symbol map
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Load the historical stock quotes
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)
# quotes = [quotes_yahoo(symbol, start_date, end_date, asobject=True)
#                 for symbol in symbols]
cnt=1
quotes = []
for symbol in symbols:
    try:
        #df = web.DataReader(symbol, "yahoo", start_date, end_date)
        df = yf.download(symbol, start=start_date, end=end_date)
        print("DF: ",df.head())
        print("CNT: ", cnt)
        quotes.append(df)
        cnt += 1
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        quotes.append(None)

# Extract opening and closing quotes
# opening_quotes = np.array([quote.open for quote in quotes]).astype(np.float64)
# closing_quotes = np.array([quote.close for quote in quotes]).astype(np.float64)
opening_quotes = np.array([quote["Open"].values for quote in quotes], dtype=float)
closing_quotes = np.array([quote["Close"].values for quote in quotes], dtype=float)


# Compute differences between opening and closing quotes
quotes_diff = closing_quotes - opening_quotes

# Normalize the data
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Create a graph model
# edge_model = covariance.GraphLassoCV()
edge_model = covariance.GraphicalLassoCV()

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Build clustering model using Affinity Propagation model
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Print the results of clustering
print('\nClustering of stocks based on difference in opening and closing quotes:\n')
for i in range(num_labels + 1):
    print("Cluster", i+1, "==>", ', '.join(names[labels == i]))

