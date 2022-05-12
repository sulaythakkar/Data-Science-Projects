import pandas as pd
from datetime import datetime

stocks = pd.read_csv("sphist.csv")
stocks["Date"] = pd.to_datetime(stocks["Date"])

stocks = stocks.sort_values("Date")

#Indicator 1 - Avg Price for the past 5 trading days
stocks["avg_5"] = stocks["Close"].rolling(window=5).mean()
stocks["avg_5"] = stocks.avg_5.shift(1)
#Indicator 2 - Avg Price for the past 30 trading days
stocks["avg_10"] = stocks["Close"].rolling(window=10).mean()
stocks["avg_10"] = stocks.avg_10.shift(1)
#Indicator 3 - Standard Deviation for the past 5 trading days
stocks["std_5"] = stocks["Close"].rolling(window=5).std()
stocks["std_5"] = stocks.std_5.shift(1)

#Indicator 4 - Avg Volume for the past 5 trading days
stocks["avg_vol_5"] = stocks["Volume"].rolling(window=5).mean()
stocks["avg_vol_5"] = stocks.avg_vol_5.shift(1)
#Indicator 3 - Avg Volume for the past 10 trading days
stocks["avg_vol_10"] = stocks["Volume"].rolling(window=10).mean()
stocks["avg_vol_10"] = stocks.avg_vol_10.shift(1)
#print(stocks.head(10))


stocks = stocks[stocks["Date"] >= datetime(1951,1,3)]
stocks = stocks.dropna(axis=0)

train = stocks[stocks["Date"] < datetime(2013,1,1)]
test = stocks[stocks["Date"] >= datetime(2013,1,1)]

#print(train.head())
#print(test.head())

#Training
features = train.columns
features = features.drop(["Close", "High", "Low", "Open", "Volume", "Adj Close", "Date"])
print(features)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[features],train["Close"])
predictions = model.predict(test[features])

mae = (predictions - test["Close"]).abs().mean()
print(mae)