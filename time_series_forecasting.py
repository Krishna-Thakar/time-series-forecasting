
"""
Time Series Analysis and Forecasting
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from sklearn.metrics import mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    stock_data = data[["Date", "Close"]]
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data = stock_data.set_index("Date")
    return stock_data

# Function to visualize data
def visualize_data(stock_data):
    plt.style.use('ggplot')
    plt.figure(figsize=(18, 8))
    plt.plot(stock_data['Close'], linewidth=3, color='blue')
    plt.title('Tesla Stock Closing Price', fontsize=30)
    plt.xlabel('Dates', fontsize=20)
    plt.ylabel('Close Prices', fontsize=20)
    plt.grid(True)
    plt.show()

# Function to test stationarity
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(48).mean()
    rolstd = timeseries.rolling(48).std()
    
    plt.figure(figsize=(18, 8))
    plt.plot(timeseries, color='blue', label='Original', linewidth=3)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=3)
    plt.plot(rolstd, color='black', label='Rolling Std', linewidth=4)
    plt.legend(loc='best', fontsize=20, shadow=True, facecolor='lightpink', edgecolor='k')
    plt.title('Rolling Mean and Standard Deviation', fontsize=25)
    plt.show()

    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)

# Function to build and evaluate ARIMA model
def arima_model(train_data, test_data):
    history = [x for x in train_data['Close']]
    predictions = list()
    
    for t in range(len(test_data)):
        model = ARIMA(history, order=(1, 1, 1))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test_data['Close'].iloc[t])  # Access the 'Close' value from test_data
    
    rmse = np.sqrt(mean_squared_error(test_data['Close'], predictions))
    print('RMSE of ARIMA Model:', rmse)
    
    plt.figure(figsize=(18, 8))
    plt.plot(test_data.index, test_data['Close'], label='True Test Close Value', linewidth=5)
    plt.plot(test_data.index, predictions, label='Predictions on test data', linewidth=5)
    plt.legend(fontsize=20, shadow=True, facecolor='lightpink', edgecolor='k')
    plt.show()

# Main function to run the analysis
def main():
    # Load data
    stock_data = load_data("TSLA.csv")
    
    # Visualize data
    visualize_data(stock_data)
    
    # Test for stationarity
    test_stationarity(stock_data['Close'])
    
    # Split data into train and test sets
    train_data = stock_data[0:-60]
    test_data = stock_data[-60:]
    
    # Build and evaluate ARIMA model
    arima_model(train_data, test_data)

# Entry point of the script
if __name__ == "__main__":
    main()
