# Time Series Analysis and Forecasting: Tesla Stock Prices

This project aims to perform time series analysis and forecasting on Tesla stock prices using the ARIMA model.

## Dataset

The dataset used consists of historical Tesla stock prices.

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Scikit-learn

## Data Preprocessing

- Loaded the dataset and converted the 'Date' column to datetime format
- Set the 'Date' column as the index
- Visualized the closing prices

## Model Training and Evaluation

- Tested for stationarity using rolling statistics and the Augmented Dickey-Fuller test
- Built and evaluated an ARIMA model on the training and test datasets
- Calculated the RMSE for the model's predictions

## Results

The model's Root Mean Squared Error (RMSE) was printed to the console, and the true vs. predicted values were plotted.

## How to Run

1. Install the necessary libraries:
    ```shell
    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
    ```
2. Ensure you have the dataset file named `TSLA.csv` in the same directory.
3. Run the script:
    ```shell
    python time_series_forecasting.py
    ```



