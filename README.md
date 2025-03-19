# Stock Price Prediction Using LSTM

## Overview
This project implements a stock price prediction model using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock data and leverages various technical indicators to make predictions.

## Features
- Uses **LSTM** for time-series forecasting.
- Includes **technical indicators** such as moving averages and Bollinger Bands.
- **Feature scaling** using MinMaxScaler.
- **Early stopping** to prevent overfitting.
- **Visualization** of actual vs. predicted stock prices.

## Data Preparation
The dataset (`reliance.csv`) should contain:
- Date (as index)
- Open, High, Low, Close, Volume columns
- Additional engineered features such as moving averages and Bollinger Bands are computed in the script.

## Model Architecture
- **Two LSTM layers** with dropout for regularization.
- **Dense layers** to generate the final stock price prediction.
- **Adam optimizer** with Mean Squared Error (MSE) loss function.

## How to Run
1. Install required libraries:
   ```sh
   pip install pandas numpy matplotlib tensorflow scikit-learn
   ```
2. Place `reliance.csv` in the same directory.
3. Run the script:
   ```sh
   python stock_price_lstm.py
   ```
4. The model will train and plot actual vs. predicted stock prices.

## Results
The script generates a graph comparing actual stock prices with predicted values, helping visualize the model's accuracy.

## Future Improvements
- Experiment with different sequence lengths.
- Try different optimizers (e.g., RMSprop, AdamW).
- Incorporate more technical indicators like RSI, MACD.
- Tune hyperparameters for better accuracy.

## License
This project is open-source under the MIT License.
