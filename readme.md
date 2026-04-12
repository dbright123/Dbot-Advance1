# Forex Trading with Machine Learning

This repository contains a Python script that leverages machine learning for Forex trading on MetaTrader 5. The script utilizes a range of technical indicators to forecast the future price of currency pairs and execute orders accordingly.

## Getting Started

To use the script, follow these steps:

1. Install the required Python libraries:
   - MetaTrader 5
   - NumPy
   - scikit-learn
   - joblib

2. Train a machine learning model on historical data. You can choose any machine learning algorithm, but we recommend using a support vector machine (SVM).

3. Save the trained model to a file using joblib.

4. Edit the `target_market` list in the script to include the currency pairs you wish to trade.

5. Replace the `models` list with the trained models saved in step 3.

6. Start MetaTrader 5 and ensure that the "Allow Automated Trading" checkbox is selected in the "Tools" -> "Options" -> "Expert Advisors" tab.

7. Run the script.

The script continuously monitors the current price of the currency pairs in the `target_market` list and places orders based on predictions. If the script forecasts an increase in a currency pair's price, it will place a buy order. Conversely, if it predicts a decrease, it will place a sell order.

The script also includes several risk management features, such as stop-losses and take-profits, which can be customized to match your risk tolerance.

Please be aware that this script is intended for educational purposes only. It is not financial advice, and there are no guarantees of profitability.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the [MIT License](LICENSE).
