
# Stock Price Prediction with LSTM/GRU and Hyperparameter Tuning

This project aims to predict stock prices for Apple Inc. (AAPL) using LSTM/GRU models with hyperparameter tuning. The project utilizes historical stock data, technical indicators, and walk-forward validation to train and evaluate the model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)

## Installation

### Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras Tuner
- yfinance
- TA-Lib

### TA-Lib Installation

To install TA-Lib, follow these steps:

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Python Package Installation

Install the required Python packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras-tuner yfinance
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

2. Run the script:

```bash
python stock_prediction.py
```

## Model Training and Evaluation

The script performs the following steps:

1. Downloads historical stock data for Apple Inc. (AAPL) from Yahoo Finance.
2. Adds technical indicators (Moving Averages, RSI, MACD) to the data.
3. Preprocesses the data by scaling and creating training/testing datasets.
4. Defines a hypermodel using LSTM/GRU layers with hyperparameter tuning using Keras Tuner.
5. Uses TimeSeriesSplit for walk-forward validation.
6. Trains and evaluates the model, and saves the best model and plots.

## Results

The script will output the following results:

1. A plot showing the actual vs. predicted stock prices.
2. The model's Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
3. A plot showing the training and validation loss over epochs.
4. Saved best model and plots as PNG files.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
