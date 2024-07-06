import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import keras_tuner as kt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import talib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs found: {gpus}")
    except RuntimeError as e:
        logger.error(f"GPU runtime error: {e}")
else:
    logger.info("No GPUs found, using CPU.")

# Download historical data for Apple Inc. (AAPL)
try:
    stock_data = yf.download('AAPL', start='2015-01-01', end='2024-06-01')
    # stock_data = yf.Ticker('AAPL').history(period='max')
    data = stock_data[['Close', 'Volume']]
    logger.info("Data downloaded successfully.")
except Exception as e:
    logger.error(f"Error downloading data: {e}")
    data = None

if data is not None:
    # Add moving averages and technical indicators
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = talib.RSI(data['Close'])
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'])
    
    # Drop initial NaN values
    data.dropna(inplace=True)

    # Data Preprocessing with additional features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_len = int(np.ceil(len(data) * 0.8))
    train_data = data[:train_data_len]
    test_data = data[train_data_len:]

    # Fit the scaler on training data only
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Create training data sets
    def create_dataset(data, time_step=60):
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i-time_step:i])
            y.append(data[i, 0])
        return np.array(x), np.array(y)
    
    x_train, y_train = create_dataset(scaled_train_data)
    x_test, y_test = create_dataset(scaled_test_data)

    # Define the Hypermodel
    def build_model(hp):
        model = Sequential()

        rnn_layer = hp.Choice('rnn_layer', ['LSTM', 'GRU'])
        units = hp.Int('units', min_value=28, max_value=32, step=1)
        dropout_rate = hp.Float('dropout', min_value=0.14, max_value=0.15, step=0.001)
        
        if rnn_layer == 'LSTM':
            model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        else:
            model.add(GRU(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
            
        model.add(Dropout(rate=dropout_rate))
        
        if rnn_layer == 'LSTM':
            model.add(LSTM(units=units, return_sequences=False))
        else:
            model.add(GRU(units=units, return_sequences=False))
        
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=units))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Initialize the Tuner with Bayesian Optimization
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=30,
        executions_per_trial=1,
        directory='my_dir',
        project_name='stock_price_prediction'
    )

    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Walk-Forward Validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, val_index in tscv.split(x_train):
        x_t_train, x_val = x_train[train_index], x_train[val_index]
        y_t_train, y_val = y_train[train_index], y_train[val_index]

        # Search for the best hyperparameters
        tuner.search(x_t_train, y_t_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping])

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Retrain the best model on the entire training data with a suitable batch size
    history = best_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # Save the best model with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f'best_stock_model_{timestamp}.keras'
    best_model.save(model_filename)
    logger.info(f"Model saved as {model_filename}")

    # Predicting the values with the best model
    predictions = best_model.predict(x_test)

    # Inverse transform predictions correctly
    inverse_test_data = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_test_data.shape[1] - 1))), axis=1)
    )[:, 0]

    # Prepare data for plotting
    train = data[:train_data_len]
    valid = data[train_data_len:].copy()

    # Ensure the predictions array matches the length of the valid data frame
    valid = valid[-len(inverse_test_data):]

    valid['Predictions'] = inverse_test_data

    # Plotting the data
    plt.figure(figsize=(14, 7))
    plt.plot(train['Close'], label='Train Data')
    plt.plot(valid['Close'], label='Actual Prices')
    plt.plot(valid['Predictions'], label='Predicted Prices')
    plt.title('Apple Inc. (AAPL) Stock Price Prediction with Hyperparameter Tuning')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # Save the plot as a PNG file
    plot_filename = f'stock_prediction_plot_{timestamp}.png'
    plt.savefig(plot_filename)
    logger.info(f"Plot saved as {plot_filename}")

    # Calculate and print evaluation metrics
    mse = mean_squared_error(valid['Close'], valid['Predictions'])
    mae = mean_absolute_error(valid['Close'], valid['Predictions'])
    rmse = np.sqrt(mse)

    logger.info(f"Mean Absolute Error (MAE): {mae}")
    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot training & validation loss values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_filename = f'loss_plot_{timestamp}.png'
    plt.savefig(loss_plot_filename)
    logger.info(f"Loss plot saved as {loss_plot_filename}")
else:
    logger.error("No data available for training.")
