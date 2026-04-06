# Imports PyTorch, a very powerful machine learning framework used especially for
# building and training neural networks.
import torch

# The neural network module of PyTorch. Gives us access to tools we will use later.
import torch.nn as nn

# The neural network module of PyTorch. 
import torch.optim as optim

# Powerful Python library used for computing.
import numpy as np

# Handles reading and managing CSV data.
import pandas as pd

# Library for drawing graphs.
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset

import os

# How many past days we will look at to predict the next day.
SEQUENCE_LENGTH = 30

# How many sequences are grouped together and passed through the model at one time.
BATCH_SIZE = 32

# How many full passes of the dataset the model will make.
EPOCHS = 20

# How many memory units are inside each LSTM layer.
# More => model can learn more complex patterns, but takes longer to train.
HIDDEN_SIZE = 64

# How many LSTM layers are stacked on each other.
# More layers lets model learn patterns at different levels of abstraction.
NUM_LAYERS = 2

# Controls size of weight updates during training.
LEARNING_RATE = 0.001

# How much data should be test data.
TEST_SPLIT = 0.2

# Where the model weights will be saved.
MODEL_PATH = "stock_lstm.pth"

# Loads CSV sorting by date and extracts the closing prices.
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    # Sorts from oldest to newest, then resets the indexes based on new ordering.
    df = df.sort_values("date").reset_index(drop=True)
    # Extracts only the closing price column, transforms it into a 2D column vector.
    prices = df["close"].values.astype(float).reshape(-1, 1)
    # Returns both price array and dates.
    return prices, df["date"].values

# Scales the raw price data into [0, 1] using MinMaxScaler.
# With LSTMs, scale matters. Values between 0 and 1 keeps everything consistent.
def scale_data(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Learns the min and max of the data, then applies scaling.
    scaled = scaler.fit_transform(prices)
    # Returns scaled data and the scaler applied so we can reverse the scaling.
    return scaled, scaler

# Converts the flat sequence of daily prices to (X, y) pairs.
# The model can learn from these easier. This is a sliding window technique.
# For each position i, X is a window of 'sequence_length' consecutive prices,
# and y is the single price that comes right after that window.
# Example where sequence_length=3 and prices [1,2,3,4,5]:
# X[0] = [1,2,3], y[0] = 4; X[1] = [2, 3, 4], y[1] = 5.
def build_sequences(scaled_prices, sequence_length):
    X, y = [], []
    # Stops before last index to so there is a next day to predict.
    for i in range(len(scaled_prices) - sequence_length):
        X.append(scaled_prices[i:i + sequence_length])
        y.append(scaled_prices[i + sequence_length])
    # Converts both to NumPy arrays which is what is required by PyTorch Tensors.
    return np.array(X), np.array(y)

# Splits sequences into training and testing sets. These are not shuffled,
# as if the model had access to future data but tested on past data would
# make accuracy better than it really is.
def split_data(X, y, test_split):
    split = int(len(X) * (1 - test_split))
    return X[:split], X[split:], y[:split], y[split:]

# Converts NumPy arrays into PyTorch tensors and wraps them in data loaders for the
# model to process in batches. The train_loader shuffles, as it is already split,
# and the test loader does not, to keep evaluation consistnet.
def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    # Converts each split from a NumPy array to PyTorch float tensor. 
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # TensorDataset pairs each input sequence with its corresponding label so they stay
    # linked when the DataLoader shuffles or batches them.
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # Also returns the raw test tensors for evaluation.
    return train_loader, test_loader, X_test_t, y_test_t

# Class that defines the structure of the neural network.
# LSTM (Long Short-Term Memory) is a type of recurrent neural network designed to learn
# patterns across sequences of data. 
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        # Calls PyTorch's initialization to get some important built in functions
        # like tracking parameters and saving / loading weights.
        super(StockLSTM, self).__init__()
        # Creates the LSTM layer, 
        # input_size=1 - each step has only 1 feature (the closing price), 
        # hidden_size=64 - each LSTM cell has 64 memory units, 
        # num_layers=2 - two LSTM layers stacked, first feeds second,
        # batch_first - tells Pytorch input will be (batch, sequence, features)
        # dropout=0.2 - randomly zeroes out 20% of connections between
        # layers, prevents overfitting.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Single fully connected layer, compresses the 64 cells into 1 value.
        self.fc = nn.Linear(hidden_size, 1)

    # Defines how data flows through the model
    def forward(self, x):
        # Runs the input through the LSTM, returning out (hidden states across each time step)
        out, _ = self.lstm(x)
        # Grabs last time step output, the only one that has seen the whole 30 day window.
        # This gets passed into the fc layer to return 1 value.
        return self.fc(out[:, -1, :])

def train_model(model, train_loader, epochs):
    # Our loss function, averages square differences between predicted and actual prices.
    criterion = nn.MSELoss()
    # Adam is a gradient descent algorithm, adaptively adjusts learning rate for each param.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Puts model into training mode.
    model.train()

    # Loops through whole dataset epochs # of times.
    for epoch in range(epochs):
        total_loss = 0
        # Iterates over each batch (shuffled and grouped sequenced to batch of 32)
        for X_batch, y_batch in train_loader:
            # Clears the gradients PyTorch saves automatically for each batch.
            optimizer.zero_grad()
            # Runs the batch through the model (calls forward() internally)
            output = model(X_batch)
            # Computes how wrong the predictions were
            loss = criterion(output, y_batch)
            # Backpropogation, goes through each operation and computes how much each
            # weight contributed to the loss.
            loss.backward()
            # Uses those gradients to shift weights in the direction to reduce loss.
            optimizer.step()
            # Accumulate the loss over each batch.
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.6f}")


def evaluate_model(model, X_test_t, y_test_t, scaler):
    # Puts model into evaluation mode
    model.eval()
    # Removes gradients as we are just checking accuracy, more efficient.
    with torch.no_grad():
        # Runs test sequence through the model and converts outputs to NumPy array.
        predictions = model(X_test_t).numpy()

    # Both predictions and actuals are in scaled [0,1] range.
    # inverse_transform reverses the MinMaxScaler to get the actual dollar value back.
    predictions_unscaled = scaler.inverse_transform(predictions)
    actuals_unscaled = scaler.inverse_transform(y_test_t.numpy())

    # Root Mean Squared Error measures average prediction error in dollars.
    rmse = np.sqrt(np.mean((predictions_unscaled - actuals_unscaled) ** 2))

    # Average percentage error across all predictions. Divides by actuals
    # to normalize
    mape = np.mean(np.abs((actuals_unscaled - predictions_unscaled) / actuals_unscaled)) * 100

    # Measures how well predictions capture the variance in actual prices.
    # 1.0 is perfect, 0.0 means the model is no better than predicting the mean,
    # and negative means it's worse than that.
    ss_res = np.sum((actuals_unscaled - predictions_unscaled) ** 2)
    ss_tot = np.sum((actuals_unscaled - np.mean(actuals_unscaled)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test R²:   {r2:.4f}")

    return predictions_unscaled, actuals_unscaled

# Uses matplotlib to create graph of actual and predictions.
def plot_results(predictions, actuals, dates_test):
    plt.figure(figsize=(12, 5))
    plt.plot(dates_test, actuals, label="Actual Price", color="blue")
    plt.plot(dates_test, predictions, label="Predicted Price", color="orange", linestyle="--")
    plt.title("Stock Price: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def predict_next(model, scaled_prices, scaler, sequence_length):
    model.eval()
    # Grabs the most recent 30 days which we will extrapolate from.
    last_sequence = scaled_prices[-sequence_length:]
    # Adds batch dimension as model requires (1, 30, 1) shape for input instead of (30, 1)
    input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # Runs the sequence through the model and converts to NumPy.
        prediction_scaled = model(input_tensor).numpy()

    # Reverses the MinMaxScaler to convert the output to dollar amount.
    prediction = scaler.inverse_transform(prediction_scaled)
    print(f"\nPredicted next closing price: ${prediction[0][0]:.2f}")


def main():
    csv_path = input("Enter path to stock CSV file: ").strip()

    print("Loading data...")
    # Extracts closing prices and dates.
    prices, dates = load_data(csv_path)

    print("Scaling and building sequences...")
    # Scales prices to [0, 1] and builds sliding window.
    scaled_prices, scaler = scale_data(prices)
    X, y = build_sequences(scaled_prices, SEQUENCE_LENGTH)
    # Splits sequences chronologically into train and test sets.
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SPLIT)

    # Wraps the splits into DataLoaders for batched training and eval.
    train_loader, test_loader, X_test_t, y_test_t = make_loaders(
        X_train, y_train, X_test, y_test, BATCH_SIZE
    )

    # Creates model with the hyperparamaters we made earlier.
    model = StockLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

    use_saved = input("Load saved model? (y/n): ").strip().lower()

    if use_saved == "y" and os.path.exists(MODEL_PATH):
        # Loads the weights
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Loaded saved model.")
    else:
        print("Training model...")
        train_model(model, train_loader, EPOCHS)
        # Saves the weights so we don't have to retrain.
        torch.save(model.state_dict(), MODEL_PATH)

    print("Evaluating...")
    predictions, actuals = evaluate_model(model, X_test_t, y_test_t, scaler)

    # Aligns dates with test predictions.
    split_index = len(prices) - len(predictions)
    dates_test = dates[split_index:]

    print("Plotting results...")
    plot_results(predictions, actuals, dates_test)

    # Uses last 30 days for next closing day prediction.
    predict_next(model, scaled_prices, scaler, SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
