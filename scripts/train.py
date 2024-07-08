import joblib
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

X_train, X_test, y_train, y_test = joblib.load("data/preprocessed_data.pkl")

model_ = Sequential(
    [
        Input(shape=(8,)),  # Input layer (assuming 8 features)
        Dense(64, activation="relu"),  # Single hidden layer with 64 neurons
        Dense(1),  # Output layer
    ]
)

model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),  # Input layer
        Dense(8, activation="relu"),  # First hidden layer with 8 neurons
        Dense(8, activation="relu"),  # Second hidden layer with 8 neurons
        Dense(4),  # Third hidden layer with 4 neurons
        Dense(2),  # Fourth hidden layer with 2 neurons
        Dense(1),  # Output layer
    ]
)

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)


model.save("model/concrete_model.h5")
