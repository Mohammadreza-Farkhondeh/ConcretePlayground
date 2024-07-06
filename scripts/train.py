import joblib
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

X_train, X_test, y_train, y_test = joblib.load("data/preprocessed_data.pkl")

model = Sequential(
    [
        Input(shape=(8,)),
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(X_train, y_train, epochs=100, validation_split=0.1)

model.save("model/concrete_model.h5")
