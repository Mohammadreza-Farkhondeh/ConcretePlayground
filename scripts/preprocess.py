import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/concrete_compressive_strength.csv")

df.dropna(inplace=True)

df["Age (day)"] = np.log(df["Age (day)"])

target_variable = "Concrete compressive strength(MPa, megapascals) "
X = df.drop(columns=[target_variable])
y = df[target_variable]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.01, random_state=42
# )

X_train, X_test, y_train, y_test = X, None, y, None

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

joblib.dump((X_train, X_test, y_train, y_test), "data/preprocessed_data.pkl")
joblib.dump(scaler, "data/scaler.pkl")

scaler_params = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()}
with open("pages/scaler_params.json", "w") as f:
    json.dump(scaler_params, f)
