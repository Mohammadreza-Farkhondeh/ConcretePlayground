import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/concrete_compressive_strength.csv')

df.dropna(inplace=True)

target_variable = "Concrete compressive strength(MPa, megapascals) "
X = df.drop(columns=[target_variable])
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump((X_train, X_test, y_train, y_test), 'data/preprocessed_data.pkl')
joblib.dump(scaler, 'data/scaler.pkl')

