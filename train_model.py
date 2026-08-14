import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
import numpy as np
import joblib

df = pd.read_csv("dataset/housing.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.duplicated().sum())

X = df.drop(['Price', 'Address'], axis=1)
y = df['Price']

print("Original dataset shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns)

print("\nX first 5 rows:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model=LinearRegression()
model.fit(X_train,y_train)

Prediction=model.predict(X_test)
score= r2_score(y_test,Prediction)
mae = mean_absolute_error(y_test, Prediction)
rmse = np.sqrt(mean_squared_error(y_test,Prediction))
print("The R2_Score is ",score)
print("MAE is", mae)
print("RMSE is", rmse)
print("="*40)
print("Model trained successfully")
print("="*40)

joblib.dump(model,"House_price_prediction_pkl")
print("MODEL SAVED SUCCESSFULLY")