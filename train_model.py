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

# EXPLORATORY DATA ANALYSIS (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], kde=True)
plt.title("Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.savefig("images/price_distribution.png", bbox_inches="tight")
plt.show()


# 2. Average Area Income vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Avg. Area Income"], y=df["Price"])
plt.title("Average Area Income vs House Price")
plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.savefig("images/income_vs_price.png", bbox_inches="tight")
plt.show()


# 3. House Age vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Avg. Area House Age"], y=df["Price"])
plt.title("Average Area House Age vs House Price")
plt.xlabel("Average Area House Age")
plt.ylabel("House Price")
plt.savefig("images/house_age_vs_price.png", bbox_inches="tight")
plt.show()


# 4. Number of Rooms vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Avg. Area Number of Rooms"], y=df["Price"])
plt.title("Number of Rooms vs House Price")
plt.xlabel("Average Area Number of Rooms")
plt.ylabel("House Price")
plt.savefig("images/rooms_vs_price.png", bbox_inches="tight")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png", bbox_inches="tight")
plt.show()


# 6. Boxplot for Price
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Price"])
plt.title("Boxplot of House Price")
plt.ylabel("House Price")
plt.savefig("images/price_boxplot.png", bbox_inches="tight")
plt.show()


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