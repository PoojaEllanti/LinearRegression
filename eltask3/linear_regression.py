# Linear Regression on Housing Price Prediction Dataset

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("Housing.csv")  # Make sure housing.csv is in the same folder

# Step 3: Preview data
print("First 5 rows of data:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 4: Clean data (remove rows with missing values)
df = df.dropna()
print(f"\nData shape after removing missing values: {df.shape}")

# Step 5: Simple Linear Regression (using 'area' to predict 'price')
X_simple = df[['area']]
y = df['price']

# Step 6: Split data into train and test sets (simple regression)
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Step 7: Train the simple linear regression model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

# Step 8: Predict prices using test data
y_pred_simple = simple_model.predict(X_test)

# Step 9: Evaluate the model
print("\nSimple Linear Regression Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_simple))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_simple))
print("R-squared (R²):", r2_score(y_test, y_pred_simple))

# Step 10: Plot regression line (Simple Linear Regression)
plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred_simple, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 11: Print intercept and coefficient
print("\nSimple Linear Regression Coefficients:")
print("Intercept:", simple_model.intercept_)
print("Slope (Area Coefficient):", simple_model.coef_[0])

# Step 12: Multiple Linear Regression (area, bedrooms, bathrooms → price)
X_multi = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 13: Split data into train and test sets (multiple regression)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Step 14: Train the multiple linear regression model
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# Step 15: Predict and evaluate
y_pred_multi = multi_model.predict(X_test_m)

print("\nMultiple Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test_m, y_pred_multi))
print("MSE:", mean_squared_error(y_test_m, y_pred_multi))
print("R²:", r2_score(y_test_m, y_pred_multi))

# Step 16: Print coefficients
print("\nMultiple Linear Regression Coefficients:")
print("Intercept:", multi_model.intercept_)
print("Feature Coefficients:")
for feature, coef in zip(['area', 'bedrooms', 'bathrooms'], multi_model.coef_):
    print(f"{feature}: {coef}")
