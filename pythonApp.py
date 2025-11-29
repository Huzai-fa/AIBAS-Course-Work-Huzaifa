import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import statsmodels.api as sm

df = pd.read_csv('dataset01.csv')
x = df[['x']]
y = df['y']
# solution for task # 1
num_entries = len(y)
print(f"Number of data entries: {num_entries}")

# solution for task # 2
mean_for_y = y.mean()
print(f"Mean: {mean_for_y:.6f}")

#solution for task # 3
standard_deviation = y.std()
print(f"Standard deviation: {standard_deviation}")

#solution for task # 4
variance_for_y = y.var()
print(f"4. Variance: {variance_for_y:.6f}")

#solution for task # 5
min_y = y.min()
max_y = y.max()
print(f"Minimum: {min_y:.6f}")
print(f"Maximum: {max_y:.6f}")

# Task 6: Determine and print the OLS model using statsmodels
print("\n6. OLS Model Results:")
print("=" * 50)

# Prepare data for statsmodels (add constant for intercept)
x_with_const = sm.add_constant(x)

# Create and fit OLS model
model = sm.OLS(y, x_with_const)
result = model.fit()
# Print the detailed summary
print(result.summary())

# Extract and print the simplified equation
slope = result.params[1]
intercept = result.params[0]
print(f"\nOLS Equation: y = {slope:.6f} * x + {intercept:.6f}")

# Store the model results
result.save('OLS_model.txt')
print("OLS model saved to 'OLS_model.txt'")
