import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('dataset02.csv')  # Replace with your CSV file path

print("Original data shape:", df.shape)

# Step 1: Drop non-numerical columns and NaN values
df = df.select_dtypes(include=[np.number])  # Keep only numerical columns
df = df.dropna()  # Remove rows with NaN values
print("After dropping non-numerical and NaN:", df.shape)

# Step 2: Remove outliers using Z-score (|Z| > 3)
z_scores = np.abs(stats.zscore(df))
df = df[(z_scores < 3).all(axis=1)]
print("After removing outliers:", df.shape)

# Step 3: Normalize data (Min-Max scaling to 0-1 range)
df_normalized = (df - df.min()) / (df.max() - df.min())

print("\nFinal normalized data shape:", df_normalized.shape)

# Step 4: Split data into training (80%) and testing (20%) sets
# Assuming the last column is the target variable
X = df_normalized.iloc[:, :-1]  # All columns except last
y = df_normalized.iloc[:, -1]   # Last column as target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create training and testing DataFrames
training_data = X_train.copy()
training_data['target'] = y_train

testing_data = X_test.copy()
testing_data['target'] = y_test

# Save to CSV files
training_data.to_csv('dataset02_training.csv', index=False)
testing_data.to_csv('dataset02_testing.csv', index=False)

print(f"\nTraining set shape: {training_data.shape}")
print(f"Testing set shape: {testing_data.shape}")
print("Files saved: 'dataset02_training.csv' and 'dataset02_testing.csv'")

# Step 5: Implement OLS model using ONLY training data
print("\n" + "="*50)
print("OLS MODEL USING TRAINING DATA ONLY")
print("="*50)

# Add constant for intercept term
X_train_with_const = sm.add_constant(X_train)

# Create and fit OLS model using training data only
model = sm.OLS(y_train, X_train_with_const)
results = model.fit()

# Display OLS results
print(results.summary())

# Optional: Make predictions on test data to see performance
X_test_with_const = sm.add_constant(X_test)
y_pred = results.predict(X_test_with_const)

# Calculate test performance metrics
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = results.rsquared

print(f"\nModel Performance (on training data):")
print(f"R-squared: {r2:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Create scatter plot visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Get feature names (all columns except the last one which is target)
feature_names = df_normalized.columns[:-1]
n_features = len(feature_names)

for i, feature in enumerate(feature_names):
    if i < 4:  # Plot up to 4 features in 2x2 grid
        ax = axes[i]
        
        # Plot training data in orange
        ax.scatter(training_data[feature], training_data['target'],
                  color='orange', alpha=0.7, label='Training Data', s=30)
        
        # Plot testing data in blue
        ax.scatter(testing_data[feature], testing_data['target'],
                  color='blue', alpha=0.7, label='Testing Data', s=30)
        
        # Create red line plot (OLS regression line)
        # Generate points for the regression line
        x_line = np.linspace(df_normalized[feature].min(), df_normalized[feature].max(), 100)
        
        # Create prediction matrix for this feature
        X_line = np.zeros((100, n_features))
        X_line[:, i] = x_line  # Set current feature values
        
        # Add constant and predict
        X_line_const = sm.add_constant(X_line)
        y_line = results.predict(X_line_const)
        
        # Plot the red regression line
        ax.plot(x_line, y_line, color='red', linewidth=2, label='OLS Model')
        
        ax.set_xlabel(f'Feature: {feature}')
        ax.set_ylabel('Target Variable')
        ax.set_title(f'{feature} vs Target')
        ax.legend()
        ax.grid(True, alpha=0.3)

# Hide any unused subplots
for i in range(n_features, 4):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('UE_04_App2_ScatterVisualizationAndOlsModel.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("\nScatter plot saved as 'UE_04_App2_ScatterVisualizationAndOlsModel.pdf'")

# 2) Box Plot of all dimensions
print("\n" + "="*50)
print("CREATING BOX PLOT")
print("="*50)

plt.figure(figsize=(12, 8))
df_normalized.boxplot()
plt.title('Box Plot of All Data Dimensions')
plt.ylabel('Normalized Values')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('UE_04_App2_BoxPlot.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Box plot saved as 'UE_04_App2_BoxPlot.pdf'")

# 3) Diagnostic Plots
print("\n" + "="*50)
print("CREATING DIAGNOSTIC PLOTS")
print("="*50)

try:
    # Import the diagnostic class
    from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
    
    # Create diagnostic plots
    fig = plt.figure(figsize=(15, 10))
    diagnostic = LinearRegDiagnostic(results)
    diagnostic.plot_diagnostics()
    
    plt.tight_layout()
    plt.savefig('UE_04_App2_DiagnosticPlots.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("Diagnostic plots saved as 'UE_04_App2_DiagnosticPlots.pdf'")
    
except ImportError:
    print("Warning: UE_04_LinearRegDiagnostic.py not found. Creating basic diagnostic plots instead.")
    
    # Create basic diagnostic plots manually
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get fitted values and residuals
    fitted_values = results.fittedvalues
    residuals = results.resid
    
    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q Plot
    sm.qqplot(residuals, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scale-Location Plot (sqrt standardized residuals vs fitted)
    standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
    axes[1, 0].scatter(fitted_values, standardized_residuals, alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals vs Leverage
    influence = results.get_influence()
    leverage = influence.hat_matrix_diag
    axes[1, 1].scatter(leverage, residuals, alpha=0.6)
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('UE_04_App2_DiagnosticPlots.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("Basic diagnostic plots saved as 'UE_04_App2_DiagnosticPlots.pdf'")

print("\n" + "="*50)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*50)
print("Generated files:")
print("1. UE_04_App2_ScatterVisualizationAndOlsModel.pdf")
print("2. UE_04_App2_BoxPlot.pdf")
print("3. UE_04_App2_DiagnosticPlots.pdf")
print("4. dataset02_training.csv")
print("5. dataset02_testing.csv")
