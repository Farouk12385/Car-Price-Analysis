# %%
#  Task 4 — Model 1: Linear Regression (Car Price Prediction)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# %%
train = pd.read_csv('train_preprocessed.csv')
test  = pd.read_csv('test_preprocessed.csv')

print(f" Data loaded — Train: {train.shape}  |  Test: {test.shape}")

# %%
TARGET = 'price'

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test  = test.drop(columns=[TARGET])
y_test  = test[TARGET]




# %%
corr_matrix1 = train.corr(numeric_only=True)

corr_with_price = corr_matrix1['price']

top_corr = corr_with_price.sort_values(ascending=False)

print("\n📊 Top Features Correlated with Price:")
print(top_corr.head(15).to_string())



# %%
# Features Correlation Matrix 
plt.figure(figsize=(14, 10))

top_features = top_corr.head(15).index
corr_matrix = train[top_features].corr()

sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    linewidths=0.5
)

plt.title("Features Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("features_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print(" Plot saved → features_correlation_matrix.png")



# %%
#  Train 
lr = LinearRegression()
lr.fit(X_train, y_train)
print(" Model trained successfully.")

# %%
#  Predict 
y_pred_train = lr.predict(X_train)
y_pred_test  = lr.predict(X_test)

# %%
# Evaluate
def evaluate(y_true, y_pred, split_name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'─'*42}")
    print(f"  {split_name} Set Evaluation")
    print(f"{'─'*42}")
    print(f"  MAE  (Mean Absolute Error)         : {mae:.4f}")
    print(f"  MSE  (Mean Squared Error)           : {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error)      : {rmse:.4f}")
    print(f"  R²   (Coefficient of Determination) : {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2 }

train_metrics = evaluate(y_train, y_pred_train, "Train")
test_metrics  = evaluate(y_test,  y_pred_test,  "Test")

# %%
# Overfitting Check 
print(f"\n{'─'*42}")
print("  Overfitting Check")
print(f"{'─'*42}")
gap = train_metrics['R2'] - test_metrics['R2']
print(f"  Train R² : {train_metrics['R2']:.4f}")
print(f"  Test  R² : {test_metrics['R2']:.4f}")
print(f"  R² Gap   : {gap:.4f}  {'  possible overfit' if gap > 0.1 else ' looks fine'}")

# %%
# Visualisations 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Linear Regression — Model Evaluation", fontsize=14, fontweight='bold')

#  Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.4, edgecolors='k', linewidths=0.3, color='steelblue')
mn = min(y_test.min(), y_pred_test.min())
mx = max(y_test.max(), y_pred_test.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")
ax.legend()

#  Residuals vs Predicted
residuals = y_test - y_pred_test
ax = axes[1]
ax.scatter(y_pred_test, residuals, alpha=0.4, edgecolors='k', linewidths=0.3, color='coral')
ax.axhline(0, color='red', lw=1.5, linestyle='--')
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted")

# Residuals Distribution
ax = axes[2]
ax.hist(residuals, bins=40, color='mediumseagreen', edgecolor='black', linewidth=0.5)
ax.axvline(0, color='red', lw=1.5, linestyle='--')
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.set_title("Residuals Distribution")

plt.tight_layout()
plt.savefig("linear_regression_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Plot saved → linear_regression_evaluation.png")

# %%
# Top 15 Feature Coefficients
coef_df = pd.DataFrame({
    'Feature'    : X_train.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False).head(15)

print("\nTop 15 Most Influential Features:")
print(coef_df.to_string(index=False))

fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = ['steelblue' if c > 0 else 'coral' for c in coef_df['Coefficient']]
ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax2.axvline(0, color='black', lw=0.8)
ax2.set_title("Top 15 Feature Coefficients")
ax2.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("linear_regression_coefficients.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Coefficients plot saved → linear_regression_coefficients.png")


