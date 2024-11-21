import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 500

# 3 input features
X = np.random.rand(n_samples, 3)
# Target variable (y) - let's use a simple formula for demonstration
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, n_samples)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 5. Save the model
joblib.dump(model, "xgb_model.pkl")
print("Model saved as xgb_model.pkl")
