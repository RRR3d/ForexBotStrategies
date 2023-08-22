import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from calculate_rsi_module import calculate_rsi

# Load the trained model from the saved file
model_filename = "forex_prediction_model.joblib"
loaded_model = joblib.load(model_filename)

# Fetch new data using yfinance
new_data = yf.download("USDEUR=X", start="2023-08-19", end="2023-08-20", progress=False)

# Calculate additional technical indicators for the new data
new_data["SMA_50"] = new_data["Close"].rolling(window=50).mean()
new_data["SMA_200"] = new_data["Close"].rolling(window=200).mean()

# Calculate RSI for the new data
new_data["RSI"] = calculate_rsi(new_data["Close"], window=14)

# Select features
features = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200", "RSI"]
new_features = new_data[features]

# Handle missing values using the same imputer used during training
imputer = SimpleImputer(strategy="mean")
new_features_imputed = imputer.transform(new_features)

# Make predictions using the loaded model
predictions = loaded_model.predict(new_features_imputed)
prediction_probabilities = loaded_model.predict_proba(new_features_imputed)

# Combine predictions with the original data
new_data["Predicted_Label"] = predictions
new_data["Predicted_Prob_Up"] = prediction_probabilities[:, 1]
new_data["Predicted_Prob_Down"] = prediction_probabilities[:, 0]

# Print the predictions and probabilities
print(new_data)
