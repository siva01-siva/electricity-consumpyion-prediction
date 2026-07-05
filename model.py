import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
data = pd.read_csv("electricity.csv")

# Input Features
X = data[["Temperature", "Humidity", "Hour"]]

# Output
y = data["Consumption"]

# Create Model
model = RandomForestRegressor(random_state=42)

# Train Model
model.fit(X, y)

# Predict
prediction = model.predict([[31, 58, 12]])

print("Predicted Electricity Consumption:", prediction[0], "kWh")