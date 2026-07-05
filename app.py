from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load dataset
data = pd.read_csv("electricity.csv")

# Input and Output
X = data[["Temperature", "Humidity", "Hour"]]
y = data["Consumption"]

# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        hour = float(request.form["hour"])

        prediction = model.predict([[temperature, humidity, hour]])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)