from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your model, scaler and feature columns (adjust paths as needed)
model = pickle.load(open("churn_prediction.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
feature_cols = pickle.load(open("feature_columns.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Safely get all form fields
    form = request.form

    # Required fields list (add all you want)
    fields = [
        'SIM_Provider', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]

    data = {}
    for f in fields:
        val = form.get(f)
        if val is None:
            # Handle missing gracefully, you can set defaults if needed
            val = ''
        data[f] = val

    # Convert numeric fields properly
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        try:
            data[col] = float(data[col]) if data[col] != '' else 0.0
        except ValueError:
            data[col] = 0.0

    # Make dataframe and prepare input (one-hot encoding etc) like your training data
    df = pd.DataFrame([data])

    # Make sure df columns match training columns with fill_value 0
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale data
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    result = "Customer will CHURN ❌" if prediction == 1 else "Customer will NOT churn ✅"

    return render_template("index.html", prediction_text=result, data=data)


if __name__ == "__main__":
    app.run(debug=True)
