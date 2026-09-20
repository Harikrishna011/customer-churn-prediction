from flask import Flask, request, jsonify, render_template
import torch
import joblib
from model import ChurnNeuralNetwork
import os
import pandas as pd
app=Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models folder
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Load preprocessing objects
encoder = joblib.load(
    os.path.join(MODEL_DIR, "encoder.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_names = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

input_size=len(feature_names)
model=ChurnNeuralNetwork(input_size)
model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "churn_model.pth"),
        weights_only=True
    )
)

model.eval()
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Convert input JSON into DataFrame
    input_df = pd.DataFrame([data])

    # Separate categorical and numerical columns
    categorical_columns = input_df.select_dtypes(
        include=["object", "string"]
    ).columns

    numerical_columns = input_df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Encode categorical features
    encoded_data = encoder.transform(
        input_df[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    # Combine numerical + encoded data
    final_data = pd.concat(
        [
            input_df[numerical_columns].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Make sure feature order matches training
    final_data = final_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale
    final_data_scaled = scaler.transform(final_data)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        final_data_scaled,
        dtype=torch.float32
    )

    # Prediction
    model.eval()

    with torch.no_grad():
        probability = model(input_tensor).item()

    prediction = 1 if probability >= 0.5 else 0

    return jsonify({
        "churn_probability": probability,
        "prediction": prediction,
        "result": "Churn" if prediction == 1 else "No Churn"
    })
print("Expected features:")
print(feature_names)
if __name__ == "__main__":
    app.run(debug=True)