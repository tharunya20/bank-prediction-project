from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("bank_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = [float(x) for x in request.form.values()]

        while len(features) < 42:
            features.append(1)

        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        output = prediction[0]

        return render_template(
            'index.html',
            prediction_text=f'Prediction: {output}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {e}'
        )

if __name__ == "__main__":
    app.run(debug=True)