from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form.get('feature', '').strip()

        if not features:
            return render_template('index.html',
                error="Please enter feature values.",
                feature_input=features)

        features_list = [x.strip() for x in features.split(',')]

        if len(features_list) != 31:
            return render_template('index.html',
                error=f"Expected 31 features, but got {len(features_list)}. Please check your input.",
                feature_input=features)

        np_features = np.asarray(features_list, dtype=np.float64).reshape(1, -1)
        pred = model.predict(np_features)
        proba = model.predict_proba(np_features)[0]

        result = "cancerous" if pred[0] == 1 else "benign"
        confidence = round(float(max(proba)) * 100, 2)

        return render_template('index.html',
            result=result,
            confidence=confidence,
            feature_input=features)

    except ValueError:
        return render_template('index.html',
            error="Invalid input: make sure all 31 values are numeric.",
            feature_input=request.form.get('feature', ''))
    except Exception as e:
        return render_template('index.html',
            error=f"An error occurred: {str(e)}",
            feature_input=request.form.get('feature', ''))


if __name__ == '__main__':
    app.run(debug=True)
