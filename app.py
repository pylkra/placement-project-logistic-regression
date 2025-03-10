# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(f'int_features: {int_features}')
    print(f'final_features: {final_features}')
    
    # Make prediction
    prediction_text = model.predict(final_features)
    output = 'Placed' if prediction_text[0] == 1 else 'Not Placed'
    print(f'Prediction: {output}')

    return render_template('index.html', prediction_text='placement: {}'.format(output))

if __name__ == "__main__":
    print("Starting Flask server... 🚀")
    app.run(debug=True, host="127.0.0.1", port=5000)