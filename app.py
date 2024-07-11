from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('.\heart_disease_prediction (2).pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    
    # Create feature array
    features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return result
    result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease Detected'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
