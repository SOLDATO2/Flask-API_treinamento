import joblib
from sklearn.metrics import r2_score
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    X = []

    X.append(features['Age'])
    X.append(features['Gender'])
    X.append(features['Hypertension'])
    X.append(features['Heart Disease'])
    X.append(features['Average Glucose Level'])
    X.append(features['Smoking Status'])
    X.append(features['Physical Activity'])
    X.append(features['Stroke History'])
    X.append(features['Family History of Stroke'])
    
    model = joblib.load('model.joblib')
    y_pred = model.predict([X])
    AVC = [0,1]
    

    response_data = {
        'class': AVC[y_pred[0]],
        'message': 'Você corre risco de AVC.' if y_pred[0] == 1 else 'Você não corre risco de AVC.'
    }
    
    print(response_data)  #printar resposta no terminal
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
