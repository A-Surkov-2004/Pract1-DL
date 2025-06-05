from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

gdata = pd.read_csv('bank.csv')
gdata = gdata.drop('deposit', axis=1)

with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global gdata
    data = gdata.copy()

    client_data = request.get_json()

    new_row = list(client_data.values())
    new_df = pd.DataFrame([new_row], columns=data.columns)

    data = pd.concat([data, new_df], ignore_index=True, )

    # кодируем бинарные категориальные признаки
    data['default'] = data['default'].apply(lambda x: 0 if x == 'no' else 1)
    data['housing'] = data['housing'].apply(lambda x: 0 if x == 'no' else 1)
    data['loan'] = data['loan'].apply(lambda x: 0 if x == 'no' else 1)

    # оставшиеся категориальные признаки кодируем с помощью OneHot
    data = pd.get_dummies(data)

    X_test = data.iloc[-1]

    predictions = loaded_model.predict(np.array(X_test).reshape(1, -1))

    #features = np.array(data['features']).reshape(1, -1)
    ans = str(predictions[len(predictions)-1])
    return ans
    # prediction = model.predict(features)
    # return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
