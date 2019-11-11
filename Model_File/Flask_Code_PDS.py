import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
scaler = pickle.load(open('tranform.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    
    final_features = np.pad(final_features, (0, 63), 'constant')
    
    final_features = scaler.transform(final_features)
    
    prediction = model.predict_proba(final_features)

    output = prediction[0]

    return render_template('interface.html', prediction_text='Readmission probability is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

