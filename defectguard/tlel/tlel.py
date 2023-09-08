from flask import Flask, request
import pickle
import pandas as pd
import os
from TLEL import TLEL

app = Flask(__name__)

@app.route('/api/tlel', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "input": 14 features
        }
    '''
    request_data = request.get_json()

    # get input
    features = ["ns", "nd", "nf", "entrophy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    input = request_data["input"]
    input = {key:[input[key]] for key in features}
    input = pd.DataFrame(input)

    # get model
    model_path = os.path.join(os.getcwd(), "model")
    files = os.listdir(model_path)
    predict = []
    for file in files:
        with open(os.path.join(model_path, file), "rb") as f:
            model = pickle.load(f) 
        predict.append(model.predict_proba(input)[:, 1][0])
    
    # if app.debug:
    #     print(input)
    #     print(predict)
    # Create response like form below:
    '''
        {
            "id": string,
            "output": float
        }
    '''
    return {
        'id': request_data["id"],
        'output': 1/6 * sum(predict)
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5005)