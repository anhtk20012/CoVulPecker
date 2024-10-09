import os
import json
from flask import Flask, request
# from utils import VulnDetector
from utils import VulnDetectorExplainer
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask! This is the home page."

@app.route('/predict/section', methods=["POST"])

def section():
    req = request.get_json(force=True)
    
    # range = "13-15"
    # prediction = "False"
    # lines = []
    # average = 3/11
    # results = [{"range": range, "pred": prediction, "lines": lines, "average": average}]

    range = '-'.join(str(int(e)+1) for e in req['lines'])
    result = VulnDetectorExplainer()
    results = result.run_detector(req['code'])

    prediction = results[0]
    if prediction:
        lines = results[1]
        number = int(req['lines'][1])+1 - int(req['lines'][0]+1)
        average = len(lines) / number
    else:
        lines = []
        average = 1
        
    results = [{"range": range, "pred": prediction, "lines": lines, "average": average}]
    print(results)
    return json.dumps(results)

if __name__ == '__main__':
    app.run(debug=False)
    