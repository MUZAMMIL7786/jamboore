from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('OLS_model.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    sno = int(request.form.get("sno"))
    gre = int(request.form.get("gre"))
    toefl = float(request.form.get("toefl"))
    university = float(request.form.get("university"))
    sop = float(request.form.get("sop"))
    lor = float(request.form.get("lor"))
    cgpa = float(request.form.get("cgpa"))
    research = float(request.form.get("research"))

    result = model.predict(np.array([1.0, toefl, university, sop, lor, cgpa, research]).reshape(1, 7))

    result =  f"The possibility of student placing is {np.round(result[0], 2)}%"
    return render_template("index.html", result = result)

if __name__ == '__main__':
    app.run(debug = True)

