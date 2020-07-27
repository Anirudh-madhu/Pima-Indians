from flask import Flask, request, render_template
import pickle
import numpy

app= Flask(__name__)
model= pickle.load(open("dogs.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    feature = [float(x) for x in request.form.values()]
    feature = [numpy.array(feature)]

    prediction = model.predict(feature)

    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text='Have diabetes or not {}'.format(output))


    

if __name__=="__main__":
    app.run(debug=True)

