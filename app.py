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
    # return 'fo'
    #print(feature)
    return render_template('index.html', prediction_text='Purchased {}'.format(output))


    #return  render_template('index.html', prediction_text=" Employee salary should be ${}". format(output))


if __name__=="__main__":
    app.run(debug=True)

