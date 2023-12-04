
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained pickle model
with open("./app/trainedmodel/StackedPickle.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        features = [
            float(request.form['Latitude']),
            float(request.form['Longitude']),
            float(request.form['Price']),
            float(request.form['Number_of_reviews']),
            float(request.form['Calculated_host_listing_count']),
            float(request.form['Availability']),
            float(request.form['Number_of_reviews_ltm'])
        ]

        # Process the features and make a prediction using the model
        features = np.array(features)
        inputs = features.reshape(1, -1)
        prediction = model.predict(inputs)[0]
        
        # Return the prediction to the HTML template
        return render_template("index.html", prediction=prediction)
    
    # If it's a GET request or form submission is not valid, render the form
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
