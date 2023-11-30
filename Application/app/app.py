
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained pickle model
with open("trainedmodel/StackedPickle.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        # Extracting user input from the form
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])
        price = float(request.form['Price'])
        num_reviews = float(request.form['Number_of_reviews'])
        host_listings_count = float(request.form['Calculated_host_listing_count'])
        availability = float(request.form['Availability'])
        num_reviews_ltm = float(request.form['Number_of_reviews_ltm'])

        # Creating a feature vector for prediction
        features = [latitude, longitude, price, num_reviews, host_listings_count, availability, num_reviews_ltm]
        features = np.array(features).reshape(1, -1)

        # Making a prediction using the pre-trained model
        prediction = model.predict(features)[0]
        
        # Return the prediction to the HTML template along with variable descriptions
        return render_template("index.html", prediction=prediction,
                               latitude=latitude, longitude=longitude, price=price,
                               num_reviews=num_reviews, host_listings_count=host_listings_count,
                               availability=availability, num_reviews_ltm=num_reviews_ltm)
    
    # If it's a GET request or form submission is not valid, render the form
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

