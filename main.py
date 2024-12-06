from flask import Flask, render_template, request
import joblib   # Loading the trained model
import numpy as np  # For numerical operations

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Homepage
@app.route('/')
def home():
    # Use index.html as template for the homepage
    return render_template('index.html')

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    
    input = request.form.values()   # Get input values from the submission
    input_values = []       # List to store input values

    for value in input:
        
        if value:   # Check if value is not empty
            try:
                input_values.append(float(value))

            except ValueError:
                # If conversion fails, print error message and append 0.0
                print(f"Invalid input: {value}") 
                input_values.append(0.0)

        else:
            # If value is empty, append 0.0
            input_values.append(0.0)

    # Convert list to NumPy array and reshape for prediction
    input_array = np.array(input_values).reshape(1, -1)

    # Make a prediction using the trained model
    prediction = model.predict(input_array)

    # Output prediction result in homepage
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

# Run application in debug mode if script executed directly
if __name__ == "__main__":
    app.run(debug=True)