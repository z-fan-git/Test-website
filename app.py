from flask import Flask, request, render_template
import xgboost as xgb
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pretrained model
model = joblib.load("xgb_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve inputs
        try:
            var1 = float(request.form["var1"])
            var2 = float(request.form["var2"])
            var3 = float(request.form["var3"])
            
            # Prepare data for prediction
            input_data = [[var1, var2, var3]]
            prediction = model.predict(input_data)[0]
            
            return render_template("index.html", result=prediction)
        except Exception as e:
            return render_template("index.html", error="Invalid input. Please check your entries.")
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
