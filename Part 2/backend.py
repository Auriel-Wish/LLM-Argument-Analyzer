# flask_app.py
from flask import Flask, request, jsonify, render_template
from funcs import eval_argument

# Create a Flask application
app = Flask(__name__)

# Route to serve the frontend page with the textbox
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint to receive text data from the frontend
@app.route("/submit", methods=["POST"])
def submit():
    # Get the text data from the POST request
    user_text = request.form["user_text"]

    # Process the text data (example: simply return it back)
    return jsonify(eval_argument(user_text))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)