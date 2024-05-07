# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# backend.py
# ----------------------------------------------------------------
from flask import Flask, request, jsonify, render_template
from funcs import eval_argument

app = Flask(__name__)

# Initial webpage
@app.route("/")
def index():
    return render_template("index.html")

# Get the argument and return its evaluation
@app.route("/get_analysis", methods=["POST"])
def get_analysis():
    user_argument = request.form["user_argument"]
    return jsonify(eval_argument(user_argument))

# Run application
if __name__ == "__main__":
    app.run()