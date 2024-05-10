# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# backend.py
# ----------------------------------------------------------------
from flask import Flask, request, jsonify, render_template
from funcs import eval_argument_premade, eval_argument_IBM, breakdown_argument, summarize, get_feedback, compare

app = Flask(__name__)

# Initial webpage
@app.route("/")
def index():
    return render_template("index.html")

# Get the argument and return its evaluation (model 1)
@app.route("/get_analysis_premade", methods=["POST"])
def get_analysis_premade():
    user_argument = request.form["user_argument"]
    user_argument = summarize(user_argument)
    eval = eval_argument_premade(user_argument)
    breakdown = breakdown_argument(user_argument)
    ret = [eval, breakdown]
    return jsonify(ret)

# Get the argument and return its evaluation (model 2)
@app.route("/get_analysis_IBM", methods=["POST"])
def get_analysis_IBM():
    user_argument = request.form["user_argument"]
    user_argument = summarize(user_argument)
    eval = eval_argument_IBM(user_argument)
    breakdown = breakdown_argument(user_argument)
    ret = [eval, breakdown]
    return jsonify(ret)

@app.route("/in_depth", methods=["POST"])
def in_depth():
    user_argument = request.form["user_argument"]
    arg_score = float(request.form["arg_qual"])
    feedback = get_feedback(user_argument, arg_score)
    return jsonify(feedback)

@app.route("/get_comparison", methods=["POST"])
def get_comparison():
    user_argument2 = request.form["user_argument2"]
    user_argument3 = request.form["user_argument3"]
    comparison = compare(user_argument2, user_argument3)
    return jsonify(comparison)

# Run application
if __name__ == "__main__":
    app.run()