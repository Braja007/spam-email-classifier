from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the trained vectorizer and classifier
cv_path = os.path.join("models", "cv.pkl")
clf_path = os.path.join("models", "clf.pkl")

with open(cv_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(clf_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')

    # Handle empty input
    if not email or email.strip() == "":
        return render_template("index.html", error="⚠️ Please enter an email.", email=email)

    # Vectorize input email
    tokenized_email = vectorizer.transform([email])
    prediction = model.predict(tokenized_email)[0]  # 1: Ham, 0: Spam

    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
