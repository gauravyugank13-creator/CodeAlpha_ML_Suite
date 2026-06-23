import os
import sys
import datetime
import json
import uuid
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# Resolve and append task_1_credit_scoring directory path to sys.path
task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.predictor import CreditPredictor

app = Flask(__name__)
app.secret_key = "credit_scoring_secret_key"

# Instantiate and load the global model predictor
predictor = CreditPredictor()
model_load_error = None

try:
    predictor.load_pipeline()
except Exception as e:
    model_load_error = f"Model load failure: {str(e)}"
    print(f"[Error] Failed to load production model: {e}")

# Ensure prediction directory exists
os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

# Whitelist values for categorical inputs
SEX_CATEGORIES = ["male", "female"]
JOB_CATEGORIES = [0, 1, 2, 3]
HOUSING_CATEGORIES = ["own", "rent", "free"]
SAVING_CATEGORIES = ["little", "moderate", "quite rich", "rich", "unknown"]
CHECKING_CATEGORIES = ["little", "moderate", "rich", "unknown"]
PURPOSE_CATEGORIES = [
    "business", "car", "domestic appliances", "education", 
    "furniture/equipment", "radio/TV", "repairs", "vacation/others"
]

def append_to_csv_history(record):
    """
    Appends a prediction record to the prediction_history.csv file.
    """
    history_csv_path = os.path.join(Config.PREDICTIONS_DIR, "prediction_history.csv")
    df = pd.DataFrame([record])
    
    # If the file doesn't exist, write headers, otherwise append without headers
    if not os.path.exists(history_csv_path):
        df.to_csv(history_csv_path, index=False)
    else:
        df.to_csv(history_csv_path, mode="a", header=False, index=False)

def save_individual_prediction_json(timestamp, inputs, prediction, prob_good, prob_bad, confidence):
    """
    Saves an individual prediction result as a JSON file.
    """
    unique_id = uuid.uuid4().hex[:8]
    formatted_ts = timestamp.replace("-", "").replace(" ", "_").replace(":", "")
    filename = f"pred_{formatted_ts}_{unique_id}.json"
    filepath = os.path.join(Config.PREDICTIONS_DIR, filename)
    
    data = {
      "timestamp": timestamp,
      "inputs": inputs,
      "prediction": prediction,
      "probabilities": {
        "GOOD": prob_good,
        "BAD": prob_bad
      },
      "confidence_pct": confidence
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

@app.route("/", methods=["GET"])
def home():
    # If the model failed to load, display warning alert
    return render_template("index.html", model_error=model_load_error)

@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error:
        return render_template("index.html", error=f"Cannot run prediction. {model_load_error}")
        
    try:
        # 1. Collect inputs
        age_str = request.form.get("Age")
        sex = request.form.get("Sex")
        job_str = request.form.get("Job")
        housing = request.form.get("Housing")
        saving = request.form.get("Saving accounts")
        checking = request.form.get("Checking account")
        credit_str = request.form.get("Credit amount")
        duration_str = request.form.get("Duration")
        purpose = request.form.get("Purpose")
        
        # 2. Check for missing values
        fields = [age_str, sex, job_str, housing, saving, checking, credit_str, duration_str, purpose]
        if any(f is None or f == "" for f in fields):
            return render_template("index.html", error="Validation Error: Missing values in required fields.")
            
        # 3. Server-side type conversion and validation
        try:
            age = int(age_str)
            job = int(job_str)
            credit_amount = int(credit_str)
            duration = int(duration_str)
        except ValueError:
            return render_template("index.html", error="Validation Error: Age, Job, Credit amount, and Duration must be valid numbers.")
            
        # Range checks
        if age < 18 or age > 100:
            return render_template("index.html", error="Validation Error: Age must be between 18 and 100.")
        if credit_amount < 100 or credit_amount > 100000:
            return render_template("index.html", error="Validation Error: Credit amount must be between 100 and 100,000 DM.")
        if duration < 1 or duration > 120:
            return render_template("index.html", error="Validation Error: Duration must be between 1 and 120 months.")
            
        # Categorical whitelisting
        if sex not in SEX_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Sex category: {sex}")
        if job not in JOB_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Job category: {job}")
        if housing not in HOUSING_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Housing category: {housing}")
        if saving not in SAVING_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Saving account category: {saving}")
        if checking not in CHECKING_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Checking account category: {checking}")
        if purpose not in PURPOSE_CATEGORIES:
            return render_template("index.html", error=f"Validation Error: Invalid Purpose category: {purpose}")
            
        # 4. Prepare input row DataFrame
        input_data = {
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving accounts": saving,
            "Checking account": checking,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose
        }
        df_input = pd.DataFrame([input_data])
        
        # 5. Execute Prediction
        # predict_risk_score returns prob_bad (class 1)
        prob_bad = predictor.predict_risk_score(df_input)
        prob_good = 1.0 - prob_bad
        decision = predictor.predict_decision(df_input) # 0 = Good, 1 = Bad
        
        # Determine risk mapping
        if decision == 0:
            predicted_class = "GOOD"
            confidence = prob_good * 100
            recommendation = "LOW CREDIT RISK"
            rec_class = "risk-low"
        else:
            predicted_class = "BAD"
            confidence = prob_bad * 100
            recommendation = "HIGH CREDIT RISK"
            rec_class = "risk-high"
            
        # 6. Save Logging Records
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to CSV history registry
        csv_record = {
            "timestamp": timestamp,
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving accounts": saving,
            "Checking account": checking,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose,
            "prediction": predicted_class,
            "probability_good": round(prob_good, 4),
            "probability_bad": round(prob_bad, 4),
            "confidence_pct": round(confidence, 2)
        }
        append_to_csv_history(csv_record)
        
        # Save individual prediction json log
        save_individual_prediction_json(
            timestamp=timestamp,
            inputs=input_data,
            prediction=predicted_class,
            prob_good=round(prob_good, 4),
            prob_bad=round(prob_bad, 4),
            confidence=round(confidence, 2)
        )
        
        return render_template(
            "result.html",
            inputs=input_data,
            predicted_class=predicted_class,
            prob_good=round(prob_good * 100, 2),
            prob_bad=round(prob_bad * 100, 2),
            confidence=round(confidence, 2),
            recommendation=recommendation,
            rec_class=rec_class
        )
        
    except Exception as e:
        print(f"[Error] Error during prediction pipeline execution: {e}")
        return render_template("index.html", error=f"An unexpected error occurred during prediction: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", error="Error 404: Page not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("index.html", error="Error 500: Internal server error."), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
