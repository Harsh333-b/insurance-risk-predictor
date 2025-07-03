from flask import Flask, render_template, request, send_file
import pandas as pd
import shap
import joblib
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from fpdf import FPDF

app = Flask(__name__)

risk_model = joblib.load("model/xgb_classifier.pkl")
price_model = xgb.XGBRegressor()
price_model.load_model("model/xgb_regressor.json")
explainer = joblib.load("model/explainer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect user input
    user_input = {
        "age": int(request.form["age"]),
        "sex": int(request.form["sex"]),
        "bmi": float(request.form["bmi"]),
        "children": int(request.form["children"]),
        "smoker": int(request.form["smoker"]),
        "region": int(request.form["region"]),
        "MI-ALL": int(request.form["MI-ALL"]),
        "STRAIN": int(request.form["STRAIN"]),
        "Q-ISC": int(request.form["Q-ISC"])
    }

    df_input = pd.DataFrame([user_input])

    # Prediction
    risk_prob = risk_model.predict_proba(df_input)[0][1]
    risk_label = "High Risk" if risk_prob > 0.5 else "Low Risk"
    cost_prediction = price_model.predict(df_input)[0]
    confidence = round(risk_prob * 100, 2)

    shap_values = explainer(df_input)
    max_feature = df_input.columns[shap_values.values[0].argmax()]
    shap_text = f"Your risk is primarily affected by your '{max_feature}' value."

    os.makedirs("static", exist_ok=True)

    plt.clf()
    shap.plots.bar(shap_values[0], show=False)
    plt.savefig("static/shap_plot.png", bbox_inches="tight", dpi=200)
    plt.close()

    plt.clf()
    shap.plots.force(shap_values[0], matplotlib=True, show=False)
    plt.savefig("static/force_plot.png", bbox_inches="tight", dpi=200)
    plt.close()

    if risk_label == "High Risk":
        recommendation = (
            "We recommend reviewing your health coverage and considering a higher coverage plan. "
            "Additionally, explore preventive health programs and schedule regular checkups."
        )
    else:
        recommendation = (
            "Your risk is currently low. Maintain a healthy lifestyle and review your policy annually. "
            "You may consider a cost-effective plan if affordability is a concern."
        )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Predicted Risk: {risk_label}\nEstimated Charges: Rs. {cost_prediction:,.2f}\n"
                          f"Confidence Score: {confidence}%\n\nSHAP Reason: {shap_text}\n\n"
                          f"Policy Recommendation: {recommendation}")
    pdf.image("static/shap_plot.png", x=10, w=180)
    pdf.image("static/force_plot.png", x=10, w=180)
    pdf.output("static/report.pdf", "F")

    return render_template("result.html",
                           prediction=f"Predicted Risk: {risk_label}, Estimated Charges: Rs. {cost_prediction:,.2f}",
                           confidence=f"Confidence Score: {confidence}%",
                           shap_reason=shap_text,
                           recommendation=recommendation,
                           shap_image="static/shap_plot.png",
                           force_image="static/force_plot.png")

@app.route("/download", methods=["GET"])
def download():
    return send_file("static/report.pdf", as_attachment=True)

@app.route("/feedback", methods=["POST"])
def feedback():
    user_feedback = request.form.get("feedback", "")
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(user_feedback + "\n---\n")
    return render_template("thanks.html")

if __name__ == "__main__":
    app.run(debug=True)
