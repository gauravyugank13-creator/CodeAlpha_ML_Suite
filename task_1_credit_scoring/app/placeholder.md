# Flask Web Application Serving Plan

In a future phase, a Flask web application will be added here to serve the trained Credit Scoring model.

## Planned Features
- **Credit Assessment Form:** An interactive web form allowing users to input applicant characteristics (income, debt-to-income, credit history length, etc.).
- **Real-Time Prediction API:** A POST endpoint `/predict` that processes inputs through the preprocessing pipeline and returns the classification category (Low Risk vs High Risk) and risk probability.
- **Score Dashboard:** Visual indicators (gauge charts) showing credit risk categories and probability of default.
- **Error Handling:** Form validators ensuring all inputs conform to acceptable data bounds.
