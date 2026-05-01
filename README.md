## AI-Powered Credit Card Fraud Detection

### The Problem

Credit card fraud is one of the biggest financial threats in digital transactions. The dataset behind this project was highly imbalanced, noisy, and difficult to interpret — with fraudulent transactions being extremely rare compared to legitimate ones.

This imbalance often leads to poor model performance, where traditional systems either miss fraud cases or generate too many false alarms, costing businesses both money and customer trust.

### The Solution (My Approach)

To tackle this, I built an end-to-end machine learning web application focused on high-accuracy fraud detection.

### Data Cleaning & Processing

Handled extreme class imbalance
Applied feature scaling using StandardScaler
Cleaned and validated transaction data
Built a structured preprocessing pipeline for consistency


### Model Engineering

Instead of relying on a single model, I implemented a Stacking Ensemble Model combining:

Logistic Regression
Random Forest
XGBoost / LightGBM
CatBoost
Meta Learner (final prediction layer)

This approach significantly improves prediction stability and accuracy.

### The Result

Achieved high fraud detection accuracy with reduced false positives
Enabled real-time fraud prediction for both single and bulk transactions
Designed a system that can potentially save thousands in fraud losses by early detection
Delivered a production-ready ML app with a clean and interactive UI

### Live Demo

https://credit-card-fraud-detection-appi.streamlit.app/

### Key Features

Real-time single transaction prediction
Bulk CSV fraud detection
Fraud probability scoring (0 – 100)
End-to-end ML pipeline
Clean and professional Streamlit interface

### Why This Project Stands Out

This is not just a model — it’s a complete business-ready solution.
It demonstrates:

Strong understanding of real-world data challenges
Advanced use of ensemble learning
Ability to deploy ML into a usable product
