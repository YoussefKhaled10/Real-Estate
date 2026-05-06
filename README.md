# Real Estate Price Prediction – End‑to‑End Machine Learning System

## Project Overview
This project is a **production‑ready end‑to‑end machine learning system** for predicting real estate property prices.

The main challenge of this project was dealing with **large, noisy, and low‑quality real‑world data**, similar to what is typically encountered in industry environments.

The system covers the **entire machine learning lifecycle**, from raw data preprocessing to deployment with an API and an interactive user interface.

---

## Key Features
- ✅ Extensive data cleaning and preprocessing for dirty, inconsistent data
- ✅ Domain‑driven feature engineering
- ✅ Target encoding for high‑cardinality categorical features
- ✅ Multiple regression models with comparison
- ✅ Hyperparameter optimization using **Optuna**
- ✅ Model explainability using **SHAP**
- ✅ **FastAPI** backend for inference
- ✅ **Gradio** UI for interactive predictions
- ✅ Clean, modular, production‑grade project structure

---

## Data Challenges
The dataset was **large and poorly structured**, containing:
- Missing values
- Inconsistent text formats (areas, titles, locations)
- Outliers and unrealistic values
- Noisy categorical variables with high cardinality

### To handle these challenges, the project implements:
- Robust text parsing using **regular expressions**
- **KNN Imputation** for missing numerical values
- **IQR‑based outlier removal**
- Logical and domain‑based validation constraints

---

## Project Architecture
User (Gradio UI)
↓
FastAPI Backend
↓
Inference Pipeline
↓
ML Model + Encoder

### Folder Structure

src/
├── data/        → loading, cleaning, splitting
├── features/    → feature encoding & transformations
├── models/      → training, evaluation, prediction logic
└── pipelines/   → training & inference pipelines
deployment/
├── fastapi/     → REST API backend
└── gradio/      → UI frontend
notebooks/       → EDA, experiments, analysis

This structure mirrors **real‑world ML systems used in production**.

---

##  Technologies Used
- Python
- Pandas / NumPy
- Scikit‑learn
- LightGBM
- Optuna
- SHAP
- FastAPI
- Gradio

---

##  Modeling
Several regression models were evaluated, including:
- ✅ LightGBM
- CatBoost
- XGBoost
- Random Forest
- Gradient Boosting

**LightGBM** was selected as the final model due to its strong performance and stability after hyperparameter tuning.

---

## Explainability
**SHAP** was used to:
- Analyze global feature importance
- Understand individual predictions
- Validate model behavior on real estate features

This makes the model **transparent and interpretable**, which is critical for pricing systems.

---

## Deployment

### Backend
- FastAPI exposes a `/predict` endpoint
- Full input validation with **user‑friendly error messages**
- Production‑safe inference pipeline

### Frontend
- Gradio UI allows users to input property details
- Communicates directly with the FastAPI backend
- Displays clear predictions and validation feedback

---

## Validation & Robustness
The system includes **realistic, domain‑driven input validation**, preventing:
- Impossible building configurations  
  (e.g., apartment on floor 50 of a 10‑floor building)
- Unrealistic areas or room counts
- Invalid construction years
- Logically inconsistent inputs

All errors are returned as **clean, human‑readable messages**.

---

## Use Cases
- Real estate price estimation
- ML system architecture demonstration
- Base system for production deployment

---

## Final Notes
This project demonstrates:

Strong data engineering skills
Real‑world machine learning problem solving
Attention to production quality
Clean and scalable software architecture
Full end‑to‑end ML deployment


## Author
Youssef Khaled

Machine Learning Engineer
