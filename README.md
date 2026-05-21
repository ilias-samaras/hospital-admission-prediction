# Hospital Admission Prediction: Stacking Ensemble Machine Learning

## Project Overview
This repository contains a Machine Learning solution designed to predict patient disposition (Admit vs. Discharge) upon visiting a hospital's Emergency Department (ED). The objective is to provide a predictive mechanism that aids in hospital bed management, resource allocation, and overall operational efficiency.

## Methodology

### 1. Data Preprocessing & Pipeline Architecture
To strictly prevent **Data Leakage** and ensure robust deployment, the data preparation steps were integrated into a Scikit-Learn `Pipeline` utilizing a `ColumnTransformer`. 
* **Numeric Features** (e.g., Age, Vitals): Missing values were handled using `Median Imputation`, followed by `Standardization`.
* **Categorical Features** (e.g., Triage ESI, Gender, Department): Missing values were imputed using the `Most Frequent` strategy, followed by `One-Hot Encoding`. 
* **Domain Shift Handling:** The encoder was configured with `handle_unknown='ignore'`. This proved crucial for external validation, allowing the model to seamlessly process unseen categories (e.g., a completely new hospital department) without runtime errors.

### 2. Stacking Ensemble Architecture
A **Stacking Classifier** was developed to leverage the diversity of multiple algorithms, reducing both bias and variance:
* **Base Learners (Level 0):**
  * `Random Forest` (Bagging)
  * `XGBoost` (Boosting)
  * `LinearSVC` (Support Vector Machine)
  * `GaussianNB` (Probabilistic)
* **Meta-Learner (Level 1):** `Logistic Regression` was used to combine the predictions of the base models.
* **Validation:** 5-fold Cross-Validation (Out-of-Fold predictions) was utilized during the base learner training to prevent the meta-learner from overfitting.

## Results & Evaluation

### Internal Validation (Test Set - 35%)
The model was evaluated on a hold-out test set of 171,073 patient records:
* **Overall Accuracy:** 79%
* **Discharge Recall:** 88%
* **Admit Recall:** 59%

### External Validation (Domain Shift)
The system was tested on a completely new, unseen dataset from a 3rd Hospital (71,706 records). The model successfully handled the domain shift (including a new department category 'C'):
* **Overall Accuracy:** 82%
* The model maintained strong predictive power for discharges, effectively demonstrating its production-readiness in new healthcare environments.

## Business Value
* **Bed Optimization:** The high recall (88%) for patient discharges allows hospital administration to proactively plan bed cleaning and reduce ED overcrowding.
* **Decision Support:** The model serves as an effective AI assistant, providing a reliable "second opinion" to triage nurses and attending physicians.
* **Scalability:** The pipeline architecture ensures zero IT maintenance costs when deploying the model to new hospitals with slightly varying categorical data formats.

## Technologies Used
* Python
* Scikit-Learn (Pipelines, Ensembles, Imputation)
* XGBoost
* Pandas & NumPy
* Matplotlib (Learning Curves, Confusion Matrices)
