# Heart Failure Survival Prediction

A statistical and machine learning project focused on predicting survival outcomes in patients with heart failure using classification algorithms.

Course: Advanced Topics in Statistics  
Department of Industrial Engineering and Management, Ben-Gurion University  
Submission Date: March 15, 2024  
Team: Tom Damari, Orin Cohen, Shira Yaakov  
Supervisor: Prof. Israel Perat

---

## Project Overview
Heart failure is a chronic condition that significantly impacts quality of life and survival rates. This project explores a dataset of heart failure patients and uses statistical analysis and machine learning models to identify the factors most strongly associated with patient survival.
We analyzed patient demographics, clinical biomarkers, and lifestyle attributes to find patterns, correlations, and predictive indicators. Multiple classification models were trained and evaluated, with **Random Forest** demonstrating the best performance.

---

## Research Goals
- Predict patient survival using health, demographic, and physiological features
- Identify the most influential variables on survival outcomes

### Key Hypotheses
1. **Positive correlation** between ejection fraction and survival
2. **Negative correlation** between age and survival
3. **Negative correlation** between serum creatinine and survival

---

## Files Included
| File Name                                                | Description                                    |
|----------------------------------------------------------|------------------------------------------------|
| `main.py`                                                | Python code for data analysis & modeling       |
| `HeartFailureProject.R`                                  | R script for statistical exploration           |
| `heart_failure_clinical_records_dataset.csv`             | Raw dataset from UCI repository                |
| `heart_failure_clinical_records_dataset_withMarking.csv` | Annotated version of the dataset               |
| `dataForRegression.csv`                                  | Prepared data for regression analysis          |
| `train_data.csv`                                         | Training set (pre-processed)                   |
| `test_data.csv`                                          | Test set (pre-processed)                       |

---

## Methods Used
- **Descriptive Statistics**
- **Correlation & Multicollinearity Analysis**
- **Outlier Detection**
- **Model Evaluation Metrics (Accuracy, Confusion Matrix, etc.)**
- **Classification Models**:
  - Decision Tree
  - Random Forest (Best performance)
  - Support Vector Machine (SVM)
  - Logistic Regression

---

## Dataset Source
The dataset was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).  
It contains **299 patient records** with 13 clinical features and a binary target indicating death event during follow-up.

---

## Results Summary
- **Best Model:** Random Forest  
- **Top Predictive Features (in order of importance):**
  1. Serum Creatinine
  2. Early dropout (Month 0)
  3. Ejection Fraction
  4. Dropout during Month 1

---

