# Hospital Readmission Prediction Using Machine Learning

## Project Overview

This project focuses on predicting patient readmissions using various machine learning models.
The dataset used in this project includes patient admission data and various medical features from
kaggle: https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data/data 

The goal is to build a predictive model using classification algorithms like Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).

## Data Description

The project uses multiple datasets such as:
- `diabetic_data.csv`: Contains information about diabetic patients.
- `LengthOfStay.csv`: Contains hospital stay duration data.
- `hospital-readmissions-orig.csv`: Hospital readmission data.
- `HDHI Admission data.csv`: Patient admission data with medical features.

The data is preprocessed by handling missing values, encoding categorical variables, and scaling features to prepare for model training.

## Features Used

Key features used for prediction:
- **GENDER**: Gender of the patient.
- **TYPE OF ADMISSION-EMERGENCY/OPD**: Type of admission (emergency or outpatient).
- **RURAL**: Whether the patient resides in a rural area.
- **PLATELETS, GLUCOSE, TLC, UREA, HB, AGE, CREATININE**: Medical lab results.
- **DURATION OF STAY**: Length of stay in the hospital.
- **EF, BNP**: Medical indicators related to heart function.
- **duration of intensive unit stay**: Duration spent in the ICU.

## Preprocessing

1. **Handling Missing Values**: Missing values in numerical columns were replaced with the column's mean.
2. **Categorical Encoding**: Applied label encoding to categorical variables.
3. **Feature Scaling**: Used `StandardScaler` to scale features for improved model performance.
4. **Handling Duplicates**: Duplicates based on the 'MRD No.' column were identified, and only the latest record was kept.

## Models and Hyperparameter Tuning

The following models were trained, and hyperparameter tuning was done using GridSearchCV:

1. **Decision Tree**
   - Tuned parameters: `max_depth`, `min_samples_split`
   
2. **Random Forest**
   - Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`
   
3. **K-Nearest Neighbors (KNN)**
   - Tuned parameters: `n_neighbors`, `weights`

Models were evaluated using cross-validation, accuracy, confusion matrix, and classification reports.

## Feature Importance

Feature importance was computed using the Random Forest model, and the results were visualized using a bar plot to identify which medical features were most influential in predicting readmission.

## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- scikit-learn
- seaborn
- matplotlib
