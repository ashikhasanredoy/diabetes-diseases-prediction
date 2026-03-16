author= Ashik Hasan Redoy
author_email=ashikhasanhredoy@gmail.com

Diabetes Prediction Model
A machine-learning based binary classification model that predicts whether a patient is likely to have diabetes based on clinical measurements.
This project uses a structured medical dataset containing 768 patient records and 9 key health-related features.

Overview
This project uses the Pima Indians Diabetes Dataset to train a classification model that predicts whether a person has diabetes (1) or not (0). The trained model is deployed as a Flask web application where users can enter their health data and receive an instant prediction.

Dataset Description:

The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains diagnostic measurements from 768 female patients of Pima Indian heritage, all aged 21 or older.
FeatureDescriptionPregnanciesNumber of times pregnantGlucosePlasma glucose concentration (2-hour oral glucose tolerance test)BloodPressureDiastolic blood pressure (mm Hg)SkinThicknessTriceps skin fold thickness (mm)Insulin2-Hour serum insulin (mu U/ml)BMIBody Mass Index (weight in kg / height in m²)DiabetesPedigreeFunctionDiabetes pedigree function (genetic risk score)AgeAge of the patient (years)OutcomeTarget variable — 1 = Diabetic, 0 = Not Diabetic



Model Details
The following classifiers are trained and compared automatically using RandomizedSearchCV (5-fold cross-validation):

.Decision Tree Classifier
.Random Forest Classifier
.AdaBoost Classifier
.Gradient Boosting Classifier
.Support Vector Classifier (SVC)
.Logistic Regression
.K-Nearest Neighbors Classifier

The model with the highest test accuracy is automatically saved. A minimum accuracy threshold of 70% is required, otherwise an exception is raised.

Technologies Used
.Python
.NumPy
.Pandas
.Matplotlib / Seaborn (optional)
.Scikit-learn
.Jupyter Notebook / Python Script
