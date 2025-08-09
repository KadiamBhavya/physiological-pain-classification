# 3D Facial Expression Recognition

## Tech Stack
Programming Language: Python 
Machine Learning Model: Random Forest Classifier
Visualization: Matplotlib (Box plots, Time-series plots)
Libraries: scikit-learn, pandas, numpy, matplotlib

## problm statement
Classify human Pain vs No Pain states based on physiological signals such as blood pressure, EDA (electrodermal activity), and respiration rate.
Evaluate classification accuracy using 10-fold cross-validation and analyze the most discriminative statistical features.

## Solution
Preprocess physiological sensor data by extracting statistical features (mean, variance, min, max) from time-series readings.
Train a Random Forest classifier and evaluate its performance using subject-independent 10-fold cross-validation.
Visualize feature distributions and time-series plots for selected subjects.

# How to Setup
## Backend (Python Environment):
Navigate to the root project directory and create a Python virtual environment:
python -m venv env
Activate the environment:

On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate
Install dependencies:
pip install -r requirements.txt

## Running the Project:
Use the following command to start the evaluation pipeline:
python Project2.py <data_type> <dataset_path>
Arguments:
<data_type>: dia (BP Dia_mmHg), sys (LA Systolic BP_mmHg), eda (EDA_microsiemens), res (Respiration Rate_BPM), or all (use all types)
<dataset_path>: Path to the physiological dataset CSV
python Project2.py eda ./Project2Data.csv

# How Does It Work?
## Step 1: Data Loading
Reads CSV with columns: System ID, Data Type, Class, and time-series values.
Stores the first three columns in a DataFrame and keeps the time-series as a single string in a Data column.

## Step 2: Preprocessing & Feature Extraction
Converts each time-series into a numeric array.
Extracts statistical features: mean, variance, min, max.

## Step 3: Model Training
Uses a RandomForestClassifier.
Applies 10-fold K-Fold cross-validation (shuffle=True, random_state=42).
Evaluates on accuracy, precision (Pain as positive class), and recall.

## Step 4: Metrics & Visualization
Outputs:
Average confusion matrix
Accuracy, Precision, Recall scores
Box plot of extracted features
Time-series plot of selected instance (default: F001)

# Glossary
EDA: Electrodermal Activity, measures skin conductance changes related to sweat gland activity.
K-Fold Cross-Validation: Technique for validating model performance by splitting data into k folds for multiple training/testing iterations.
Statistical Features: Simple numerical summaries like mean, variance, min, max extracted from time-series data.
Confusion Matrix: Table showing correct vs incorrect predictions for each class.

# Use-Cases
P0: Healthcare monitoring — detect signs of discomfort or pain in patients in real-time.
P1: Sports science — measure pain response under physical exertion.
P2: Clinical trials — track and analyze patient pain trends.

# Solution Architecture
Input: CSV file containing physiological sensor data.
Process: Data filtering → Feature extraction → Model training → Cross-validation.
Output: Classification metrics and visual plots.






