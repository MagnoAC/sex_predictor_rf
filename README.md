# sex_predictor_rf
Exploratory data analysis of patients data and a random forest model to predict the sex of those patients

# Install

You will need to have a software to execute a Jupyter Notebook.

If you do not have Python installed, it is highly recommended that you install the Anaconda distribution of Python (3.6.x version is preferred).

If you have those requirements installed:
  
  1. Pull all files from this repository
  
  2. cd to the directory where "requirements.txt" can be found
  
  3. create and activate a virtual env:
```
conda create --name env
conda activate env
```
4. run on your anaconda prompt: 
```
pip3 install -r requirements.txt on anaconda prompt
```
# Code

The code used for analyse data and to create the Random forest model can be seen on ```"Sex Predictor Exploratory Data Analysis + Modelling.ipynb"``` file

# How to run sex_predictory.py

1. Activate your env where you installed the requirements.txt
2. Move the patients csv file (newsample.csv) you want to predict into the repository you have pulled
3. Make sure the rf_model.pkl file is on the same repo
4. run this script to use the predictor
```
(env) python sex_predictor.py --input_file newsample.csv
```
Obs: If accuse error of not having imblearn lib, run this command:
```
conda install -c glemaitre imbalanced-learn
```

# Data
The patients data used to build the model contains 288 rows and 18 variables (17 features and 1 target variable)

### Features

- age: in years
- cp: chest pain type
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholesterol in mg/dl
- fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- restecg: resting electrocardiographic results
- thalach: maximum heart rate achieved
- nar: number of arms
- exang: exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- hc: patient's hair colour
- sk: patient's skin colour
- trf: time spent in traffic daily (in seconds)
- ca: number of major vessels (0-3) colored by flourosopy
- thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
### Target Variable 
- sex: (M = male; F = female)
