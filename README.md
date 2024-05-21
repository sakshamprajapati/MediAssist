```markdown
# MediAssist 🩺

Welcome to MediAssist, a machine learning-based health assistant that helps predict diseases based on symptoms, provides disease descriptions and precautions, and suggests doctors available for consultation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

## Overview
MediAssist leverages machine learning models—Support Vector Machine (SVM), Naive Bayes, and Random Forest—to predict diseases based on user-input symptoms. It also provides a brief description of the disease, precautionary measures, and a list of doctors available for consultation.

## Features
- Disease prediction using symptoms
- Description and precautions for the predicted disease
- Doctor recommendation based on the predicted disease
- Web application interface using Streamlit

## Setup

### Prerequisites
Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MediAssist.git
   cd MediAssist
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset files (`doctor.csv`, `symptom_Description.csv`, `symptom_precaution.csv`, `training_data.csv`) in the `dataset` directory.

### Requirements
- pandas
- numpy
- scikit-learn
- streamlit
- pickle

## Usage

1. Train the models and save them:
   ```bash
   python train_models.py
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Training
The `train_models.py` script is used to train the machine learning models and save them using pickle. Below is the code to train and save the models:

```python
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data
data = pd.read_csv('dataset/training_data.csv').dropna(axis=1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train and save models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    pickle.dump(model, open(f'models/{model_name.lower().replace(" ", "_")}_model.pkl', 'wb'))
```

## Dataset
- `doctor.csv`: Contains information about doctors.
- `symptom_Description.csv`: Contains descriptions of symptoms.
- `symptom_precaution.csv`: Contains precautionary measures for symptoms.
- `training_data.csv`: Training data for the models.
- `test_data.csv`: Data for testing the models (optional).

## Acknowledgements
This project utilizes various datasets and libraries to provide its functionality. Special thanks to the open-source community and contributors of the libraries used.

## Disclaimer
This application is an ML model and should not be used as a substitute for professional medical advice. Always consult with a real doctor for accurate diagnosis and treatment.
```

Feel free to customize the content according to your project's specific details and needs.
