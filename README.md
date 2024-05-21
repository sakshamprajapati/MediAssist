```markdown
# MediAssist 🩺

Welcome to **MediAssist**, your AI-powered health assistant designed to predict diseases based on your symptoms, provide useful descriptions and precautionary measures, and recommend doctors for consultation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)

## Overview
MediAssist is a sophisticated health assistant that leverages machine learning models to analyze user-input symptoms and predict possible diseases. It not only provides predictions but also offers descriptions, precautions, and doctor recommendations.

## Features
- **Disease Prediction**: Get predictions based on symptoms using SVM, Naive Bayes, and Random Forest models.
- **Detailed Information**: Access descriptions and precautions for predicted diseases.
- **Doctor Recommendations**: Find doctors who can help with the predicted disease.
- **User-Friendly Interface**: Easy-to-use web application built with Streamlit.

## Installation

### Prerequisites
Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MediAssist.git
   cd MediAssist
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**:
   Place the following dataset files in the `dataset` directory:
   - `doctor.csv`
   - `symptom_Description.csv`
   - `symptom_precaution.csv`
   - `training_data.csv`

## Usage

1. **Train and Save Models**:
   ```bash
   python train_models.py
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Navigate to the App**:
   Open your browser and go to `http://localhost:8501` to interact with MediAssist.

## Model Training
The `train_models.py` script trains the machine learning models and saves them for future predictions. Below is the code snippet:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
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

## Datasets
- **doctor.csv**: Information about doctors.
- **symptom_Description.csv**: Descriptions of various symptoms.
- **symptom_precaution.csv**: Precautionary measures for symptoms.
- **training_data.csv**: Training data for model development.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer
This application is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.
```
