import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle  # Importing pickle for saving and loading models


data = pd.read_csv('dataset/training_data.csv').dropna(axis=1)
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Training and saving the SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
pickle.dump(svm_model, open('models/svm_model.pkl', 'wb'))  # Save the SVM model

# Training and saving the Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
pickle.dump(nb_model, open('models/nb_model.pkl', 'wb'))  # Save the Naive Bayes model

# Training and saving the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))  # Save the Random Forest model
