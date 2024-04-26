import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load datasets
def load_data(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Digits':
        data = datasets.load_digits()
    else:
        data = None
    return data

# Sidebar for dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Digits"))
dataset = load_data(dataset_name)

# Sidebar for classifier selection
classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Neural Network", "Naive Bayes"))

# Sidebar for input method selection
input_method = st.sidebar.selectbox("Select Input Method", ("Single Feature Input", "List Input"))

# Function to get classifier based on selection
def get_classifier(cl_name):
    if cl_name == "Logistic Regression":
        # Logistic Regression:
        # - Simple to implement and fast to train.
        # - Good baseline model for binary classification problems.
        # - Assumes a linear relationship between the input features and the log-odds of the target being 1.
        clf = LogisticRegression()
    elif cl_name == "Neural Network":
        # Neural Network (MLPClassifier):
        # - Capable of capturing complex nonlinear relationships between features.
        # - Requires more data and computational power compared to simpler models.
        # - Can overfit on small datasets without proper regularization and tuning.
        clf = MLPClassifier(max_iter=1000)
    elif cl_name == "Naive Bayes":
        # Naive Bayes:
        # - Based on Bayes' Theorem with an assumption of independence among predictors.
        # - Fast and effective for large datasets.
        # - Works well with categorical input features, often used in text classification.
        clf = GaussianNB()
    return clf

clf = get_classifier(classifier_name)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1234)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
clf.fit(X_train_scaled, y_train)

# Prediction Section
st.title('Model Prediction Interface')

# Input fields generation based on user selection
if input_method == "Single Feature Input":
    inputs = [st.number_input(f"{feature}", step=0.01) for feature in dataset.feature_names]
else:  # List Input
    input_string = st.text_input("Enter features as a comma-separated list")
    inputs = list(map(float, input_string.split(','))) if input_string else []

if st.button("Predict"):
    if len(inputs) == len(dataset.feature_names):
        inputs_scaled = scaler.transform([inputs])
        prediction = clf.predict(inputs_scaled)
        st.write(f"Prediction: {dataset.target_names[prediction[0]]}")
    else:
        st.error(f"Please input the correct number of features: {len(dataset.feature_names)}")

# Display accuracy of the model
predictions = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
st.write(f"Model Accuracy: {accuracy:.2f}")
