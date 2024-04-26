This Streamlit application enables users to interact with machine learning models to predict outcomes based on the Iris and Digits datasets from sklearn.datasets.

Features
Dataset Selection: Choose between the Iris and Digits datasets.
Model Selection: Select from three classifiers: Logistic Regression, Neural Network, or Naive Bayes.
Input Methods: Input feature values one at a time or all at once using a comma-separated list.
Dynamic Predictions: Based on the provided inputs and selected model, the application predicts and displays the outcome.
Model Accuracy Display: Shows the accuracy of the selected classifier based on the test data.

How to Run the Application:
1. Dowmload the folder
2. go to the file location
3. Run the following command to create a virtual environment: python -m venv venv
4. Activate the virtual environment:
    On Windows:
        venv\Scripts\activate
    On macOS/Linux:
        source venv/bin/activate
3. pip install -r requirements.txt # This will install all required libraries
4. run the application -> streamlit run app.py

