import io
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

# Load dataset 
file_path = 'churn.csv'
data = pd.read_csv(file_path)

# Streamlit title and description
st.title("Customer Churn Prediction Dashboard")
st.write("This interactive dashboard allows you to explore and predict customer churn.")

# Sidebar for navigation
section = st.sidebar.radio("Choose Section", ["Project Details","EDA", "Model Training & Prediction", "Predict on New Data"])

# Common preprocessing
le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])
data['Gender'] = le.fit_transform(data['Gender'])
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname','Exited'])
y = data["Exited"]

# EDA Section
if section == "EDA":
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    
    st.write("#### Data Summary")
    st.dataframe(data.describe())

    st.write("#### Data Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    info_lines = info_str.splitlines()
    info_data = [line.split(maxsplit=4) for line in info_lines[3:-1]]
    columns = ['Index', 'Column', 'Non-Null Count', 'Dtype', 'Memory Usage']
    info_df = pd.DataFrame(info_data, columns=columns)
    st.dataframe(info_df)
    
    # Visualizations
    st.write("### Visualizations")
    fig, ax = plt.subplots()
    sns.countplot(hue='Exited', x='Exited', data=data, palette='coolwarm', ax=ax)
    st.pyplot(fig)

    feature_name = st.sidebar.selectbox("Select Feature", ("Gender", "Age", "Geography", "CreditScore"))
    if feature_name == 'Age':
        st.write("##### Visualization of Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=data, x='Age', hue='Exited', multiple='stack', palette='coolwarm', ax=ax)
        st.pyplot(fig)
    elif feature_name == 'Gender':
        st.write("##### Visualization of Gender Distribution")
        data['Gender'] = le.fit_transform(data['Gender'])
        fig, ax = plt.subplots()
        sns.countplot(hue='Exited', x='Gender', data=data, palette='coolwarm', ax=ax)
        st.pyplot(fig)
    elif feature_name == 'Geography':
        st.write("##### Visualization of Geography Distribution")
        data['Geography'] = le.fit_transform(data['Geography'])
        fig, ax = plt.subplots()
        sns.countplot(hue='Exited', x='Geography', data=data, palette='coolwarm', ax=ax)
        st.pyplot(fig)
    elif feature_name == 'CreditScore':
        st.write("##### Visualization of CreditScore Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=data, x='CreditScore', hue='Exited', multiple='stack', palette='coolwarm', ax=ax)
        st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    df = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Model Training & Prediction Section
elif section == "Model Training & Prediction":
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X[:9500], y[:9500], test_size=0.2, random_state=42)
    
    classifier_name = st.sidebar.selectbox("Select Classifier", 
                                           ("Random Forest", "Logistic Regression", "K-Nearest Neighbors", 
                                            "Decision Tree", "Naive Bayes", "Neural Network"))

    if classifier_name == "Random Forest":
        model = RandomForestClassifier()
    elif classifier_name == "Logistic Regression":
        model = LogisticRegression()
    elif classifier_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif classifier_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif classifier_name == "Naive Bayes":
        model = GaussianNB()
    else:
        model = MLPClassifier()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("### Model Performance Metrics")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.3f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.3f}")
    
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Save the model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    if classifier_name == "Random Forest":
        st.write("### SHAP Analysis")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[:,:,1], X_test, plot_type="bar")
        st.pyplot(fig,bbox_inches='tight')

# Predict on New Data Section
elif section == "Predict on New Data":
    st.write("### Input Features for Prediction")
    
    # Create input fields for features
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=0.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
    is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
 
    # Manually map geography and gender to numerical values
    geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_mapping = {'Male': 1, 'Female': 0}

    # Convert input features to a DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography_mapping[geography]],
        'Gender': [gender_mapping[gender]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary]
    })

    print(input_data)
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)


    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.write("### The customer is likely to churn.")
        else:
            st.write("### The customer is not likely to churn.")
else:
    st.write("### Project Objective and Details")
    st.write("""
    **Objective:**
    The goal of this project is to predict whether a bank customer will leave the bank based on their demographics and financial information. 
    This project aims to develop your skills in classification tasks, data preprocessing, and predictive modeling.
    
    **Project Overview:**
    In this project, you will analyze a dataset containing various details about bank customers. Your main tasks are to:
    
    - **Predict Customer Churn:** Build a predictive model to estimate the likelihood of a customer leaving the bank.
    - **Feature Impact Analysis:** Identify and analyze the variables that most significantly impact customer churn.
    - **Recommendation:** Provide suggestions and recommendations based on your analysis and findings for the firm.
    
    **Dataset Description:**
    The Bank Customer Churn Prediction dataset includes the following features:
    
    - RowNumber: Row number.
    - CustomerId: Unique identification key for different customers.
    - Surname: Customer's last name.
    - CreditScore: Credit score of the customer.
    - Geography: Country of the customer.
    - Age: Age of the customer.
    - Tenure: Number of years the customer has been with the bank.
    - Balance: Bank balance of the customer.
    - NumOfProducts: Number of bank products the customer is utilizing.
    - HasCrCard: Binary flag indicating whether the customer holds a credit card with the bank.
    - IsActiveMember: Binary flag indicating whether the customer is an active member with the bank.
    - EstimatedSalary: Estimated salary of the customer in dollars.
    - Exited: Binary flag indicating if the customer closed the account with the bank (1 if closed, 0 if retained).
    """)