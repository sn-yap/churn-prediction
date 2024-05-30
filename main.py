# Import the required packages and modules
# Data Manipulation
import numpy as np
import pandas as pd
from functools import reduce

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# from IPython.display import IFrame

# Data Preprocessing
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# Modeling
from kmodes.kprototypes import KPrototypes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import time
# from memory_profiler import memory_usage

# Deployment
import joblib
import streamlit as st

# Others
from datetime import datetime, date
from io import StringIO
import csv
import os


# Define the base path
base_path = "./"

# Load the preprocessor pipeline
preprocessor = joblib.load(base_path + 'preprocessor_pipeline.pkl')

# Load the models
lr = joblib.load(base_path + 'Logistic Regression_original.joblib')
knn = joblib.load(base_path + 'KNeighborsClassifier_original.joblib')
gb = joblib.load(base_path + 'GaussianNB_original.joblib')
dt = joblib.load(base_path + 'Decision Tree_original.joblib')
rf = joblib.load(base_path + 'Random Forest_original.joblib')

# Define the path to the Excel file
excel_file_path = "data_warehouse.xlsx"


st.title("Attrition is Reversible: Unlocking Potential through Integrated Customer Segmentation and Churn Prediction in Telecom")


# Define the tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Home", "Data Upload", "Data Preprocessing", "Churn Prediction", "Insights and Actions", "Data Warehouse"])


# Content for Tab 1 - Home
with tab1:
    st.write("""
    ## Home
    
    Use this application to predict customer churn and develop effective retention strategies.
    
    Welcome to our Telecom Churn Prediction Application, designed to help telecom providers retain their valuable customers by accurately predcting churn. 
    In the highly competitive telecom industry, customer attrition, or churn, can significantly impact revenue and growth.
    Our application leverages advanced machine learning techniques and integrated customer segmentation to provide actionable insights and enable proactive customer retention strategies.
    
    ### Steps to Use the Web Application
    
    #### Data Upload
    - Navigate to the data upload section.
    - Choose your preferred method for data input:
        - **Manually Input Data:** Enter customer data manually in the provided form and submit.
        - **Download Template and Upload CSV:** Download the provided template, fill in your customer data, and upload the completed CSV file.
        - **Upload CSV:** Directly upload your customer data file in the specified format.
    - Ensure that data contains all required fields for accurate prediction.
    
    #### Data Preprocessing 
    - This application will preprocess the uploaded data, handling missing values, and normalizing the data as needed.
    - Review the preprocessed data summary to ensure accuracy.
        
    #### Churn Prediction 
    - Initiate the churn prediction process.
    - The application will apply mahcine leanring algorithms to predict the likelihood of churn for each customer
    - Review the prediction results, which will include churn labels for each customer
        
    #### Insights and Actions
    - Access detailed reports and visualizations to understand the factors contributing to churn.
    - Use the insights to develop targeted retention startegies for high-risk customers.

    #### Data Warehousing
    - Consolidate and store customer data from various sources into a centralized data warehouse.
    - Perform regular data extraction, transformation, and loading (ETL) processes to keep the data warehouse up-to-date.
    - Utilize the data warehouse to generate comprehensive reports and insights for informed decision-making.

    Thank you for choosing our Telecom Churn Prediction Web Application to empower your customer retention efforts. We look forward to helping you achieve your business goals.
    
    ***Disclaimer***

    *The predictions and insights provided by our application are based on the data supplied by users and the performance of our machine learning models. While we strive to deliver accurate predictions, we cannot guarantee absolute accuracy. The application is intended to assist with customer retention strategies and should not be used as the sole basis for critical business decisions. Users are advised to use their judgment and consider multiple factors when interpreting the results.*    
    """)


# Content for Tab 2 - Data Upload
with tab2:
    st.header("Data Upload")
    st.write("Use this section to either manually input data or upload your customer data.")

    # Choice for the user to select input method
    input_method = st.radio("Select Input Method:", ("Manual Input", "Upload CSV"))

    # Sample template data
    template_data = {
        "Customer ID": [],
        "Satisfaction Score": [],
        "Number of Referrals": [],
        "Tenure in Months": [],
        "Offer": [],
        "Avg Monthly Long Distance Charges": [],
        "Multiple Lines": [],
        "Internet Type": [],
        "Avg Monthly GB Download": [],
        "Online Security": [],
        "Online Backup": [],
        "Device Protection Plan": [],
        "Premium Tech Support": [],
        "Streaming TV": [],
        "Streaming Movies": [],
        "Streaming Music": [],
        "Unlimited Data": [],
        "Contract": [],
        "Paperless Billing": [],
        "Payment Method": [],
        "Monthly Charge": [],
        "Total Charges": [],
        "Total Refunds": [],
        "Total Extra Data Charges": [],
        "Total Long Distance Charges": [],
        "Total Revenue": [],
        "Age": [],
        "Married": [],
        "Number of Dependents": [],
        "Region": []
    }

    relevant_columns = list(template_data.keys())

    def generate_template_csv():
        output = StringIO()
        csv_writer = csv.DictWriter(output, fieldnames=template_data.keys())
        csv_writer.writeheader()
        return output.getvalue()

    st.download_button(
        label="Download Template",
        data=generate_template_csv(),
        file_name="data_template.csv",
        mime="text/csv"
    )


    if input_method == "Manual Input":
        st.write("Please fill out the following fields to manually input data:")

        with st.expander("Customer Demographics"):
            customer_id = st.text_input("Customer ID")
            # Check if customer ID is empty or not
            if customer_id.strip():
                id_flag = True
            else:
                id_flag = False
                st.error("The Customer ID field is required. Kindly ensure it is not left empty before submitting.")


            dob = st.date_input(
                "Date of Birth",
                min_value=date(1900, 1, 1),
                max_value=datetime.now().date()
            )
            today = datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

            married = st.selectbox("Marital Status (Married)", ["Yes", "No"])

            has_dependents = st.selectbox("Dependents", ["Yes", "No"])
            if has_dependents == "Yes":
                number_of_dependents = st.number_input("Number of Dependents", min_value=1)
            else:
                number_of_dependents = 0


        with st.expander("Customer Location"):
            region = -1
            zip_code = st.text_input("Zip Code (5-digit)")
            if len(zip_code) != 5 or int(zip_code[1]) > 6:
                st.error("The zip code entered must be 5 digits in length. Please enter a valid 5-digit zip code.")
                code_flag = False
            else:
                region = int(zip_code[1])
                code_flag = True


        with st.expander("Service and Subscription Details"):
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])

            col1, col2 = st.columns(2)
            with col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                if phone_service == "Yes":
                    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

                online_security = st.selectbox("Online Security", ["Yes", "No"])
                device_protection_plan = st.selectbox("Device Protection Plan", ["Yes", "No"])
            with col2:
                internet_service = st.selectbox("Internet Service", ["Yes", "No"])
                if internet_service == "Yes":
                    internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic"])

                online_backup = st.selectbox("Online Backup", ["Yes", "No"])
                premium_tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])

            col3, col4, col5 = st.columns(3)
            with col3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            with col4:
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            with col5:
                streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])


        with st.expander("Customer Financial Profile"):
            col6, col7 = st.columns(2)
            with col6:
                avg_monthly_long_distance_charges = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0, step=0.01)
                total_long_distance_charges = st.number_input("Total Long Distance Charges", min_value=0.0, step=0.01)
                monthly_charge = st.number_input("Monthly Charge", min_value=0.0, step=0.01)
                total_revenue = st.number_input("Total Revenue", min_value=0.0, step=0.01)
                tenure_in_months = st.number_input("Tenure in Months", min_value=0)
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

            with col7:
                avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0, step=0.01)
                total_extra_data_charges = st.number_input("Total Extra Data Charges", min_value=0.0, step=0.01)
                total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)
                total_refunds = st.number_input("Total Refunds", min_value=0.0, step=0.01)
                contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
                payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])


        with st.expander("Customer Engagement Metrics"):
            has_referrers = st.selectbox("Referrers", ["Yes", "No"])
            if has_referrers == "Yes":
                number_of_referrers = st.number_input("Number of Referrers", min_value=1)
            else:
                number_of_referrers = 0

            offer = st.selectbox("Offer", ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E", "No"])
            satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5)


        if st.button("Submit", disabled=not (id_flag and code_flag)):
            # Create a DataFrame from the input
            data = {
                "Customer ID": [customer_id],
                "Satisfaction Score": [satisfaction_score],
                "Number of Referrals": [number_of_referrers],
                "Tenure in Months": [tenure_in_months],
                "Offer": [offer],
                "Avg Monthly Long Distance Charges": [avg_monthly_long_distance_charges],
                "Multiple Lines": [multiple_lines],
                "Internet Type": [internet_type],
                "Avg Monthly GB Download": [avg_monthly_gb_download],
                "Online Security": [online_security],
                "Online Backup": [online_backup],
                "Device Protection Plan": [device_protection_plan],
                "Premium Tech Support": [premium_tech_support],
                "Streaming TV": [streaming_tv],
                "Streaming Movies": [streaming_movies],
                "Streaming Music": [streaming_music],
                "Unlimited Data": [unlimited_data],
                "Contract": [contract],
                "Paperless Billing": [paperless_billing],
                "Payment Method": [payment_method],
                "Monthly Charge": [monthly_charge],
                "Total Charges": [total_charges],
                "Total Refunds": [total_refunds],
                "Total Extra Data Charges": [total_extra_data_charges],
                "Total Long Distance Charges": [total_long_distance_charges],
                "Total Revenue": [total_revenue],
                "Age": [age],
                "Married": [married],
                "Number of Dependents": [number_of_dependents],
                "Region": [region]
            }
            df = pd.DataFrame(data)
            df['Customer ID'] = customer_id
            df['Region'] = df['Region'].astype(str)

            # Store the data in the session state
            st.session_state['uploaded_data'] = df
            st.success("Data submitted successfully!")

    else:
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            # Display the preview of the uploaded data
            st.write("**Preview of Uploaded Data**")

            # Flag to track data validity
            if any((df['Region'] > 6) | (df['Region'] < 0)):
                flag = False

                # Highlight invalid records
                styled_df = df.style.apply(
                    lambda x: ['background-color: yellow' if (x['Region'] > 6 or x['Region'] < 0) and not flag else '' for _
                               in x], axis=1)
                st.write(styled_df)
                st.error("The uploaded file contains invalid records. Please correct the highlighted rows.")
            else:
                st.write(df)
                flag = True

            if st.button("Submit", disabled=not flag):
                # Filter the columns to keep only the required ones
                df = df[relevant_columns]
                df['Region'] = df['Region'].astype(str)

                # Store the data in the session state
                st.session_state['uploaded_data'] = df
                st.success("File uploaded successfully!")


# Content for Tab 3 - Data Preporcessing
with tab3:
    st.header("Data Preporcessing")
    st.write("Preprocess the uploaded data.")

    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']

        # Ensure columns are in the correct order
        numerical_cols = ['Satisfaction Score', 'Number of Referrals', 'Tenure in Months',
                          'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
                          'Monthly Charge', 'Total Charges', 'Total Refunds',
                          'Total Extra Data Charges', 'Total Long Distance Charges',
                          'Total Revenue', 'Age', 'Number of Dependents']
        categorical_cols = ['Offer', 'Multiple Lines', 'Internet Type', 'Online Security',
                            'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
                            'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data',
                            'Contract', 'Paperless Billing', 'Payment Method', 'Married', 'Region']

        # Separate numerical and categorical data
        df_num = df[numerical_cols]
        df_cat = df[categorical_cols]

        # Display the uploaded data
        st.write("**Uploaded Data**")
        st.write(df)

        # Preprocesses the data using the preprocessor pipeline and displays the result
        st.write("**Preprocessed Data**")
        preprocessed_data = preprocessor.transform(df)
        st.write(preprocessed_data)
    else:
        st.write("Please upload your data in the 'Data Upload' tab.")


# Content for Tab 4 - Churn Prediction
with tab4:
    # Function to display model predictions
    def display_model_prediction(col, model_name, prediction):
        col.markdown(f"""
        <div style="text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
            <strong>{model_name}</strong>
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            col.success("All clear: Low churn risk", icon="✅")
        else:
            col.error("Alert: High churn risk!", icon="⚠️")


    # Function to highlight rows where final_predictions is 1
    def highlight_high_churn(s):
        return ['background-color: yellow' if s['final_predictions'] == 1 else '' for _ in s]


    # Function to save record to the data warehouse
    def save_record_to_warehouse(df):
        # Append the record to the data warehouse DataFrame
        record = df.copy()
        st.session_state.data_warehouse = st.session_state.data_warehouse.append(record, ignore_index=True)

        # Save the updated data warehouse DataFrame to the Excel file
        st.session_state.data_warehouse.to_excel(excel_file_path, index=False)
        st.success("Record updated to data warehouse.")


    st.header("Churn Prediction")
    st.write("Upload your data to predict customer churn.")

    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']

        # Make predictions using all models
        lr_predictions = lr.predict(df)
        knn_predictions = knn.predict(df)
        gb_predictions = gb.predict(df)
        dt_predictions = dt.predict(df)
        rf_predictions = rf.predict(df)

        # Append predictions to the DataFrame
        df['lr_predictions'] = lr_predictions
        df['knn_predictions'] = knn_predictions
        df['gb_predictions'] = gb_predictions
        df['dt_predictions'] = dt_predictions
        df['rf_predictions'] = rf_predictions

        # Determine the majority vote for each transaction
        final_predictions = [np.argmax(np.bincount([lr_pred, knn_pred, gb_pred, dt_pred, rf_pred]))
                             for lr_pred, knn_pred, gb_pred, dt_pred, rf_pred
                             in zip(lr_predictions, knn_predictions, gb_predictions, dt_predictions, rf_predictions)]
        df['final_predictions'] = final_predictions

        if len(df) == 1:
            model_names = ['Logistic Regression', 'K Neighbors', 'Gradient Boosting', 'Decision Tree', 'Random Forest',
                           'Final Prediction']
            predictions = [lr_predictions[0], knn_predictions[0], gb_predictions[0], dt_predictions[0],
                           rf_predictions[0], final_predictions[0]]

            col1, col2, col3 = st.columns(3)
            for col, model_name, prediction in zip([col1, col2, col3], model_names[:3], predictions[:3]):
                display_model_prediction(col, model_name, prediction)

            col4, col5, col6 = st.columns(3)
            for col, model_name, prediction in zip([col4, col5, col6], model_names[3:], predictions[3:]):
                display_model_prediction(col, model_name, prediction)

        else:
            # Display the styled DataFrame
            st.write(df.style.apply(highlight_high_churn, axis=1))

        # Provide a download link for the DataFrame with predictions
        csv = df.to_csv(index=False).encode('utf-8')

        # Create two columns
        update_column, u3, u4, u5, u6, u7, export_column = st.columns(7)

        # Export button to export the current record
        if export_column.button("Export"):
            st.download_button(
                label="Export",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

        # Update button to save all records to the data warehouse
        if update_column.button("Update"):
            save_record_to_warehouse(df)

    else:
        st.write("Please upload your data in the 'Data Upload' tab.")


# Content for Tab 5 - Insights and Actions
with tab5:
    st.header("Insights and Actions")
    st.write("Generate and view detailed reports based on your data.")

    # Embed the Power BI dashboard
    st.components.v1.iframe(src = "https://app.powerbi.com/view?r=eyJrIjoiYWVkZDE5YTEtODdjOS00Mjc4LTk4YTMtNjQ1ZGNlMTExNzhmIiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D", width = 705, height = 486)


# Content for Tab 6 - Data Warehouse
with tab6:
    st.header("Data Warehouse")
    st.write("Create, read, and update records stored in the data warehouse.")

    # Check if the Excel file exists
    if os.path.exists(excel_file_path):
        # Read records from the Excel file
        data_warehouse_df = pd.read_excel(excel_file_path)
    else:
        # Create an empty DataFrame if the Excel file doesn't exist
        data_warehouse_df = pd.DataFrame(columns=relevant_columns)

    # Initialize data_warehouse in session state if it doesn't exist
    if 'data_warehouse' not in st.session_state:
        st.session_state.data_warehouse = data_warehouse_df

    # Display the data warehouse DataFrame
    st.write(st.session_state.data_warehouse)

    # Provide a download link for the data warehouse DataFrame
    csv_data = st.session_state.data_warehouse.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export All Records",
        data=csv_data,
        file_name="data_warehouse.csv",
        mime="text/csv"
    )

    # Save the data warehouse DataFrame to an Excel file
    data_warehouse_df.to_excel(excel_file_path, index=False)
