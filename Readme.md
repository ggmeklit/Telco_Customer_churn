# Telco Customer Churn

The Challenge: Telco is experiencing customer turnover that impacts long-term revenue stability.

The Goal: To move from reactive to proactive retention by identifying the service attributes, customer behaviors, and model-driven churn indicators that predict a customer's likelihood to exit, supporting targeted retention strategies and interventions.

## Data Analysis Process

### 1. Data Cleaning

The dataset was cleaned and prepared in Python. Non-predictive fields were removed where appropriate, missing or inconsistent values were handled, and categorical variables were encoded for SQL analysis and machine learning. Separate cleaned datasets were created for SQL exploration and predictive modeling.

### 2. Initial Data Exploration

Initial data exploration was performed to understand customer churn patterns across demographic, account, service, contract, and payment-related variables. This step helped identify important churn-related patterns such as contract type, tenure, internet service type, payment method, customer value tier, and customer segment differences.

### 3. SQL Analysis

SQL was used to calculate key business metrics and churn summaries. The analysis included overall churn rate, approximate monthly churn rate, churn by contract type, churn by internet service type, high-value customer churn, average revenue per month, estimated customer lifetime value, churn by tenure, and high-risk customer ranking. These queries supported the business interpretation of churn behavior and helped prepare insights for dashboard development.

### 4. Machine Learning

Three machine learning models were trained and evaluated for churn prediction:

1. Logistic Regression
   Used feature scaling and Elastic Net regularization to support interpretability and reduce overfitting.

2. Random Forest
   Used as an optimized tree-based ensemble model to capture non-linear relationships between customer attributes and churn.

3. XGBoost
   Used as a gradient boosting model with class imbalance handling to improve churn detection.

Model performance was evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Random Forest achieved the strongest overall balance based on Accuracy and F1 Score, while XGBoost achieved the highest Recall and ROC-AUC.

Chatbot(streamlit app) was created to answer any project questions :https://telcocustomerchurn-umjvfkkrrphsuyzqjg5ymf.streamlit.app/

## Key Findings

The analysis identified several important churn drivers across SQL analysis, visual exploration, and machine learning results.

Customers on shorter-term contracts, especially month-to-month plans, showed higher churn risk compared with customers on longer-term contracts. Tenure was also a major churn indicator, with newer customers generally showing higher churn risk than long-tenure customers.

Fiber optic internet service was associated with elevated churn risk and should be investigated further to determine whether pricing, reliability, service expectations, support quality, or competitive pressure may be contributing factors.

Customers with dependents were associated with lower churn risk, suggesting that household or family-based customers may be more stable. High-value customers also showed meaningful churn behavior, making them important for retention prioritization.

Machine learning results confirmed that tenure, contract type, dependents, and internet service type were consistently important predictors of churn.

## Recommendations

Telco should focus on encouraging longer-term contracts through loyalty discounts, bundled offers, or added service benefits. The company should also strengthen early-stage customer retention through onboarding, proactive check-ins, and targeted campaigns during the first months of service.

Because fiber optic customers showed elevated churn risk, Telco should investigate this segment further to identify possible issues related to price, service quality, customer support, or competitive alternatives.

Telco should also consider expanding household and family-oriented plans, since customers with dependents were associated with lower churn risk.

## Project Links

Detailed documentation:
https://mcgill-my.sharepoint.com/:w:/g/personal/meklit_gebregiorgis_mail_mcgill_ca/IQBTcD6KX0SAQqKfPHXk2PvwAQWVoisCuv8afpEGIjNTfqE?e=zG0eW2

Streamlit app:
https://telcocustomerchurn-umjvfkkrrphsuyzqjg5ymf.streamlit.app/

Power BI visualization:
https://mcgill-my.sharepoint.com/:u:/g/personal/meklit_gebregiorgis_mail_mcgill_ca/IQCHvCRiG7R0Q4gX2PQVJZ8rAfYsHdlcFf2eZJ-_wo1uQZE?e=yrLEon

## File Organization

```text
Project 1 Telco Customer Churn/
│
├── 1. Customer_churn/
│   │
│   ├── Datasets/
│   │   ├── Telco_customer_churn.xlsx
│   │   ├── cleaned_telco_churn_for_sql.csv
│   │   └── cleaned_telco_churn_predictive.csv
│   │
│   ├── sql_outputs/
│   │   └── exported SQL query result files
│   │
│   ├── telco_streamlit_langgraph_app/
│   │   ├── .streamlit/
│   │   │   └── secrets.toml
│   │   ├── app.py
│   │   ├── queries.py
│   │   ├── requirements.txt
│   │   ├── README_SETUP.md
│   │   └── .env.example
│   │
│   ├── 1.Telco_customer_churn_cleaning_IDA.ipynb
│   ├── 2.Telco_SQL_Analysis_Notebook.ipynb
│   ├── 3.Models.ipynb
│   └── Readme.md
│
├── Power BI Dashboard/
│   └── Telco churn Power BI dashboard files
│
├── SQL queries/
│   └── Individual SQL query files
│
├── Customer Churn Prediction.docx
├── Telco Customer Churn Analysis.docx
└── README.md

```

## Notes

The project combines descriptive analysis, SQL-based business metrics, machine learning prediction, dashboard visualization, and a Streamlit chatbot assistant. The goal is to provide both technical analysis and business-facing retention recommendations.
