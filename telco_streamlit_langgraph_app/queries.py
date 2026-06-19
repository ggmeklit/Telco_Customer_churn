"""Reusable SQL queries for the Telco churn project."""

queries = {
    "overall_churn_rate": """
SELECT
    COUNT("CustomerID") AS total_customers,
    SUM("Churn Value") AS total_churned,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(COUNT("CustomerID"), 0)) * 100, 2) AS churn_rate_percentage
FROM telco_churn;
""",

    "approx_monthly_churn_rate": """
SELECT
    SUM("Churn Value") AS total_churned,
    SUM("Tenure Months") AS total_customer_months,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(SUM("Tenure Months"), 0)) * 100, 2) AS avg_monthly_churn_rate_percentage
FROM telco_churn;
""",

    "churn_by_contract_type": """
SELECT
    CASE
        WHEN "Contract_One year" = 1 THEN 'One Year Contract'
        WHEN "Contract_Two year" = 1 THEN 'Two Year Contract'
        ELSE 'Month-to-Month Contract'
    END AS contract_type,
    SUM("Churn Value") AS total_churned,
    COUNT("CustomerID") AS total_customers,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(COUNT("CustomerID"), 0)) * 100, 2) AS churn_rate_percentage
FROM telco_churn
GROUP BY 1
ORDER BY churn_rate_percentage DESC;
""",

    "churn_by_internet_type": """
SELECT
    CASE
        WHEN "Internet Service_Fiber optic" = 1 THEN 'Fiber Optic'
        WHEN "Internet Service_No" = 1 THEN 'No Internet Service'
        ELSE 'DSL'
    END AS internet_type,
    SUM("Churn Value") AS total_churned,
    COUNT("CustomerID") AS total_customers,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(COUNT("CustomerID"), 0)) * 100, 2) AS churn_rate_percentage
FROM telco_churn
GROUP BY 1
ORDER BY churn_rate_percentage DESC;
""",

    "high_value_customer_churn_rate": """
WITH ThresholdCalc AS (
    SELECT "Monthly Charges" AS cutoff_value
    FROM telco_churn
    ORDER BY "Monthly Charges"
    LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.75 AS INTEGER) FROM telco_churn)
)
SELECT
    CASE
        WHEN "Monthly Charges" >= (SELECT cutoff_value FROM ThresholdCalc) THEN 'High Value'
        ELSE 'Other'
    END AS customer_segment,
    COUNT("CustomerID") AS total_customers,
    SUM("Churn Value") AS total_churned,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(COUNT("CustomerID"), 0)) * 100, 2) AS churn_rate_percentage
FROM telco_churn
GROUP BY 1
ORDER BY churn_rate_percentage DESC;
""",

    "high_value_customer_summary": """
WITH ThresholdCalc AS (
    SELECT "Monthly Charges" AS cutoff_value
    FROM telco_churn
    ORDER BY "Monthly Charges"
    LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.75 AS INTEGER) FROM telco_churn)
)
SELECT
    COUNT("CustomerID") AS total_high_value_customers,
    SUM("Churn Value") AS total_high_value_churned,
    ROUND((SUM("Churn Value") * 1.0 / NULLIF(COUNT("CustomerID"), 0)) * 100, 2) AS high_value_churn_rate_percentage
FROM telco_churn
WHERE "Monthly Charges" >= (SELECT cutoff_value FROM ThresholdCalc);
""",

    "average_revenue_per_month": """
SELECT
    ROUND(AVG("Monthly Charges"), 2) AS arpu
FROM telco_churn;
""",

    "historical_average_revenue_per_month": """
SELECT
    ROUND(SUM("Total Charges") / NULLIF(SUM("Tenure Months"), 0), 2) AS historical_arpu
FROM telco_churn;
""",

    "estimated_customer_lifetime_value": """
SELECT
    ROUND(AVG("Monthly Charges") * AVG("Tenure Months"), 2) AS estimated_avg_cltv
FROM telco_churn;
""",

    "historical_customer_lifetime_value": """
SELECT
    ROUND(AVG("Total Charges"), 2) AS historical_avg_cltv
FROM telco_churn;
""",

    "churn_by_tenure": """
WITH ChurnByTenure AS (
    SELECT
        "Tenure Months",
        SUM("Churn Value") AS churned_at_tenure
    FROM telco_churn
    GROUP BY "Tenure Months"
)
SELECT
    "Tenure Months",
    churned_at_tenure,
    SUM(churned_at_tenure) OVER (ORDER BY "Tenure Months") AS cumulative_churned,
    ROUND((SUM(churned_at_tenure) OVER (ORDER BY "Tenure Months") * 1.0 /
           (SELECT COUNT("CustomerID") FROM telco_churn)) * 100, 2) AS true_running_churn_percentage
FROM ChurnByTenure
ORDER BY "Tenure Months";
""",

    "high_risk_customers_ranked": """
SELECT
    "CustomerID",
    "Monthly Charges",
    "Payment Risk Category",
    RANK() OVER(ORDER BY "Monthly Charges" DESC) AS monthly_charges_rank
FROM telco_churn
WHERE "Payment Risk Category" = 'High Risk'
ORDER BY monthly_charges_rank
LIMIT 25;
"""
}

query_descriptions = {
    "overall_churn_rate": "Overall churn rate: total churned customers divided by total customers.",
    "approx_monthly_churn_rate": "Approximate monthly churn rate: total churned customers divided by total customer tenure months.",
    "churn_by_contract_type": "Churn by contract type: churned customers divided by total customers within each contract category.",
    "churn_by_internet_type": "Churn by internet type: churned customers divided by total customers within each internet service category.",
    "high_value_customer_churn_rate": "High-value customer churn rate: compares churn for top 25% monthly spenders against all other customers.",
    "high_value_customer_summary": "High-value customer summary: summarizes count, churned count, and churn rate for top 25% monthly spenders.",
    "average_revenue_per_month": "Average revenue per month: average monthly charge across all customers.",
    "historical_average_revenue_per_month": "Historical average revenue per month: total charges divided by total tenure months.",
    "estimated_customer_lifetime_value": "Estimated customer lifetime value: average monthly charges multiplied by average tenure months.",
    "historical_customer_lifetime_value": "Historical customer lifetime value: average total charges per customer.",
    "churn_by_tenure": "Churn by tenure: churned customers grouped by tenure month with a running cumulative churn percentage.",
    "high_risk_customers_ranked": "High-risk customers ranked: high-risk customers ranked by highest monthly charges for retention prioritization.",
}
