from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from queries import queries, query_descriptions

load_dotenv()

APP_TITLE = "Telco Customer Churn Assistant"
TABLE_NAME = "telco_churn"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

# This is the approved project context the chatbot can use for non-SQL questions.
# Keep this aligned with your final README/documentation.
PROJECT_CONTEXT = """
Project: Telco Customer Churn Prediction Analysis

Purpose:
Telco is experiencing customer churn, which can negatively affect revenue stability and long-term customer growth. The project identifies factors associated with churn and evaluates machine learning models that predict whether a customer is likely to leave.

Data and methodology:
The analysis used the Telco Customer Churn dataset. The workflow included data cleaning, exploratory analysis, feature preparation, SQL analysis, and predictive modeling in Python. Non-predictive fields were removed before modeling. The target variable was customer churn. Categorical variables were encoded for machine learning.

Models:
1. Logistic Regression
- Used feature scaling because Logistic Regression is sensitive to feature magnitude.
- Applied Elastic Net regularization, combining L1 and L2 penalties.

2. Random Forest
- Ensemble tree-based model used to capture non-linear relationships.
- Hyperparameter optimization was applied.

3. XGBoost
- Gradient boosting model that builds trees sequentially.
- Class imbalance was addressed using scale_pos_weight.

Model results:
Logistic Regression: Accuracy 0.7431, Precision 0.5108, Recall 0.7620, F1 Score 0.6116, ROC-AUC 0.8463.
Random Forest Optimized: Accuracy 0.7637, Precision 0.5380, Recall 0.7754, F1 Score 0.6353, ROC-AUC 0.8508.
XGBoost: Accuracy 0.7530, Precision 0.5229, Recall 0.7941, F1 Score 0.6306, ROC-AUC 0.8517.

Interpretation:
Random Forest had the highest Accuracy and F1 Score, making it strongest for overall balanced performance. XGBoost had the highest Recall and ROC-AUC, making it useful when the goal is to identify as many potential churners as possible. Logistic Regression performed competitively and provided interpretability through coefficients.

Important features:
XGBoost top features: Contract_Two year, Tenure Months, Dependents.
Random Forest top features: Tenure Months, Contract_Two year, Internet Service_Fiber optic.
Logistic Regression key coefficients: Tenure Months -0.7879, Dependents -0.6899, Internet Service_Fiber optic +0.6188, Contract_Two year -0.6045.

Cross-model insights:
Tenure Months, Contract Type, Dependents, and Internet Service Type appeared consistently across models. Longer tenure and two-year contracts were associated with lower churn risk. Customers with dependents were associated with lower churn risk. Fiber optic internet service was associated with higher churn risk and should be investigated further.

Recommendations:
1. Encourage longer-term contracts through loyalty discounts, bundles, or service benefits.
2. Strengthen early customer retention through onboarding, check-ins, and retention campaigns.
3. Promote household and family-oriented plans.
4. Investigate fiber optic churn to understand whether pricing, reliability, expectations, support, or competition are contributing factors.

Limitations:
The analysis identifies patterns and associations, not causation. Feature importance and model coefficients show variables useful for prediction, but further business investigation is needed to confirm root causes. The models were trained on historical data, so future changes in pricing, competitors, service quality, or customer behavior may affect churn patterns.
"""


@st.cache_data
def find_dataset() -> Path:
    """Find the cleaned SQL dataset from common project locations."""
    possible_paths = [
        Path("Datasets/cleaned_telco_churn_for_sql.csv"),
        Path("cleaned_telco_churn_for_sql.csv"),
        Path("../Datasets/cleaned_telco_churn_for_sql.csv"),
    ]
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find cleaned_telco_churn_for_sql.csv. "
        "Place it in the Datasets folder or update find_dataset() in app.py."
    )


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def create_telco_engine(csv_path: str) -> Engine:
    """Create a local SQLite database with SQLAlchemy from the cleaned CSV."""
    db_dir = Path(".streamlit_cache")
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "telco_churn.db"

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    df = pd.read_csv(csv_path)
    df.to_sql(TABLE_NAME, engine, index=False, if_exists="replace")
    return engine


def run_sql(engine: Engine, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, engine)


def schema_text(engine: Engine) -> str:
    inspector = inspect(engine)
    columns = inspector.get_columns(TABLE_NAME)
    lines = [f"Table: {TABLE_NAME}", "Columns:"]
    for col in columns:
        lines.append(f'- "{col["name"]}" ({col["type"]})')
    return "\n".join(lines)


def clean_sql(sql: str) -> str:
    """Extract SQL from a model response and remove markdown fences."""
    sql = sql.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        sql = fenced.group(1).strip()
    sql = sql.strip().rstrip(";") + ";"
    return sql


def validate_read_only_sql(sql: str) -> None:
    """Only allow read-only SQLite SELECT/WITH queries."""
    lowered = sql.lower().strip()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT/WITH queries are allowed.")

    blocked_words = [
        "insert", "update", "delete", "drop", "alter", "create", "replace",
        "truncate", "attach", "detach", "pragma", "vacuum"
    ]
    for word in blocked_words:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"Blocked potentially unsafe SQL keyword: {word}")

    # Prevent multiple statements, allowing the final semicolon only.
    if ";" in sql.strip().rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read secrets safely from environment variables first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return value


def gemini_model_candidates(preferred_model: str | None) -> list[str]:
    """Return Gemini model fallbacks so the app does not break if one model name is unavailable."""
    candidates = [
        preferred_model or "",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    unique: list[str] = []
    for model in candidates:
        model = model.strip()
        if model and model not in unique:
            unique.append(model)

    return unique


class TelcoAgentState(TypedDict, total=False):
    question: str
    route: Literal["sql", "project_context"]
    schema: str
    sql: str
    result_preview: str
    answer: str
    error: str


def choose_route(question: str) -> Literal["sql", "project_context"]:
    """Simple router: SQL for data/math questions, project_context for model/methodology questions."""
    q = question.lower()
    sql_keywords = [
        "rate", "how many", "count", "average", "avg", "total", "rank", "group",
        "contract", "internet", "fiber", "tenure", "monthly charges", "high value",
        "customer", "customers", "churn by", "revenue", "lifetime value", "cltv",
        "show", "list", "compare"
    ]
    project_keywords = [
        "model", "logistic", "random forest", "xgboost", "feature", "coefficient",
        "accuracy", "precision", "recall", "f1", "roc", "auc", "methodology",
        "clean", "cleaning", "recommendation", "limitation", "why did", "which model"
    ]

    if any(word in q for word in project_keywords) and not any(word in q for word in ["rate", "count", "average", "total", "rank"]):
        return "project_context"
    if any(word in q for word in sql_keywords):
        return "sql"
    return "project_context"


@st.cache_resource
def build_agent(_engine: Engine):
    """Build a LangGraph agent for project Q&A and controlled SQL analysis."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, StateGraph

    api_key = get_secret("GOOGLE_API_KEY")
    preferred_model = get_secret("GOOGLE_MODEL", "gemini-2.5-flash")
    schema = schema_text(_engine)

    def call_gemini(prompt: str) -> str:
        """
        Call Gemini with fallback model names.

        This prevents the app from crashing when a model alias is unavailable
        for your API key/account. It tries the model in secrets.toml first,
        then common Gemini Flash models.
        """
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing from Streamlit secrets or environment variables.")

        last_error: Exception | None = None
        for model_name in gemini_model_candidates(preferred_model):
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0,
                )
                response = llm.invoke(prompt)
                return response.content
            except Exception as exc:
                last_error = exc
                message = str(exc)

                # Try the next model only for model-name/version availability errors.
                if (
                    "404" in message
                    or "NOT_FOUND" in message
                    or "not found" in message.lower()
                    or "not supported" in message.lower()
                ):
                    continue

                # For quota, invalid key, permission, network, etc., surface the real issue.
                raise

        raise RuntimeError(
            "Gemini could not run with any fallback model. "
            f"Preferred model was {preferred_model!r}. Last error: {last_error}"
        )

    def route_question(state: TelcoAgentState) -> TelcoAgentState:
        route = choose_route(state["question"])
        return {**state, "route": route}

    def generate_sql(state: TelcoAgentState) -> TelcoAgentState:
        prompt = f"""
You are a careful data analyst answering questions about a Telco customer churn project.

Use only this SQLite table and schema:
{schema}

Important context:
- Target column: "Churn Value" where 1 means churned and 0 means retained.
- Customer identifier: "CustomerID".
- The table was prepared for SQL analysis and includes encoded columns such as contract and internet service indicators.
- Answer only questions related to this Telco churn project and database.

Task:
Write one read-only SQLite SQL query that answers the user question.
Rules:
- Return SQL only, no markdown and no explanation.
- Use only SELECT or WITH queries.
- Do not modify data.
- Use double quotes around column names that contain spaces.
- Prefer aggregated results instead of returning many raw rows.
- Add LIMIT 25 when returning customer-level rows.

User question: {state['question']}
"""
        raw_sql = call_gemini(prompt)
        sql = clean_sql(raw_sql)
        return {**state, "schema": schema, "sql": sql}

    def execute_sql(state: TelcoAgentState) -> TelcoAgentState:
        try:
            sql = state["sql"]
            validate_read_only_sql(sql)
            df_result = pd.read_sql_query(text(sql), _engine)
            preview = df_result.head(30).to_string(index=False)
            return {**state, "result_preview": preview, "error": ""}
        except Exception as exc:
            return {**state, "error": str(exc), "result_preview": ""}

    def answer_from_sql(state: TelcoAgentState) -> TelcoAgentState:
        if state.get("error"):
            return {
                **state,
                "answer": (
                    "I could not safely answer that from the database. "
                    f"Reason: {state['error']}"
                ),
            }

        prompt = f"""
You are explaining results from a Telco customer churn analysis.

User question:
{state['question']}

SQL used:
{state['sql']}

Query result preview:
{state['result_preview']}

Write a brief, clear answer. Mention that the answer is based on the SQL result.
Do not invent numbers that are not shown in the result.
"""
        answer = call_gemini(prompt)
        return {**state, "answer": answer}

    def answer_from_project_context(state: TelcoAgentState) -> TelcoAgentState:
        prompt = f"""
You are a project assistant for a Telco customer churn portfolio project.

Use only the approved project context below. If the answer is not in the context, say that the project documentation does not include that detail.

Approved project context:
{PROJECT_CONTEXT}

User question:
{state['question']}

Write a brief, accurate answer for a project reviewer.
"""
        answer = call_gemini(prompt)
        return {**state, "answer": answer}

    def route_after_classifier(state: TelcoAgentState) -> str:
        return state.get("route", "project_context")

    graph = StateGraph(TelcoAgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("answer_from_sql", answer_from_sql)
    graph.add_node("answer_from_project_context", answer_from_project_context)

    graph.set_entry_point("route_question")
    graph.add_conditional_edges(
        "route_question",
        route_after_classifier,
        {
            "sql": "generate_sql",
            "project_context": "answer_from_project_context",
        },
    )
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "answer_from_sql")
    graph.add_edge("answer_from_sql", END)
    graph.add_edge("answer_from_project_context", END)
    return graph.compile()

# -----------------------------
# App layout
# -----------------------------
st.title(APP_TITLE)
st.caption("Ask questions about the Telco churn data, SQL analysis, modeling results, and project recommendations.")

try:
    data_path = find_dataset()
    df = load_data(str(data_path))
    engine = create_telco_engine(str(data_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Project Data")
    st.write(f"Dataset: `{data_path}`")
    st.write(f"Rows: **{df.shape[0]:,}**")
    st.write(f"Columns: **{df.shape[1]:,}**")
    st.divider()
    st.write("Example questions:")
    st.code("What is the churn rate by contract type?")
    st.code("Which model had the highest recall?")
    st.code("What features were most important?")
    st.code("What recommendations came from the analysis?")

# Quick metrics
metric_df = run_sql(engine, queries["overall_churn_rate"])
monthly_df = run_sql(engine, queries["approx_monthly_churn_rate"])
arpu_df = run_sql(engine, queries["average_revenue_per_month"])

col1, col2, col3 = st.columns(3)
col1.metric("Overall Churn Rate", f"{metric_df.loc[0, 'churn_rate_percentage']}%")
col2.metric("Approx. Monthly Churn", f"{monthly_df.loc[0, 'avg_monthly_churn_rate_percentage']}%")
col3.metric("ARPU", f"${arpu_df.loc[0, 'arpu']}")

st.divider()

tab1, tab2, tab3 = st.tabs(["Approved SQL Queries", "Chat with Project", "Project Context"])

with tab1:
    st.subheader("Approved SQL Query Runner")
    selected = st.selectbox(
        "Choose a query",
        options=list(queries.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    st.write(query_descriptions[selected])

    with st.expander("Show SQL"):
        st.code(queries[selected], language="sql")

    result = run_sql(engine, queries[selected])
    st.dataframe(result, use_container_width=True)

with tab2:
    st.subheader("Ask Questions About the Project")
    st.write(
        "This chatbot uses LangGraph to route questions. Data questions are converted into read-only SQL and executed with SQLAlchemy. "
        "Modeling or methodology questions are answered from the approved project context."
    )

    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    model_name = os.getenv("GOOGLE_MODEL") or st.secrets.get("GOOGLE_MODEL", "gemini-2.5-flash")

    
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    model_name = os.getenv("GOOGLE_MODEL") or st.secrets.get("GOOGLE_MODEL", "gemini-2.5-flash")

    if not api_key:
        st.warning(
            "Add your GOOGLE_API_KEY to Streamlit secrets to enable chat. "
            "The approved SQL query runner above works without an API key."
        )
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_MODEL"] = model_name


        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question about the Telco churn project."}
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask about churn, SQL results, models, or recommendations...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        agent = build_agent(engine)
                        state = agent.invoke({"question": question})
                        answer = state.get("answer", "I could not generate an answer.")
                        st.markdown(answer)

                        if state.get("route"):
                            st.caption(f"Route used: {state['route']}")
                        if state.get("sql"):
                            with st.expander("SQL generated by the agent"):
                                st.code(state["sql"], language="sql")
                                if state.get("result_preview"):
                                    st.text(state["result_preview"])
                    except Exception as exc:
                        answer = (
                            "The chatbot could not complete this request. "
                            "Your SQL dashboard still works. "
                            f"Error: `{exc}`"
                        )
                        st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab3:
    st.subheader("Approved Project Context Used by the Chatbot")
    st.info("Keep this section aligned with your README/final documentation so the chatbot answers project questions consistently.")
    st.text(PROJECT_CONTEXT)
