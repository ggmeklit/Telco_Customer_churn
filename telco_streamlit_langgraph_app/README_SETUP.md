# Telco Streamlit LangGraph SQL Assistant

## Purpose

This app lets reviewers or stakeholders ask questions about the Telco churn project. It has three parts:

1. **Approved SQL Query Runner**: runs your fixed SQL analyses.
2. **Chat with Project**: uses LangGraph to route questions either to SQL analysis or to the approved project context.
3. **Project Context**: stores the final approved documentation summary that the chatbot uses for methodology/model questions.

## How the chatbot works

- **LangGraph** controls the workflow.
- **SQLAlchemy** creates and queries a local SQLite database from your cleaned CSV.
- **SQL agent step** converts data questions into read-only SQL.
- **Project-context step** answers model, cleaning, methodology, and recommendation questions from your approved documentation.

This first version does not use MCP. MCP is useful later if you want to connect many external tools, but it is not necessary for a focused Streamlit portfolio app.

## Where people access it

- While developing locally, you access it in your browser at `http://localhost:8501`.
- Other people can access it only after you deploy it, for example through Streamlit Community Cloud, Render, or another hosting platform that gives you a shareable URL.

## Folder placement

Place these files in your main project folder:

```text
Project 1 Telco Customer Churn/
├── app.py
├── queries.py
├── requirements.txt
├── .env
├── Datasets/
│   └── cleaned_telco_churn_for_sql.csv
```

## Install packages

From your project folder, run:

```bash
pip install -r requirements.txt
```

## Add API key for chatbot

Create a `.env` file in the project folder:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

The approved SQL query runner works without an API key. The chatbot tab needs the API key.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example questions

SQL/data questions:

- What is the churn rate by contract type?
- Which internet service type has the highest churn rate?
- What is the average monthly revenue?
- Rank high-risk customers by monthly charges.
- How does churn change by tenure?

Project/model questions:

- Which model performed best?
- Which model had the highest recall?
- What features were most important?
- Why does recall matter for churn prediction?
- What recommendations came from the analysis?
