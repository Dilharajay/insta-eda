"""
DataNarrator — LangChain EDA Agent
Uses LangGraph's create_react_agent (compatible with LangChain 1.x+)
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from agent.tools import (
    get_data_shape,
    get_missing_values,
    get_descriptive_stats,
    get_outlier_detection,
    get_correlation_analysis,
    get_categorical_analysis,
    get_ml_recommendation,
)

# All tools the agent can use
EDA_TOOLS = [
    get_data_shape,
    get_missing_values,
    get_descriptive_stats,
    get_outlier_detection,
    get_correlation_analysis,
    get_categorical_analysis,
    get_ml_recommendation,
]

SYSTEM_PROMPT = """
You are DataNarrator, an expert data scientist and analyst.
Your job is to perform a complete Exploratory Data Analysis (EDA) on a dataset
and produce a well-structured, human-readable Markdown report.

You have access to a set of tools. Use ALL of them in sequence to gather every
piece of information you need before writing the report.

Tool calling order:
1. get_data_shape          — understand the structure
2. get_missing_values      — check data quality
3. get_descriptive_stats   — summarize numeric features
4. get_outlier_detection   — flag anomalies
5. get_correlation_analysis — find feature relationships
6. get_categorical_analysis — understand text/category columns
7. get_ml_recommendation   — suggest ML problem type and pipeline

After calling all tools, write a Markdown report with these exact sections:

# DataNarrator EDA Report

## 1. Dataset Overview
## 2. Data Quality Assessment
## 3. Statistical Summary
## 4. Outlier Analysis
## 5. Feature Correlations
## 6. Categorical Features
## 7. ML Recommendations & Starter Pipeline
## 8. Key Takeaways

Be specific, insightful, and actionable. Write like a senior data scientist
presenting findings to a team. Use bullet points, bold key terms, and tables
where helpful. Do NOT just repeat raw numbers — interpret them.
"""

USER_PROMPT = (
    "Please perform a complete EDA on the loaded dataset. "
    "Use all available tools, then write the full Markdown report."
)


def build_agent(api_key: str):
    """
    Builds and returns the LangGraph react agent.

    Args:
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.

    Returns:
        A compiled LangGraph agent.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=key,
    )

    agent = create_agent(
        model=llm,
        tools=EDA_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_eda(df, api_key: str) -> str:
    """
    Main entry point. Loads the dataframe and runs the full EDA pipeline.

    Args:
        df: A pandas DataFrame.
        api_key: Optional Google API key string.

    Returns:
        Markdown report string.
    """
    from agent.tools import load_dataframe
    load_dataframe(df)

    agent = build_agent(api_key=api_key)

    result = agent.invoke({
        "messages": [HumanMessage(content=USER_PROMPT)]
    })

    # The final AI message contains the written report
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 100:
            return msg.content

    return "Agent completed but returned no report. Check your API key and try again."