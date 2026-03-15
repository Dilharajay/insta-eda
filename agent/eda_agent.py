"""
InstaEDA — EDA Pipeline
Runs all 7 tools directly in Python, then sends results to Gemini
in a single LLM call to write the report. No agent loop — 1 API call per run.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.config import SYSTEM_PROMPT

from agent.tools import (
    load_dataframe,
    get_data_shape,
    get_missing_values,
    get_descriptive_stats,
    get_outlier_detection,
    get_correlation_analysis,
    get_categorical_analysis,
    get_ml_recommendation,
)

def _collect_tool_results() -> str:
    """Runs all 7 EDA tools directly and returns their combined output as a string."""
    tools = [
        ("Data Shape & Dtypes",      get_data_shape),
        ("Missing Values",           get_missing_values),
        ("Descriptive Statistics",   get_descriptive_stats),
        ("Outlier Detection",        get_outlier_detection),
        ("Correlation Analysis",     get_correlation_analysis),
        ("Categorical Analysis",     get_categorical_analysis),
        ("ML Recommendation",        get_ml_recommendation),
    ]

    sections = []
    for label, tool in tools:
        try:
            result = tool.invoke("")
            sections.append(f"### {label}\n{result}")
        except Exception as e:
            sections.append(f"### {label}\nError: {e}")

    return "\n\n".join(sections)


def run_eda(df, api_key: str = None, model_name: str = "gemini-1.5-flash") -> str:
    """
    Main entry point. Runs all EDA tools then calls Gemini once to write the report.

    Args:
        df: A pandas DataFrame.
        api_key: Optional Google API key string.
        model_name: The name of the Gemini model to use.

    Returns:
        Markdown report string.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")

    load_dataframe(df)

    tool_results = _collect_tool_results()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=key,
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here are the EDA tool results:\n\n{tool_results}\n\nNow write the full report."),
    ])

    return response.content