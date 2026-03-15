"""
InstaEDA — EDA Pipeline
Runs all 7 tools directly in Python, then sends results to Gemini
in a single LLM call to write the report. No agent loop — 1 API call per run.
"""

import os
import re
import json
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

def _collect_tool_results_raw() -> dict:
    """Runs all 7 EDA tools directly and returns their results as a dictionary."""
    tools = [
        ("data_shape",           get_data_shape),
        ("missing_values",       get_missing_values),
        ("descriptive_stats",    get_descriptive_stats),
        ("outlier_detection",    get_outlier_detection),
        ("correlation_analysis", get_correlation_analysis),
        ("categorical_analysis", get_categorical_analysis),
        ("ml_recommendation",    get_ml_recommendation),
    ]

    results = {}
    for key, tool in tools:
        try:
            result = tool.invoke("")
            results[key] = result
        except Exception as e:
            results[key] = f"Error: {e}"

    return results


def run_eda(df, api_key: str = None, model_name: str = "gemini-1.5-flash") -> dict:
    """
    Main entry point. Runs all EDA tools then calls Gemini once to write the report.

    Args:
        df: A pandas DataFrame.
        api_key: Optional Google API key string.
        model_name: The name of the Gemini model to use.

    Returns:
        Dictionary containing 'report' (str) and 'raw_results' (dict).
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")

    load_dataframe(df)

    raw_results = _collect_tool_results_raw()

    # Format results for Gemini
    formatted_sections = []
    for key_label, result in raw_results.items():
        formatted_sections.append(f"### {key_label.replace('_', ' ').title()}\n{result}")

    tool_results_str = "\n\n".join(formatted_sections)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=key,
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here are the EDA tool results:\n\n{tool_results_str}\n\nNow write the full report."),
    ])

    report_content = response.content
    viz_configs = []

    # Try to extract JSON from the report
    try:
        json_pattern = r'\[\s*\{.*\}\s*\]' # Matches a JSON array
        match = re.search(json_pattern, report_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            viz_configs = json.loads(json_str)
    except Exception as e:
        # Fallback to an empty list if JSON parsing fails
        pass

    return {
        "report": report_content,
        "raw_results": raw_results,
        "viz_configs": viz_configs
    }