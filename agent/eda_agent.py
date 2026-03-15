"""
DataNarrator — LangChain EDA Agent
Orchestrates all tools and generates the final narrative report.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
1. get_data_shape         — understand the structure
2. get_missing_values     — check data quality
3. get_descriptive_stats  — summarize numeric features
4. get_outlier_detection  — flag anomalies
5. get_correlation_analysis — find feature relationships
6. get_categorical_analysis — understand text/category columns
7. get_ml_recommendation  — suggest ML problem type and pipeline

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


def build_agent(api_key: str = None) -> AgentExecutor:
    """
    Builds and returns the LangChain EDA agent executor.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        AgentExecutor ready to run.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # fast and cheap — upgrade to gpt-4o for better reports
        temperature=0.3,
        openai_api_key=key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=EDA_TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=EDA_TOOLS,
        verbose=True,          # set to False to silence tool logs in production
        max_iterations=12,     # enough to call all 7 tools plus reasoning steps
        handle_parsing_errors=True,
    )

    return executor


def run_eda(df, api_key: str = None) -> str:
    """
    Main entry point. Loads the dataframe and runs the full EDA pipeline.

    Args:
        df: A pandas DataFrame.
        api_key: Optional OpenAI API key string.

    Returns:
        Markdown report string.
    """
    from agent.tools import load_dataframe
    load_dataframe(df)

    executor = build_agent(api_key=api_key)

    result = executor.invoke({
        "input": (
            "Please perform a complete EDA on the loaded dataset. "
            "Use all available tools, then write the full Markdown report."
        )
    })

    return result.get("output", "Agent returned no output.")