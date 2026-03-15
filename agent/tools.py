"""
Each tool wraps a pandas operation and returns a clean string summary
that the LLM agent can reason over.
"""

import json
import pandas as pd
import numpy as np
from langchain_core.tools import tool

# Global state: the agent loads the dataframe once and all tools share it
_df: pd.DataFrame = None


def load_dataframe(df: pd.DataFrame):
    """Call this once before running the agent to register the dataset."""
    global _df
    _df = df


def _require_df():
    if _df is None:
        raise ValueError("No dataframe loaded. Call load_dataframe() first.")
    return _df


# ─────────────────────────────────────────────
# Tool 1 — Shape & Basic Info
# ─────────────────────────────────────────────
@tool
def get_data_shape(input: str = "") -> str:
    """
    Returns the number of rows and columns in the dataset,
    plus the column names and their data types.
    """
    df = _require_df()
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return json.dumps(info, indent=2)


# ─────────────────────────────────────────────
# Tool 2 — Missing Values
# ─────────────────────────────────────────────
@tool
def get_missing_values(input: str = "") -> str:
    """
    Analyzes missing values per column.
    Returns count and percentage of nulls for each column.
    Highlights columns with more than 5% missing data.
    """
    df = _require_df()
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    result = {}
    for col in df.columns:
        if missing[col] > 0:
            result[col] = {
                "missing_count": int(missing[col]),
                "missing_percent": float(pct[col]),
                "concern": pct[col] > 5,
            }

    if not result:
        return "No missing values found. Dataset is complete."

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# Tool 3 — Descriptive Statistics
# ─────────────────────────────────────────────
@tool
def get_descriptive_stats(input: str = "") -> str:
    """
    Returns descriptive statistics (mean, std, min, max, quartiles)
    for all numeric columns in the dataset.
    """
    df = _require_df()
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return "No numeric columns found in the dataset."

    stats = numeric_df.describe().round(4).to_dict()
    return json.dumps(stats, indent=2)


# ─────────────────────────────────────────────
# Tool 4 — Outlier Detection (IQR method)
# ─────────────────────────────────────────────
@tool
def get_outlier_detection(input: str = "") -> str:
    """
    Detects outliers in numeric columns using the IQR (Interquartile Range) method.
    Returns which columns have outliers and how many rows are affected.
    """
    df = _require_df()
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return "No numeric columns found for outlier detection."

    result = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())

        if outlier_count > 0:
            result[col] = {
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / len(df) * 100, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }

    if not result:
        return "No significant outliers detected across numeric columns."

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# Tool 5 — Correlation Analysis
# ─────────────────────────────────────────────
@tool
def get_correlation_analysis(input: str = "") -> str:
    """
    Computes Pearson correlations between all numeric columns.
    Returns the top 10 strongest (positive or negative) correlations,
    which helps identify redundant features or predictive relationships.
    """
    df = _require_df()
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return "Need at least 2 numeric columns for correlation analysis."

    corr_matrix = numeric_df.corr()
    # Extract unique pairs (upper triangle only)
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": round(float(val), 4)})

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    top_pairs = pairs[:10]

    return json.dumps(top_pairs, indent=2)


# ─────────────────────────────────────────────
# Tool 6 — Categorical Column Analysis
# ─────────────────────────────────────────────
@tool
def get_categorical_analysis(input: str = "") -> str:
    """
    Analyzes categorical (object/string) columns.
    Returns unique value counts and the top 5 most frequent values per column.
    Flags high-cardinality columns that may need encoding attention.
    """
    df = _require_df()
    cat_df = df.select_dtypes(include=["object", "category"])

    if cat_df.empty:
        return "No categorical columns found in the dataset."

    result = {}
    for col in cat_df.columns:
        vc = df[col].value_counts()
        result[col] = {
            "unique_values": int(df[col].nunique()),
            "high_cardinality": df[col].nunique() > 20,
            "top_5_values": vc.head(5).to_dict(),
        }

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# Tool 7 — ML Problem Type Recommendation
# ─────────────────────────────────────────────
@tool
def get_ml_recommendation(input: str = "") -> str:
    """
    Analyzes the dataset structure and suggests what kind of ML problem
    it is suited for (classification, regression, clustering, time series).
    Also recommends preprocessing steps and a starter sklearn pipeline.
    """
    df = _require_df()

    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    cat_cols = df.select_dtypes(include=["object", "category"]).shape[1]
    total_rows = len(df)
    has_datetime = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)

    hints = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "total_rows": total_rows,
        "has_datetime_column": has_datetime,
        "dataset_size": "small" if total_rows < 1000 else "medium" if total_rows < 50000 else "large",
        "suggested_problems": [],
        "preprocessing_hints": [],
        "starter_pipeline_suggestion": "",
    }

    if has_datetime:
        hints["suggested_problems"].append("Time Series Forecasting")
        hints["starter_pipeline_suggestion"] = (
            "Consider ARIMA, SARIMA, or Prophet for forecasting. "
            "Parse datetime index and resample to your target frequency."
        )

    if cat_cols > 0:
        hints["preprocessing_hints"].append("Encode categoricals with OrdinalEncoder or OneHotEncoder")
        hints["suggested_problems"].append("Classification (if target is categorical)")

    if num_cols > 3:
        hints["suggested_problems"].append("Regression (if target is continuous)")
        hints["suggested_problems"].append("Clustering (if no clear target column)")
        hints["preprocessing_hints"].append("Scale numeric features with StandardScaler or MinMaxScaler")

    if total_rows < 500:
        hints["preprocessing_hints"].append("Small dataset: prefer simpler models (LogisticRegression, RandomForest) over deep learning")

    if not hints["starter_pipeline_suggestion"]:
        hints["starter_pipeline_suggestion"] = (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])"
        )

    return json.dumps(hints, indent=2)