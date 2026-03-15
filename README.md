# InstaEDA

> Drop in a CSV. Get a full EDA report written by an AI agent.

InstaEDA is a LangChain-powered agent that autonomously runs Exploratory Data Analysis on any CSV dataset and produces a clean, human-readable Markdown report — complete with statistical insights, outlier flags, feature correlations, and ML pipeline recommendations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

Upload any CSV → agent calls 7 EDA tools in sequence → full report generated in ~30–60 seconds.

---

## Features

- **7 autonomous EDA tools** — shape, nulls, stats, outliers, correlations, categoricals, ML hints
- **LLM-written narrative** — not just numbers, but interpreted insights
- **Streamlit UI** — drag-and-drop CSV, download report as `.md`
- **ML recommendations** — problem type detection + starter sklearn pipeline
- **Works with any CSV** — no schema assumptions

---

## Project Structure

```
datnarrator/
├── app.py                      # Streamlit UI (entry point)
├── agent/
│   ├── eda_agent.py            # LangChain agent + executor
│   └── tools.py                # 7 custom pandas EDA tools
├── utils/
│   └── report.py               # Report saving + metadata injection
├── sample_data/
│   └── employee_sample.csv     # Sample dataset for testing
└── requirements.txt
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/datnarrator.git
cd datnarrator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your google API key

```bash
export GOOGLE_API_KEY="AIza..."
```

Or enter it directly in the Streamlit sidebar. Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 4. Run the app

```bash
streamlit run app.py
```

---

## How It Works

The agent follows a ReAct-style tool calling loop:

```
User uploads CSV
       │
       ▼
  load_dataframe()          ← registers df in global state
       │
       ▼
  LangChain Agent starts
       │
       ├── get_data_shape()
       ├── get_missing_values()
       ├── get_descriptive_stats()
       ├── get_outlier_detection()
       ├── get_correlation_analysis()
       ├── get_categorical_analysis()
       └── get_ml_recommendation()
       │
       ▼
  LLM synthesizes all tool outputs
       │
       ▼
  Markdown report returned to UI
```

---

## Use It Without the UI

```python
import pandas as pd
from agent.eda_agent import run_eda

df = pd.read_csv("your_data.csv")
report = run_eda(df, api_key="AIza...")
print(report)
```

---

## Extending DataNarrator

Adding a new tool is straightforward:

```python
# agent/tools.py
from langchain_core.tools import tool

@tool
def get_skewness_analysis(input: str = "") -> str:
    """Returns skewness scores for all numeric columns."""
    df = _require_df()
    skew = df.select_dtypes(include='number').skew().round(4).to_dict()
    return json.dumps(skew, indent=2)
```

Then add it to `EDA_TOOLS` in `eda_agent.py`. That is all.

---

## Tech Stack

| Component | Library |
|---|---|
| Agent framework | LangChain 0.3 |
| LLM | Google Gemini 1.5 Flash |
| Data analysis | pandas, numpy |
| UI | Streamlit |
| Tool protocol | LangChain `@tool` decorator |

---

## License

MIT