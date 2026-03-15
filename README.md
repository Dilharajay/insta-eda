# ⚡ InstaEDA

> Drop in a CSV. Get a full EDA report + AI-recommended visuals — powered by Gemini.

InstaEDA is a LangChain-powered agent that autonomously runs Exploratory Data Analysis on any CSV dataset. It produces a clean, human-readable Markdown report and an interactive dashboard with AI-selected visualizations tailored to your specific data.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.12%2B-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Plotly](https://img.shields.io/badge/Visuals-Plotly-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **7 autonomous EDA tools** — shape, nulls, stats, outliers, correlations, categoricals, ML hints.
- **AI-Driven Visuals** — Gemini identifies the most important insights and recommends specific Plotly charts.
- **Analysis History** — Automatically saves every report to your account for later retrieval.
- **Interactive Dashboard** — Load past analyses instantly from the sidebar.
- **Dynamic Feature Selection** — Automatically excludes identifiers (IDs, UUIDs) and noise from plots to focus on real data.
- **User Authentication** — Secure login/signup system with password hashing (`bcrypt`).
- **Account Management** — Change your username or password directly from the dashboard.
- **Persistent Settings** — Securely save your Google API key to your account so you don't have to re-enter it.
- **Model Selection** — Choose between different Gemini models (e.g., `gemini-3.1-pro`, `gemini-1.5-flash`) in the settings.
- **Interactive Dashboard** — Explore your data visually with zoomable, filterable Plotly charts.
- **ML Recommendations** — Problem type detection + starter sklearn pipeline.

---

## Project Structure

```
insta-eda/
├── app.py                      # Streamlit UI (entry point)
├── agent/
│   ├── config.py               # AI instructions & prompt templates
│   ├── eda_agent.py            # LangChain executor & result parser
│   └── tools.py                # 7 custom pandas EDA tools
├── utils/
│   ├── auth.py                 # SQLite + bcrypt authentication logic
│   └── report.py               # Report saving + metadata injection
├── sample_data/                # Sample datasets for testing
├── instaeda.db                 # SQLite database (auto-generated)
└── pyproject.toml              # Project dependencies (uv/pip)
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/insta-eda.git
cd insta-eda
```

### 2. Install dependencies

```bash
pip install .
# or using uv
uv sync
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Setup

1. **Sign Up / Login** in the sidebar.
2. Go to the **Settings** tab.
3. Enter your **Google API Key** and click **Save API Key**. Get a free key at [aistudio.google.com](https://aistudio.google.com).
4. Choose your preferred **Gemini Model**.

---

## How It Works

1. **User Uploads CSV:** The file is parsed and registered in the agent's global state.
2. **Autonomous Tool Execution:** The system runs all 7 EDA tools to gather raw data.
3. **LLM Analysis:** Gemini reviews the tool outputs, writes the narrative report, and **selects the best features** for visualization while filtering out noise (like IDs).
4. **Interactive Rendering:** The UI dynamically builds a Plotly dashboard based on the AI's specific recommendations for *that* dataset.
5. **Report Delivery:** You can download the full Markdown report for your documentation.

---

## Tech Stack

| Component | Library |
|---|---|
| Agent framework | LangChain 1.2+ |
| LLM | Google Gemini (selectable) |
| Data analysis | pandas, numpy |
| Visualizations | Plotly Express |
| UI | Streamlit |
| Database | SQLite3 |
| Security | bcrypt (Password Hashing) |

---

## License

MIT
