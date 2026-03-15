"""
utils/report.py — Helpers for saving and formatting EDA reports.
"""

import os
from datetime import datetime
from pathlib import Path


def save_report(markdown: str, output_dir: str = "reports") -> str:
    """
    Saves the Markdown report to a timestamped file.

    Args:
        markdown: The Markdown string from the agent.
        output_dir: Directory to save the report in.

    Returns:
        Path to the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"eda_report_{timestamp}.md")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)

    return filename


def inject_metadata(markdown: str, dataset_name: str, rows: int, cols: int) -> str:
    """
    Adds a metadata header block to the top of the report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""---
**Dataset:** {dataset_name}  
**Shape:** {rows} rows × {cols} columns  
**Generated:** {timestamp}  
**Tool:** DataNarrator v1.0
---

"""
    return header + markdown