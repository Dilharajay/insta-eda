import streamlit as st
import json
import pandas as pd
import plotly.express as px

def render_visuals():
    st.subheader("📊 Data Exploration Dashboard (AI-Selected)")
    raw = st.session_state["raw_results"]
    viz_configs = st.session_state.get("viz_configs", [])
    
    if not viz_configs:
        st.info("Gemini didn't recommend specific visuals for this dataset. Here is the default analysis.")
        viz_configs = [
            {"tool": "missing_values", "chart_type": "bar", "title": "Missing Values Overview"},
            {"tool": "outlier_detection", "chart_type": "bar", "title": "Outlier Counts"},
            {"tool": "correlation_analysis", "chart_type": "heatmap", "title": "Top Correlations"},
            {"tool": "categorical_analysis", "chart_type": "pie", "title": "Categorical Distributions"}
        ]

    for config in viz_configs:
        tool_key = config.get("tool")
        chart_type = config.get("chart_type")
        title = config.get("title", "Insight Chart")
        description = config.get("description", "")
        params = config.get("params", {})
        selected_cols = params.get("columns", [])
        reasoning = params.get("reasoning", "")
        
        try:
            tool_data = json.loads(raw[tool_key]) if tool_key in raw and isinstance(raw[tool_key], str) else raw.get(tool_key)
            if not tool_data or (isinstance(tool_data, str) and tool_data.startswith("No")):
                continue

            # Filter tool_data if columns are specified
            if selected_cols and isinstance(tool_data, dict):
                tool_data = {k: v for k, v in tool_data.items() if k in selected_cols}
                if not tool_data: 
                    st.warning(f"AI requested columns {selected_cols} but they weren't found in tool results.")
                    continue

            st.write(f"### {title}")
            if description:
                st.caption(description)
            if reasoning:
                st.info(f"**AI Reasoning:** {reasoning}")

            if tool_key == "missing_values":
                mv_df = pd.DataFrame.from_dict(tool_data, orient='index').reset_index()
                mv_df.columns = ['Column', 'Missing Count', 'Percent', 'Concern']
                fig = px.bar(mv_df, x='Column', y='Percent', color='Concern', 
                           title=title, color_discrete_map={True: '#ef553b', False: '#636efa'})
                st.plotly_chart(fig, use_container_width=True)

            elif tool_key == "outlier_detection":
                out_df = pd.DataFrame.from_dict(tool_data, orient='index').reset_index()
                out_df.columns = ['Column', 'Outlier Count', 'Percent', 'Lower', 'Upper']
                fig = px.bar(out_df, x='Column', y='Outlier Count', 
                            title=title, color_discrete_sequence=['#ab63fa'])
                st.plotly_chart(fig, use_container_width=True)

            elif tool_key == "correlation_analysis":
                corr_df = pd.DataFrame(tool_data)
                if selected_cols:
                    corr_df = corr_df[corr_df['feature_a'].isin(selected_cols) | corr_df['feature_b'].isin(selected_cols)]
                
                if not corr_df.empty:
                    fig = px.bar(corr_df, x='correlation', y='feature_a', color='feature_b',
                                 orientation='h', title=title)
                    st.plotly_chart(fig, use_container_width=True)

            elif tool_key == "categorical_analysis":
                cols = st.columns(min(len(tool_data), 3))
                for i, (col_name, info) in enumerate(tool_data.items()):
                    with cols[i % 3]:
                        top_v = pd.DataFrame.from_dict(info['top_5_values'], orient='index').reset_index()
                        top_v.columns = ['Value', 'Count']
                        fig = px.pie(top_v, names='Value', values='Count', title=f"{col_name}")
                        st.plotly_chart(fig, use_container_width=True)

            elif tool_key == "descriptive_stats":
                stats_df = pd.DataFrame(tool_data).T.reset_index()
                stats_df.columns = ['Feature', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                if selected_cols:
                    stats_df = stats_df[stats_df['Feature'].isin(selected_cols)]
                
                if not stats_df.empty:
                    fig = px.bar(stats_df, x='Feature', y='mean', error_y='std', title=title)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            pass
