import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
import google.generativeai as genai
import io
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Campaign Optimization AI",
    page_icon="🚀",
    layout="wide"
)

# --- REQUIRED COLUMNS FOR THE APP ---
REQUIRED_COLS_INTERNAL = [
    'campaign', 'historical_reach', 'ad_spend', 'engagement_rate',
    'competitor_ad_spend', 'seasonality_factor', 'repeat_customer_rate', 'campaign_risk'
]
REQUIRED_COLS_FRIENDLY = {
    'campaign': 'Campaign Name',
    'historical_reach': 'Historical Reach',
    'ad_spend': 'Ad Spend',
    'engagement_rate': 'Engagement Rate',
    'competitor_ad_spend': 'Competitor Ad Spend',
    'seasonality_factor': 'Seasonality Factor',
    'repeat_customer_rate': 'Repeat Customer Rate',
    'campaign_risk': 'Campaign Risk'
}


# --- Helper Functions ---

@st.cache_data
def load_raw_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    return None

def process_dataframe(df):
    df_processed = df.copy()
    for col in REQUIRED_COLS_INTERNAL:
        if col != 'campaign':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(inplace=True)

    df_processed['efficiency_score'] = (df_processed['historical_reach'] / 
                                        (df_processed['ad_spend'] * df_processed['engagement_rate'] + 1e-6)).round(4)
    df_processed['potential_growth'] = (df_processed['repeat_customer_rate'] * df_processed['seasonality_factor']).round(4)
    return df_processed

def run_optimization(df, total_budget, total_customers):
    df_opt = df.copy()
    df_opt['optimization_potential'] = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)) * df_opt['engagement_rate'] * df_opt['seasonality_factor']
    c = -df_opt['optimization_potential'].values
    reach_per_dollar = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)).values
    A_ub = [np.ones(len(df_opt)), -reach_per_dollar]
    b_ub = [total_budget, -total_customers]
    bounds = [(0, None) for _ in range(len(df_opt))]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        allocated_budgets = result.x
        df_opt['allocated_budget'] = allocated_budgets.round(2)
        df_opt['estimated_reach'] = (allocated_budgets * reach_per_dollar).astype(int)
        return df_opt.sort_values(by="optimization_potential", ascending=False)
    else:
        st.warning("Optimization could not find a solution. This may be due to overly restrictive constraints.")
        return None

def ask_gemini(question, df, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        df_string = buffer.getvalue()
        prompt = f"""You are an expert marketing analyst AI. Your task is to provide data-driven strategic advice.
        CONTEXT: Here is the campaign performance data you need to analyze:
        --- START OF DATA ---
        {df_string}
        --- END OF DATA ---
        USER'S QUESTION: "{question}"
        Please provide a clear, concise, and actionable answer based on the data."""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}."

# --- Sidebar ---
with st.sidebar:
    st.title("🚀 Intelligent Campaign AI")
    data_source = st.radio(
        "Select Data Source",
        ("Use Sample Data", "Upload Your Own Data"),
        key="data_source_radio"
    )
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ("Campaign Performance Dashboard", "Optimization Engine", "AI Insights by Generative AI Agent")
    )

# --- Main App Logic ---

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# --- THIS IS THE CORRECTED LOGIC BLOCK ---
if data_source == "Use Sample Data":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'sample_campaign_data.csv')
        sample_df = pd.read_csv(file_path)
        
        # Standardize the column names of the sample data
        sample_df.columns = [col.strip().lower().replace(' ', '_') for col in sample_df.columns]
        
        # Check if standardized columns match the required internal names
        if all(col in sample_df.columns for col in REQUIRED_COLS_INTERNAL):
            st.session_state.df_processed = process_dataframe(sample_df)
        else:
            st.error("The sample_campaign_data.csv file has incorrect column headers.")
            st.session_state.df_processed = None
            
    except Exception as e:
        st.error(f"Could not load or process sample_campaign_data.csv. Error: {e}")
        st.session_state.df_processed = None
# --- END OF CORRECTED LOGIC BLOCK ---

else: # "Upload Your Own Data"
    uploaded_file = st.file_uploader(
        "Upload your campaign data (CSV or Excel)",
        type=["csv", "xls", "xlsx"]
    )
    if uploaded_file:
        raw_df = load_raw_data(uploaded_file)
        
        if raw_df is not None:
            st.info("File Uploaded. Please map your columns below.")
            with st.expander("Step 2: Map Your Columns", expanded=True):
                uploaded_cols = raw_df.columns.tolist()
                col_mapping = {}
                
                form = st.form(key="column_mapping_form")
                cols = form.columns(2)
                for i, (internal_name, friendly_name) in enumerate(REQUIRED_COLS_FRIENDLY.items()):
                    with cols[i % 2]:
                        col_mapping[internal_name] = form.selectbox(
                            f"Select column for '{friendly_name}'",
                            options=uploaded_cols,
                            key=f"map_{internal_name}"
                        )
                
                submitted = form.form_submit_button("Confirm Mapping and Analyze Data")

                if submitted:
                    if len(set(col_mapping.values())) != len(REQUIRED_COLS_FRIENDLY):
                        st.error("Error: The same uploaded column cannot be used for multiple required fields.")
                    else:
                        rename_dict = {v: k for k, v in col_mapping.items()}
                        df_renamed = raw_df.rename(columns=rename_dict)
                        df_final_raw = df_renamed[REQUIRED_COLS_INTERNAL]
                        st.session_state.df_processed = process_dataframe(df_final_raw)
                        st.success("Columns mapped successfully! The app is now ready.")
                        st.rerun()
    else:
        # Clear processed data if no file is uploaded
        st.session_state.df_processed = None


# --- Display Content IF Data is Ready ---
if st.session_state.df_processed is None:
    st.info("Please select a data source and follow the steps to begin analysis.")
    st.stop()

df = st.session_state.df_processed

if analysis_mode == "Campaign Performance Dashboard":
    st.header("📊 Comprehensive Campaign Performance Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Multi-Dimensional Analysis", "Correlation Insights", "Performance Radar", "Detailed Campaign Metrics"])
    with tab1:
        st.subheader("Campaign Performance Landscape")
        fig = px.scatter(df, x='historical_reach', y='engagement_rate', size='ad_spend', color='campaign_risk', hover_name='campaign',
                         hover_data={'ad_spend': ':,', 'campaign_risk': ':.2f'}, color_continuous_scale=px.colors.sequential.Plasma_r,
                         title="Campaign Performance: Reach vs. Engagement")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Campaign Metrics Correlation")
        corr = df.select_dtypes(include=np.number).corr()
        fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    with tab3:
        st.subheader("Campaign Performance Radar")
        campaigns_to_compare = st.multiselect("Select campaigns to compare:", options=df['campaign'].tolist(), default=df['campaign'].tolist()[:5])
        if campaigns_to_compare:
            df_radar = df[df['campaign'].isin(campaigns_to_compare)]
            metrics_to_plot = ['ad_spend', 'historical_reach', 'engagement_rate', 'repeat_customer_rate']
            df_normalized = df_radar.copy()
            for col in metrics_to_plot:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val - min_val > 0:
                    df_normalized[col] = (df_radar[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[col] = 0.5
            fig_radar = go.Figure()
            for _, row in df_normalized.iterrows():
                fig_radar.add_trace(go.Scatterpolar(r=row[metrics_to_plot].values, theta=metrics_to_plot, fill='toself', name=row['campaign']))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Normalized Performance Metrics")
            st.plotly_chart(fig_radar, use_container_width=True)
    with tab4:
        st.subheader("Detailed Campaign Metrics & Growth Potential")
        st.dataframe(df.style.highlight_max(subset=['efficiency_score', 'potential_growth'], color='lightgreen', axis=0), use_container_width=True)
        st.download_button("Export Data as CSV", df.to_csv(index=False).encode('utf-8'), "campaign_metrics.csv", "text/csv")

elif analysis_mode == "Optimization Engine":
    st.header("⚙️ AI-Powered Optimization Engine")
    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.slider("Total Marketing Budget ($)", 50000, 1000000, 200000, 10000)
    with col2:
        total_customers = st.slider("Minimum Target Customers", 10000, 1000000, 100000, 5000)
    if st.button("🚀 Run AI-Powered Optimization", use_container_width=True):
        with st.spinner("Optimizing..."):
            optimized_df = run_optimization(df, total_budget, total_customers)
        if optimized_df is not None:
            st.subheader("📈 AI Optimization Strategy")
            fig_bar = px.bar(optimized_df, x='campaign', y='allocated_budget', color='optimization_potential',
                             color_continuous_scale='Greens', title=f"Optimized Budget Allocation for ${total_budget:,}")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.subheader("📋 Optimization Breakdown")
            st.dataframe(optimized_df[['campaign', 'allocated_budget', 'estimated_reach', 'optimization_potential']], use_container_width=True)
            st.subheader("💡 AI Strategic Recommendations")
            top_campaign = optimized_df.iloc[0]
            total_reach_original = df['historical_reach'].sum()
            total_reach_optimized = optimized_df['estimated_reach'].sum()
            reach_increase = ((total_reach_optimized - total_reach_original) / total_reach_original * 100) if total_reach_original > 0 else 100
            st.markdown(f"""
            - **🎯 Prioritize:** Focus investment on **{top_campaign['campaign']}**.
            - **💰 Budget Reallocation:** The suggested reallocation could increase overall reach by **{reach_increase:.2f}%**.
            - **👀 Top Performers:** **'{optimized_df.iloc[0]['campaign']}'** and **'{optimized_df.iloc[1]['campaign']}'** show the most promise.
            - **🤔 Review:** Consider adjusting strategies for campaigns with low allocated budgets.
            """)

elif analysis_mode == "AI Insights by Generative AI Agent":
    st.header("🤖 AI Insights by Generative AI Agent")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password", help="Get your key from Google AI Studio.")
    with st.form(key="ai_form"):
        user_question = st.text_area("Specific Campaign Strategy Question:", "Which campaign has the best performance and why?", height=100)
        submit_button = st.form_submit_button(label="Generate Strategic Insights")
    if submit_button:
        if not api_key: st.warning("Please enter your Gemini API key to proceed.")
        elif not user_question: st.warning("Please enter a question.")
        else:
            with st.spinner("Generating insights..."):
                ai_response = ask_gemini(user_question, df, api_key)
            st.markdown("---"); st.subheader("Al Strategic Insights"); st.markdown(ai_response)
