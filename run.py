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

# --- App Constants ---
REQUIRED_COLS = [
    'campaign', 'date', 'historical_reach', 'ad_spend', 'engagement_rate',
    'competitor_ad_spend', 'seasonality_factor', 'repeat_customer_rate', 'campaign_risk'
]

# --- Helper Functions (No changes to these) ---
@st.cache_data
def load_raw_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')): return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return None

def clean_and_validate(df):
    if df is None: return None
    df_clean = df.copy()
    df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
    missing_cols = [col for col in REQUIRED_COLS if col not in df_clean.columns]
    if missing_cols:
        st.error(f"Data is missing required columns: {', '.join(missing_cols)}.")
        return None
    for col in REQUIRED_COLS:
        if col not in ['campaign', 'date']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.dropna(inplace=True)
    return df_clean

def process_dataframe(df):
    df_processed = df.copy()
    df_processed['efficiency_score'] = (df_processed['historical_reach'] / (df_processed['ad_spend'] * df_processed['engagement_rate'] + 1e-6)).round(4)
    df_processed['potential_growth'] = (df_processed['repeat_customer_rate'] * df_processed['seasonality_factor']).round(4)
    return df_processed

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
        if "API key not valid" in str(e): return "The provided Gemini API key is invalid or has expired."
        return f"An error occurred with the Gemini API: {e}."

# --- THIS IS THE CORRECTED OPTIMIZATION FUNCTION ---
def run_optimization(df, total_budget, total_customers, min_alloc_perc=1, max_alloc_perc=30):
    """
    Performs linear programming with diversification constraints.
    
    Args:
        df (pd.DataFrame): The input dataframe with campaign data.
        total_budget (float): The total budget to allocate.
        total_customers (int): The minimum number of customers to reach.
        min_alloc_perc (int): The minimum percentage of the budget each campaign must receive.
        max_alloc_perc (int): The maximum percentage of the budget any single campaign can receive.
    """
    df_opt = df.copy()
    num_campaigns = len(df_opt)
    
    # Define Optimization Potential (Objective Function)
    df_opt['optimization_potential'] = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)) * df_opt['engagement_rate'] * df_opt['seasonality_factor']
    c = -df_opt['optimization_potential'].values
    
    # Define Constraints
    reach_per_dollar = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)).values
    
    # Inequality constraints (A_ub * x <= b_ub)
    # 1. Sum of budgets <= total_budget
    # 2. -Sum of reach <= -total_customers (to meet minimum reach)
    A_ub = [
        np.ones(num_campaigns),
        -reach_per_dollar
    ]
    b_ub = [
        total_budget,
        -total_customers
    ]
    
    # Define Bounds for each campaign's budget (x_i)
    # This is where we enforce diversification
    min_budget_per_campaign = (min_alloc_perc / 100) * total_budget / num_campaigns
    max_budget_per_campaign = (max_alloc_perc / 100) * total_budget
    
    bounds = [(min_budget_per_campaign, max_budget_per_campaign) for _ in range(num_campaigns)]
    
    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        allocated_budgets = result.x
        # If the optimizer allocates less than the total budget (due to constraints),
        # redistribute the remainder proportionally to the allocation.
        unallocated_budget = total_budget - allocated_budgets.sum()
        if unallocated_budget > 0 and allocated_budgets.sum() > 0:
            proportions = allocated_budgets / allocated_budgets.sum()
            allocated_budgets += unallocated_budget * proportions

        df_opt['allocated_budget'] = allocated_budgets.round(2)
        df_opt['estimated_reach'] = (df_opt['allocated_budget'] * reach_per_dollar).astype(int)
        return df_opt.sort_values(by="optimization_potential", ascending=False)
    else:
        st.warning(f"Optimization could not find a solution. This often means the constraints are too tight. Try increasing the budget, decreasing the target customers, or adjusting the allocation percentages. (Message: {result.message})")
        return None
# --- END OF CORRECTED FUNCTION ---


# --- Sidebar ---
with st.sidebar:
    st.title("🚀 Intelligent Campaign AI")
    data_source = st.radio("Select Data Source", ("Use Sample Data", "Upload Your Own Data"), key="data_source_radio")
    analysis_mode = st.selectbox("Select Analysis Mode", ("Campaign Performance Dashboard", "Optimization Engine", "AI Insights by Generative AI Agent"))

# --- Main App Logic ---
if 'df_processed' not in st.session_state: st.session_state.df_processed = None

if data_source == "Use Sample Data":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'sample_campaign_data.csv')
        raw_df = pd.read_csv(file_path)
        st.session_state.df_processed = process_dataframe(clean_and_validate(raw_df))
    except Exception as e:
        st.error(f"Could not load or process sample_campaign_data.csv. Error: {e}")
        st.session_state.df_processed = None
else:
    uploaded_file = st.file_uploader("Upload your campaign data (CSV or Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        raw_df = load_raw_data(uploaded_file)
        validated_df = clean_and_validate(raw_df)
        if validated_df is not None:
             st.session_state.df_processed = process_dataframe(validated_df)
             st.success("File uploaded and processed successfully!")
    else:
        st.session_state.df_processed = None

# --- Display Content ---
if st.session_state.df_processed is None:
    st.info("Please select a data source and follow the steps to begin analysis.")
    st.stop()

df = st.session_state.df_processed

if analysis_mode == "Campaign Performance Dashboard":
    # (This section is unchanged)
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

# --- THIS IS THE CORRECTED OPTIMIZATION ENGINE UI ---
elif analysis_mode == "Optimization Engine":
    st.header("⚙️ AI-Powered Optimization Engine")
    
    with st.expander("Settings & Constraints", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.slider("Total Marketing Budget ($)", 50000, 1000000, 200000, 10000)
        with col2:
            total_customers = st.slider("Minimum Target Customers", 10000, 1000000, 100000, 5000)
        
        st.markdown("---")
        st.subheader("Diversification Rules")
        col3, col4 = st.columns(2)
        with col3:
            min_alloc_perc = st.slider(
                "Min Allocation per Campaign (%)", 0, 10, 1, 1,
                help="The minimum percentage of the *average* campaign budget that each campaign must receive. Prevents any campaign from getting $0."
            )
        with col4:
            max_alloc_perc = st.slider(
                "Max Allocation Cap per Campaign (%)", 10, 100, 30, 5,
                help="The maximum percentage of the *total* budget that any single campaign is allowed to receive. Prevents one campaign from taking all the funds."
            )

    if st.button("🚀 Run AI-Powered Optimization", use_container_width=True):
        with st.spinner("Running realistic optimization..."):
            optimized_df = run_optimization(df, total_budget, total_customers, min_alloc_perc, max_alloc_perc)
        if optimized_df is not None:
            st.subheader("📈 AI Optimization Strategy")
            fig_bar = px.bar(
                optimized_df.sort_values('allocated_budget', ascending=False), 
                x='campaign', y='allocated_budget', color='optimization_potential',
                color_continuous_scale='Greens', title=f"Diversified Budget Allocation for ${total_budget:,}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("📋 Optimization Breakdown")
            st.dataframe(optimized_df[['campaign', 'allocated_budget', 'estimated_reach', 'optimization_potential']], use_container_width=True)
            
            st.subheader("💡 AI Strategic Recommendations")
            top_campaign = optimized_df.iloc[0]
            total_reach_original = df['historical_reach'].sum()
            total_reach_optimized = optimized_df['estimated_reach'].sum()
            reach_increase = ((total_reach_optimized - total_reach_original) / total_reach_original * 100) if total_reach_original > 0 else 100
            st.markdown(f"""
            - **🎯 Top Priority:** **{top_campaign['campaign']}** still receives the most funding due to its high potential, but its budget is now capped to allow for diversification.
            - **💸 Diversified Strategy:** The budget is now spread across multiple campaigns, with each receiving at least a minimum allocation. This reduces risk and ensures brand presence across different channels.
            - **📈 Potential Growth:** The new, more balanced strategy could increase overall customer reach by an estimated **{reach_increase:.2f}%**.
            - **🤔 Review:** Campaigns receiving only the minimum allocation should be reviewed. Consider improving their creative, targeting, or replacing them in the next planning cycle.
            """)
# --- END OF CORRECTED UI ---

elif analysis_mode == "AI Insights by Generative AI Agent":
    # (This section is unchanged)
    st.header("🤖 AI Insights by Generative AI Agent")
    with st.form(key="ai_form"):
        user_question = st.text_area("Specific Campaign Strategy Question:", "Which campaign has the best performance and why?", height=100)
        submit_button = st.form_submit_button(label="Generate Strategic Insights")
    if submit_button:
        if "gemini_api_key" not in st.secrets:
            st.error("Gemini API key not found. Please set it in your Streamlit secrets as 'gemini_api_key'.")
        elif not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating insights..."):
                api_key = st.secrets["gemini_api_key"]
                ai_response = ask_gemini(user_question, df, api_key)
            st.markdown("---"); st.subheader("Al Strategic Insights"); st.markdown(ai_response)
