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

# --- Helper Functions (No changes here) ---
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

# --- THIS IS THE ADVANCED, TIERED OPTIMIZATION FUNCTION ---
def run_optimization(df, total_budget, total_customers, min_alloc_perc=1, tier_caps=None):
    """
    Performs linear programming with tiered diversification constraints.
    """
    if tier_caps is None:
        tier_caps = {'Top': 30, 'Middle': 15, 'Bottom': 5}
        
    df_opt = df.copy()
    num_campaigns = len(df_opt)
    
    # 1. Define and Rank by Optimization Potential
    df_opt['optimization_potential'] = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)) * df_opt['engagement_rate'] * df_opt['seasonality_factor']
    df_opt = df_opt.sort_values('optimization_potential', ascending=False).reset_index(drop=True)
    c = -df_opt['optimization_potential'].values
    
    # 2. Assign Tiers
    top_tier_count = max(1, int(num_campaigns * 0.2)) # Top 20%
    middle_tier_count = max(1, int(num_campaigns * 0.5)) # Next 50%
    
    df_opt['tier'] = 'Bottom'
    df_opt.loc[:top_tier_count-1, 'tier'] = 'Top'
    df_opt.loc[top_tier_count:top_tier_count+middle_tier_count-1, 'tier'] = 'Middle'
    
    # 3. Define Bounds based on Tiers
    bounds = []
    min_budget_per_campaign = (min_alloc_perc / 100) * total_budget / num_campaigns
    
    for i, row in df_opt.iterrows():
        tier = row['tier']
        max_perc = tier_caps.get(tier, 5) # Default to 5% if tier not found
        max_budget = (max_perc / 100) * total_budget
        bounds.append((min_budget_per_campaign, max_budget))
        
    # 4. Define Constraints
    reach_per_dollar = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)).values
    
    A_ub = [
        np.ones(num_campaigns),  # Sum of budgets <= total_budget
        -reach_per_dollar       # -Sum of reach <= -total_customers
    ]
    b_ub = [
        total_budget,
        -total_customers
    ]
    
    # 5. Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        allocated_budgets = result.x
        unallocated_budget = total_budget - allocated_budgets.sum()
        if unallocated_budget > 0 and allocated_budgets.sum() > 0:
            proportions = allocated_budgets / allocated_budgets.sum()
            allocated_budgets += unallocated_budget * proportions

        df_opt['allocated_budget'] = allocated_budgets.round(2)
        df_opt['estimated_reach'] = (df_opt['allocated_budget'] * reach_per_dollar).astype(int)
        return df_opt
    else:
        st.warning(f"Optimization could not find a solution. Constraints might be too tight. Try increasing the budget or adjusting tier caps. (Message: {result.message})")
        return None
# --- END OF ADVANCED FUNCTION ---


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
    # ... (code is identical to previous version) ...
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


# --- THIS IS THE ADVANCED OPTIMIZATION ENGINE UI ---
elif analysis_mode == "Optimization Engine":
    st.header("⚙️ AI-Powered Optimization Engine")
    
    with st.expander("Settings & Constraints", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.slider("Total Marketing Budget ($)", 50000, 1000000, 200000, 10000)
        with col2:
            total_customers = st.slider("Minimum Target Customers", 10000, 1000000, 100000, 5000)
        
        st.markdown("---")
        st.subheader("Advanced: Tier-Based Allocation Caps (%)")
        col3, col4, col5 = st.columns(3)
        with col3:
            top_cap = st.slider("Top Tier Cap", 10, 50, 30, help="Max % of total budget for the top 20% of campaigns.")
        with col4:
            mid_cap = st.slider("Middle Tier Cap", 5, 30, 15, help="Max % of total budget for the middle 50% of campaigns.")
        with col5:
            bot_cap = st.slider("Bottom Tier Cap", 1, 10, 5, help="Max % of total budget for the bottom 30% of campaigns.")
            
    if st.button("🚀 Run AI-Powered Optimization", use_container_width=True):
        tier_caps = {'Top': top_cap, 'Middle': mid_cap, 'Bottom': bot_cap}
        with st.spinner("Running tiered optimization..."):
            optimized_df = run_optimization(df, total_budget, total_customers, min_alloc_perc=0, tier_caps=tier_caps)
        if optimized_df is not None:
            st.subheader("📈 AI Tiered Allocation Strategy")
            
            # Add tier information for coloring the bar chart
            fig_bar = px.bar(
                optimized_df.sort_values('allocated_budget', ascending=False), 
                x='campaign', y='allocated_budget', color='tier',
                category_orders={"tier": ["Top", "Middle", "Bottom"]},
                color_discrete_map={'Top': 'green', 'Middle': 'goldenrod', 'Bottom': 'firebrick'},
                title=f"Tiered & Diversified Budget Allocation for ${total_budget:,}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("📋 Optimization Breakdown")
            st.dataframe(optimized_df[['campaign', 'tier', 'allocated_budget', 'estimated_reach', 'optimization_potential']], use_container_width=True)
            
            st.subheader("💡 AI Strategic Recommendations")
            st.markdown(f"""
            - **Tiered Investment:** The budget is now allocated according to performance tiers. **Top-tier** campaigns receive the most significant investment, followed by **middle-tier** campaigns which show promise. **Bottom-tier** campaigns receive minimal funding, primarily for testing or maintenance.
            - **Balanced Portfolio:** This approach creates a balanced portfolio, maximizing returns from top performers while still nurturing promising campaigns and gathering data on laggards.
            - **Actionable Insights:** Focus your strategic efforts on moving campaigns from the 'Middle' to the 'Top' tier. Analyze why 'Bottom' tier campaigns are underperforming and decide whether to improve or discontinue them.
            """)
# --- END OF ADVANCED UI ---

elif analysis_mode == "AI Insights by Generative AI Agent":
    # (This section is unchanged)
    st.header("🤖 AI Insights by Generative AI Agent")
    # ... (code is identical to previous version) ...
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
