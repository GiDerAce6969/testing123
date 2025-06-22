import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
import google.generativeai as genai
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Campaign Optimization AI",
    page_icon="🚀",
    layout="wide"
)

# --- Helper Functions ---

def clean_columns(df):
    """Standardizes all column names in a DataFrame."""
    cols = df.columns
    new_cols = [col.strip().lower().replace(' ', '_') for col in cols]
    df.columns = new_cols
    return df

def validate_columns(df):
    """Checks if the essential columns exist in the DataFrame."""
    required_cols = [
        'campaign', 'historical_reach', 'ad_spend', 'engagement_rate',
        'competitor_ad_spend', 'seasonality_factor', 'repeat_customer_rate', 'campaign_risk'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}. "
            f"Please ensure your column names match the required format (e.g., 'Historical Reach', 'Ad Spend', etc.)."
        )
        return False
    return True

@st.cache_data 
def get_raw_sample_data():
    """Loads the raw sample CSV data from disk using a robust path."""
    try:
        # Get the absolute path of the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Join this directory path with the filename
        file_path = os.path.join(script_dir, 'sample_campaign_data.csv')
        # Read the CSV from the constructed absolute path
        return pd.read_csv(file_path)
    except FileNotFoundError:
        # This error message is now much more reliable
        st.error(f"FATAL: 'sample_campaign_data.csv' not found in the GitHub repository next to the script. Please ensure it has been uploaded correctly.")
        return None
# --- END OF CORRECTED FUNCTION ---

def process_dataframe(df):
    """Applies all cleaning and metric calculations to a raw dataframe."""
    if df is None:
        return None
    df_processed = clean_columns(df.copy())
    if not validate_columns(df_processed):
        return None
    # Calculate custom metrics
    df_processed['efficiency_score'] = (df_processed['historical_reach'] / 
                                        (df_processed['ad_spend'] * df_processed['engagement_rate'] + 1e-6)).round(4)
    df_processed['potential_growth'] = (df_processed['repeat_customer_rate'] * df_processed['seasonality_factor']).round(4)
    return df_processed
    
def run_optimization(df, total_budget, total_customers):
    """Performs linear programming to optimize budget allocation."""
    df_opt = df.copy()
    
    df_opt['optimization_potential'] = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)) * df_opt['engagement_rate'] * df_opt['seasonality_factor']
    
    c = -df_opt['optimization_potential'].values
    
    reach_per_dollar = (df_opt['historical_reach'] / (df_opt['ad_spend'] + 1e-6)).values
    A_ub = [
        np.ones(len(df_opt)),
        -reach_per_dollar
    ]
    b_ub = [
        total_budget,
        -total_customers
    ]
    
    bounds = [(0, None) for _ in range(len(df_opt))]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        allocated_budgets = result.x
        df_opt['allocated_budget'] = allocated_budgets.round(2)
        df_opt['estimated_reach'] = (allocated_budgets * reach_per_dollar).astype(int)
        return df_opt.sort_values(by="optimization_potential", ascending=False)
    else:
        st.warning("Optimization could not find a solution. This may be due to overly restrictive constraints (e.g., budget too low for the target customers). Please adjust the inputs.")
        return None

def ask_gemini(question, df, api_key):
    """Sends a question and dataframe context to the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        df_string = buffer.getvalue()

        prompt = f"""
        You are an expert marketing analyst AI. Your task is to provide data-driven strategic advice.
        
        CONTEXT:
        Here is the campaign performance data you need to analyze:
        --- START OF DATA ---
        {df_string}
        --- END OF DATA ---

        USER'S QUESTION:
        "{question}"

        Please provide a clear, concise, and actionable answer based on the data. If the user asks for a table, format your response accordingly.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}. Please check your API key and try again."


# --- Sidebar ---
with st.sidebar:
    st.title("🚀 Intelligent Campaign AI")
    
    # Use session state to track data source to avoid reloads
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "Use Sample Data"

    st.radio(
        "Select Data Source",
        ("Use Sample Data", "Upload Your Own Data"),
        key="data_source"
    )

    uploaded_file = None
    if st.session_state.data_source == "Upload Your Own Data":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xls", "xlsx"],
            label_visibility="collapsed"
        )
        
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ("Campaign Performance Dashboard", "Optimization Engine", "AI Insights by Generative AI Agent")
    )

# --- Data Loading and Processing Logic ---
# This block runs only when the source changes, not on every widget interaction.
if 'df' not in st.session_state:
    st.session_state.df = None

if st.session_state.data_source == "Upload Your Own Data":
    if uploaded_file is not None:
        raw_df = load_raw_data(uploaded_file)
        st.session_state.df = process_dataframe(raw_df)
    else:
        st.session_state.df = None # Clear df if no file is uploaded
else: # "Use Sample Data"
    raw_sample_df = get_raw_sample_data()
    st.session_state.df = process_dataframe(raw_sample_df)

# --- Main App Interface ---
if st.session_state.df is None:
    if st.session_state.data_source == "Upload Your Own Data":
         st.info("Please upload a data file to begin.")
    else: # Error case for missing sample data
         st.error("`sample_campaign_data.csv` not found. Please make sure it's in the same directory.")
    st.stop()

df = st.session_state.df

# (The rest of the main app interface code remains exactly the same as the previous robust version)
# ... (omitted for brevity, paste the main interface code from the previous answer here) ...
if analysis_mode == "Campaign Performance Dashboard":
    st.header("📊 Comprehensive Campaign Performance Dashboard")
    
    if st.session_state.data_source == "Upload Your Own Data" and uploaded_file:
        st.success(f"Dataset with {len(df)} campaigns successfully loaded!")
        st.dataframe(df.head())

    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Dimensional Analysis", "Correlation Insights", "Performance Radar", "Detailed Campaign Metrics"
    ])

    with tab1:
        st.subheader("Campaign Performance Landscape")
        fig = px.scatter(
            df, 
            x='historical_reach', 
            y='engagement_rate',
            size='ad_spend',
            color='campaign_risk',
            hover_name='campaign',
            hover_data={'ad_spend': ':,', 'campaign_risk': ':.2f'},
            color_continuous_scale=px.colors.sequential.Plasma_r,
            title="Campaign Performance: Reach vs. Engagement"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Campaign Metrics Correlation")
        corr = df.select_dtypes(include=np.number).corr()
        fig_heatmap = px.imshow(
            corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        st.subheader("Campaign Performance Radar")
        campaigns_to_compare = st.multiselect(
            "Select campaigns to compare:", options=df['campaign'].tolist(), default=df['campaign'].tolist()[:5]
        )
        if campaigns_to_compare:
            df_radar = df[df['campaign'].isin(campaigns_to_compare)]
            metrics_to_plot = ['ad_spend', 'historical_reach', 'engagement_rate', 'repeat_customer_rate']
            df_normalized = df_radar.copy()
            for col in metrics_to_plot:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val - min_val > 0:
                    df_normalized[col] = (df_radar[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[col] = 0.5  # Assign a neutral value if all values are the same
            
            fig_radar = go.Figure()
            for _, row in df_normalized.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[metrics_to_plot].values, theta=metrics_to_plot, fill='toself', name=row['campaign']
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Normalized Performance Metrics"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.subheader("Detailed Campaign Metrics & Growth Potential")
        st.dataframe(
            df.style.highlight_max(subset=['efficiency_score', 'potential_growth'], color='lightgreen', axis=0),
            use_container_width=True
        )
        st.download_button(
            "Export Data as CSV", df.to_csv(index=False).encode('utf-8'), "campaign_metrics.csv", "text/csv"
        )

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
            fig_bar = px.bar(
                optimized_df, x='campaign', y='allocated_budget', color='optimization_potential',
                color_continuous_scale='Greens', title=f"Optimized Budget Allocation for ${total_budget:,}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("📋 Optimization Breakdown")
            st.dataframe(
                optimized_df[['campaign', 'allocated_budget', 'estimated_reach', 'optimization_potential']],
                use_container_width=True
            )
            
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
        user_question = st.text_area(
            "Specific Campaign Strategy Question:", "Which campaign has the best performance and why?", height=100
        )
        submit_button = st.form_submit_button(label="Generate Strategic Insights")

    if submit_button:
        if not api_key:
            st.warning("Please enter your Gemini API key to proceed.")
        elif not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating insights..."):
                ai_response = ask_gemini(user_question, df, api_key)
            st.markdown("---"); st.subheader("Al Strategic Insights"); st.markdown(ai_response)
