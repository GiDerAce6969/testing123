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

@st.cache_data
def load_data(uploaded_file):
    """Loads data from CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def get_sample_data():
    """Loads the predefined sample CSV data."""
    try:
        return pd.read_csv('sample_campaign_data.csv')
    except FileNotFoundError:
        st.error("`sample_campaign_data.csv` not found. Please make sure it's in the same directory.")
        return pd.DataFrame() # Return empty dataframe

def calculate_metrics(df):
    """Calculates Efficiency Score and Potential Growth and adds them to the DataFrame."""
    df_processed = df.copy()
    # Avoid division by zero
    df_processed['Efficiency Score'] = (df_processed['Historical Reach'] / 
                                        (df_processed['Ad Spend'] * df_processed['Engagement Rate'] + 1e-6)).round(4)
    df_processed['Potential Growth'] = (df_processed['Repeat Customer Rate'] * df_processed['Seasonality Factor']).round(4)
    return df_processed

def run_optimization(df, total_budget, total_customers):
    """Performs linear programming to optimize budget allocation."""
    df_opt = df.copy()
    
    # Define Optimization Potential for the objective function
    # A higher potential score means the campaign is more desirable to fund
    df_opt['Optimization Potential'] = (df_opt['Historical Reach'] / df_opt['Ad Spend']) * df_opt['Engagement Rate'] * df_opt['Seasonality Factor']
    
    # Coefficients for the objective function (we want to maximize this, so we use negative for linprog)
    c = -df_opt['Optimization Potential'].values
    
    # Coefficients for inequality constraints (A_ub * x <= b_ub)
    # Constraint 1: Sum of allocated budgets <= total_budget
    # Constraint 2: Sum of estimated reach >= total_customers -> -Sum(reach) <= -total_customers
    reach_per_dollar = (df_opt['Historical Reach'] / df_opt['Ad Spend']).values
    A_ub = [
        np.ones(len(df_opt)),  # Budget constraint
        -reach_per_dollar      # Customer reach constraint
    ]
    b_ub = [
        total_budget,
        -total_customers
    ]
    
    # Bounds for each variable (budget for each campaign must be non-negative)
    bounds = [(0, None) for _ in range(len(df_opt))]
    
    # Solve the linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        allocated_budgets = result.x
        df_opt['Allocated Budget'] = allocated_budgets.round(2)
        df_opt['Estimated Reach'] = (allocated_budgets / df_opt['Ad Spend'] * df_opt['Historical Reach']).astype(int)
        return df_opt.sort_values(by="Optimization Potential", ascending=False)
    else:
        st.warning("Optimization could not find a solution. This may be due to overly restrictive constraints (e.g., budget too low for the target customers). Please adjust the inputs.")
        return None

def ask_gemini(question, df, api_key):
    """Sends a question and dataframe context to the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a string representation of the dataframe for the prompt
        # Using a buffer to handle potential large dataframes
        buffer = io.StringIO()
        df.to_csv(buffer)
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
    
    data_source = st.radio(
        "Select Data Source",
        ("Use Sample Data", "Upload Your Own Data"),
        key="data_source_radio"
    )

    uploaded_file = None
    if data_source == "Upload Your Own Data":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xls", "xlsx"],
            label_visibility="collapsed"
        )
        
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ("Campaign Performance Dashboard", "Optimization Engine", "AI Insights by Generative AI Agent")
    )


# --- Data Loading and Processing ---
# Use session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

if data_source == "Upload Your Own Data" and uploaded_file:
    df_loaded = load_data(uploaded_file)
    if df_loaded is not None:
        st.session_state.df = calculate_metrics(df_loaded)
elif data_source == "Use Sample Data":
    st.session_state.df = calculate_metrics(get_sample_data())

# Main app logic
if st.session_state.df is None:
    st.info("Please upload a data file or select 'Use Sample Data' to begin.")
    st.stop()

df = st.session_state.df

# --- Main App Interface ---

if analysis_mode == "Campaign Performance Dashboard":
    st.header("📊 Comprehensive Campaign Performance Dashboard")
    
    if data_source == "Upload Your Own Data" and uploaded_file:
        st.success(f"Dataset with {len(df)} campaigns successfully loaded!")
        st.dataframe(df.head())

    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Dimensional Analysis", 
        "Correlation Insights", 
        "Performance Radar", 
        "Detailed Campaign Metrics"
    ])

    with tab1:
        st.subheader("Campaign Performance Landscape")
        fig = px.scatter(
            df, 
            x='Historical Reach', 
            y='Engagement Rate',
            size='Ad Spend',
            color='Campaign Risk',
            hover_name='Campaign',
            hover_data={'Ad Spend': ':,', 'Campaign Risk': ':.2f'},
            color_continuous_scale=px.colors.sequential.Plasma_r,
            title="Campaign Performance: Reach vs. Engagement"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Campaign Metrics Correlation")
        corr = df.select_dtypes(include=np.number).corr()
        fig_heatmap = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap of Campaign Metrics"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        st.subheader("Campaign Performance Radar")
        campaigns_to_compare = st.multiselect(
            "Select campaigns to compare:",
            options=df['Campaign'].tolist(),
            default=df['Campaign'].tolist()[:5] # Default to first 5
        )
        if campaigns_to_compare:
            df_radar = df[df['Campaign'].isin(campaigns_to_compare)]
            
            # Normalize data for radar chart for better comparison
            metrics_to_plot = ['Ad Spend', 'Historical Reach', 'Engagement Rate', 'Repeat Customer Rate']
            df_normalized = df_radar.copy()
            for col in metrics_to_plot:
                df_normalized[col] = (df_radar[col] - df[col].min()) / (df[col].max() - df[col].min())

            fig_radar = go.Figure()
            for _, row in df_normalized.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[metrics_to_plot].values,
                    theta=metrics_to_plot,
                    fill='toself',
                    name=row['Campaign']
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Performance Metrics by Campaign"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.subheader("Detailed Campaign Metrics & Growth Potential")
        st.dataframe(
            df.style.highlight_max(subset=['Efficiency Score', 'Potential Growth'], color='lightgreen', axis=0),
            use_container_width=True
        )
        st.download_button(
            "Export Data as CSV",
            df.to_csv(index=False).encode('utf-8'),
            "campaign_metrics.csv",
            "text/csv"
        )

elif analysis_mode == "Optimization Engine":
    st.header("⚙️ AI-Powered Optimization Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.slider(
            "Total Marketing Budget ($)", 
            min_value=50000, 
            max_value=1000000, 
            value=200000, 
            step=10000
        )
    with col2:
        total_customers = st.slider(
            "Minimum Target Customers", 
            min_value=10000, 
            max_value=1000000, 
            value=100000, 
            step=5000
        )
        
    if st.button("🚀 Run AI-Powered Optimization", use_container_width=True):
        optimized_df = run_optimization(df, total_budget, total_customers)
        if optimized_df is not None:
            st.subheader("📈 AI Optimization Strategy")
            
            # Bar chart of allocated budget
            fig_bar = px.bar(
                optimized_df,
                x='Campaign',
                y='Allocated Budget',
                color='Optimization Potential',
                color_continuous_scale='Greens',
                title=f"Optimized Budget Allocation for ${total_budget:,}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Optimization breakdown table
            st.subheader("📋 Optimization Breakdown")
            st.dataframe(
                optimized_df[['Campaign', 'Allocated Budget', 'Estimated Reach', 'Optimization Potential']],
                use_container_width=True
            )
            
            # AI Strategic Recommendations (templated)
            st.subheader("💡 AI Strategic Recommendations")
            top_campaign = optimized_df.iloc[0]
            total_reach_original = df['Historical Reach'].sum()
            total_reach_optimized = optimized_df['Estimated Reach'].sum()
            reach_increase = ((total_reach_optimized - total_reach_original) / total_reach_original * 100) if total_reach_original > 0 else 100

            st.markdown(f"""
            - **🎯 Prioritize:** Focus investment on **{top_campaign['Campaign']}** which shows the highest optimization potential ({top_campaign['Optimization Potential']:.2f}).
            - **💰 Budget Reallocation:** The suggested reallocation could potentially increase overall reach by **{reach_increase:.2f}%**.
            - **👀 Top Performers:** The campaigns **'{optimized_df.iloc[0]['Campaign']}'** and **'{optimized_df.iloc[1]['Campaign']}'** show the most promise for high returns on investment.
            - **🤔 Review:** Consider adjusting strategies or reducing spend on campaigns with zero or low allocated budgets as they are less efficient under the current constraints.
            """)

elif analysis_mode == "AI Insights by Generative AI Agent":
    st.header("🤖 AI Insights by Generative AI Agent")
    
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password", help="Get your key from Google AI Studio.")
    
    with st.form(key="ai_form"):
        user_question = st.text_area(
            "Specific Campaign Strategy Question:",
            "Which campaign has the best performance and why? Give a detailed breakdown.",
            height=100
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
                st.markdown("---")
                st.subheader("Al Strategic Insights")
                st.markdown(ai_response)
