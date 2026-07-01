import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
@st.cache_resource
def load_model():
    return pickle.load(open("models/model.pkl","rb"))
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Public Health Data Analysis & Disease Pattern Identification",
    layout="wide"
)
px.defaults.template = "plotly_white"
# =========================
# LOAD DATA
# =========================
data = pd.read_csv("data/Global Health Statistics.csv")
# Sample for faster dashboard
data = data.sample(500000, random_state=42)
# =========================
# LOAD MODEL
# =========================
model = load_model()
le_disease = pickle.load(open("models/disease_encoder.pkl","rb"))
le_country = pickle.load(open("models/country_encoder.pkl","rb"))
le_gender = pickle.load(open("models/gender_encoder.pkl","rb"))
le_age = pickle.load(open("models/age_encoder.pkl","rb"))
# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Home"
# =========================
# SIDEBAR
# =========================
if st.session_state.page != "Home":

    page = st.sidebar.radio(
        "Navigation",
        ["📊 EDA Dashboard","🤖 Prediction","📈 Comparative Analysis"]
    )
    st.session_state.page = page
# =========================
# HOME PAGE
# =========================
if st.session_state.page == "Home":
    st.markdown("""
    <style>
     .stApp {
    background-color: #e0f7fa;
    }   
      /* REMOVE ONLY TOP EMPTY BOX */
    section.main > div:first-child {
    padding-top: 0rem !important;
    }      
    .block-container {
    padding-top: 1rem !important;  /* small spacing, not zero */
    }               
    .hero-container {
        padding: 60px 60px;
        background: linear-gradient(135deg, #e0f7fa, #f1f5f9);
        border-radius: 30px;
        box-shadow: 0px 25px 50px rgba(0,0,0,0.08);
        text-align: left;
    }
    .badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 30px;
    }
    .title {
        font-size: 58px;
        font-weight: 900;
        line-height: 1.1;
        background: linear-gradient(90deg, #0f172a, #0891b2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 25px;
    }
    .subtitle {
        font-size: 20px;
        font-weight: 600;
        color: #334155;
        max-width: 900px;
        margin-bottom: 50px;
    }
    .stButton>button {
    background: linear-gradient(90deg, #0284c7, #0369a1) !important;
    color: white !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    }
    .stButton>button:hover {
    background: linear-gradient(90deg, #0369a1, #075985) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="title">
    Public Health Data Analysis &<br>
    Disease Pattern Identification
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
    This system performs exploratory data analysis to identify top diseases,
    gender distribution, regional case comparison, and year-wise disease trends.
    It also integrates a trained Random Forest model to predict risk level.
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Explore Dashboard", use_container_width=True):
        st.session_state.page = "📈 EDA Dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
# =========================
# =========================
# =========================
# EDA DASHBOARD
# =========================
elif st.session_state.page == "📊 EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")
    # =========================
    # ADVANCED FILTERS
    # =========================
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_country = st.selectbox(
            "Select Country",
            ["All"] + list(data["Country"].unique())
        )
    with col2:
        selected_disease = st.selectbox(
            "Select Disease",
            ["All"] + list(data["Disease Name"].unique())
        )
    with col3:
        selected_gender = st.selectbox(
            "Select Gender",
            ["All"] + list(data["Gender"].unique())
        )
    col4, col5 = st.columns(2)
    with col4:
        selected_age = st.selectbox(
            "Select Age Group",
            ["All"] + list(data["Age Group"].unique())
        )
    with col5:
        selected_year = st.selectbox(
            "Select Year",
            ["All"] + sorted(data["Year"].unique())
        )
    # =========================
    # APPLY FILTERING
    # =========================
    filtered_data = data.copy()
    if selected_country != "All":
        filtered_data = filtered_data[filtered_data["Country"] == selected_country]
    if selected_disease != "All":
        filtered_data = filtered_data[filtered_data["Disease Name"] == selected_disease]
    if selected_gender != "All":
        filtered_data = filtered_data[filtered_data["Gender"] == selected_gender]
    if selected_age != "All":
        filtered_data = filtered_data[filtered_data["Age Group"] == selected_age]
    if selected_year != "All":
        filtered_data = filtered_data[filtered_data["Year"] == selected_year]
    # Show filtered size (optional but useful)
    st.write("Filtered Data Size:", len(filtered_data))
    # ------------------------
    # KPI METRICS
    # ------------------------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Diseases", filtered_data["Disease Name"].nunique())
    k2.metric("Countries", filtered_data["Country"].nunique())
    k3.metric("Records", len(filtered_data))
    k4.metric("Avg Prevalence", round(filtered_data["Prevalence Rate (%)"].mean(),2))
    st.divider()
    # ------------------------
    # ------------------------
    # Top Diseases (UPDATED)
    # ------------------------
    disease_counts = filtered_data.groupby("Disease Name")["Population Affected"].mean().sort_values(ascending=False).head(6).reset_index()
    fig1 = px.bar(
        disease_counts,
        x="Disease Name",
        y="Population Affected",
        text="Population Affected",
        color="Population Affected",
        color_continuous_scale=px.colors.sequential.Purples,
        title="Top Diseases by Average Population Affected"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig1, width="stretch")
    # ------------------------
    # Gender Distribution
    # ------------------------
    gender_counts = filtered_data["Gender"].value_counts().reset_index()
    gender_counts.columns=["Gender","Cases"]
    fig2 = px.pie(
        gender_counts,
        names="Gender",
        values="Cases",
        title="Disease Occurrence by Gender",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig2, width="stretch")
    # ------------------------
    # World Map
    # ------------------------
    country_cases = filtered_data.groupby("Country").size().reset_index(name="Cases")
    fig3 = px.choropleth(
        country_cases,
        locations="Country",
        locationmode="country names",
        color="Cases",
        color_continuous_scale="Reds",
        title="Disease Cases by Country"
    )
    st.plotly_chart(fig3, width="stretch")
    # ------------------------
    # Age Group Donut
    # ------------------------
    age_counts = filtered_data["Age Group"].value_counts().reset_index()
    age_counts.columns=["Age Group","Cases"]
    fig4 = px.pie(
        age_counts,
        names="Age Group",
        values="Cases",
        hole=0.5,
        title="Age Group Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig4, width="stretch")
    # ------------------------
    # Year Trend
    # ------------------------
    year_cases = filtered_data.groupby("Year").size().reset_index(name="Cases")
    fig7 = px.line(
        year_cases,
        x="Year",
        y="Cases",
        markers=True,
        title="Disease Trend Over Years"
    )
    st.plotly_chart(fig7, width="stretch")
    # ------------------------
    # Mortality Analysis
    # ------------------------
    severity = filtered_data.groupby("Disease Name")["Mortality Rate (%)"].mean().sort_values(ascending=False).head(6).reset_index()
    fig5 = px.bar(
        severity,
        x="Disease Name",
        y="Mortality Rate (%)",
        text="Mortality Rate (%)",
        color="Mortality Rate (%)",
        color_continuous_scale=px.colors.sequential.Peach,
        title="Average Mortality Rate by Disease"
    )
    fig5.update_traces(textposition="outside")
    fig5.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig5, width="stretch")
    # ------------------------
    # Population Impact
    # ------------------------
    pop_data = filtered_data.groupby("Disease Name")["Population Affected"].mean().sort_values(ascending=False).head(6).reset_index()
    fig6 = px.bar(
        pop_data,
        x="Disease Name",
        y="Population Affected",
        text="Population Affected",
        color="Population Affected",
        color_continuous_scale="Purples",
        title="Average Population Impact by Disease"
    )
    fig6.update_traces(textposition="outside")
    fig6.update_layout(xaxis_tickangle=-40)

    st.plotly_chart(fig6, width="stretch")
# =========================
# PREDICTION PAGE
# =========================
elif st.session_state.page == "🤖 Prediction":
    st.header("Disease Risk Level Prediction")
    disease = st.selectbox("Disease Name", le_disease.classes_)
    country = st.selectbox("Country", le_country.classes_)
    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.selectbox("Age Group", le_age.classes_)
    mortality = st.number_input("Mortality Rate (%)",0.0,100.0,1.0)
    population = st.number_input("Population Affected",1,100000000,1000)
    if st.button("Predict Risk Level"):
        disease_enc = le_disease.transform([disease])[0]
        country_enc = le_country.transform([country])[0]
        gender_enc = le_gender.transform([gender])[0]
        age_enc = le_age.transform([age])[0]
        input_df = pd.DataFrame([[disease_enc,country_enc,gender_enc,age_enc,mortality,population]],
                            columns=['Disease Name','Country','Gender','Age Group','Mortality Rate (%)','Population Affected'])
        if mortality == 0 and population == 0:
          st.success("🟢 Low Risk")
        else:
          prediction = model.predict(input_df)[0]
        # =========================
        # RESULT + INSIGHTS
        # =========================
        if prediction == "High":
            st.error("🔴 High Risk")
            st.markdown("""
            ### ⚠️ Why High Risk?
            - High mortality rate detected
            - Large population affected
            - Disease severity is high in selected region
            ### 🩺 Recommendations:
            - Immediate medical consultation
            - Follow vaccination if available
            - Maintain strict hygiene
            - Regular health monitoring
            """)
        elif prediction == "Medium":
            st.warning("🟠 Medium Risk")
            st.markdown("""
            ### ⚠️ Why Medium Risk?
            - Moderate spread observed
            - Needs attention but manageable
            ### 🩺 Recommendations:
            - Maintain hygiene
            - Monitor symptoms regularly
            - Avoid crowded places
            """)
        else:
            st.success("🟢 Low Risk")
            st.markdown("""
            ### ✅ Why Low Risk?
            - Low mortality rate
            - Controlled disease spread
            ### 🩺 Recommendations:
            - Maintain healthy lifestyle
            - Eat nutritious food
            - Regular exercise
            """)
        # Progress bar
        score_map = {"Low":30,"Medium":60,"High":90}
        st.progress(score_map[prediction]/100)
# =========================
# COMPARATIVE ANALYSIS PAGE
# =========================
# COMPARATIVE ANALYSIS PAGE
elif st.session_state.page == "📈 Comparative Analysis":
    st.header("📊 Comparative Analysis of ML Algorithms")
    results = pickle.load(open("models/model_results.pkl","rb"))
    models = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models]
    f1_scores = [results[m]["f1_macro"] for m in models]
    recall_scores = [results[m]["recall_macro"] for m in models]
    # ------------------------
    # Accuracy Chart
    # ------------------------
    fig1 = px.bar(
        x=models,
        y=accuracy,
        text=[round(a,3) for a in accuracy],
        title="Model Accuracy Comparison"
    )
    st.plotly_chart(fig1, width="stretch")
    # ------------------------
    # Accuracy vs Recall Chart
    # ------------------------
    df_compare = pd.DataFrame({
        "Model": models,
        "Accuracy": accuracy,
        "Recall": recall_scores
    })
    df_melt = df_compare.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig2 = px.bar(
        df_melt,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Accuracy vs Recall Comparison"
    )
    st.plotly_chart(fig2, width="stretch")
    # ------------------------
    # Table (LAST)
    # ------------------------
    table_df = pd.DataFrame({
        "Model": models,
        "Accuracy": [round(a,3) for a in accuracy],
        "F1 Score": [round(f,3) for f in f1_scores],
        "Recall": [round(r,3) for r in recall_scores]
    })
    st.subheader("📋 Model Comparison Table")
    st.dataframe(table_df)

    