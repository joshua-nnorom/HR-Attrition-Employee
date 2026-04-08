"""
HR Employee Attrition Prediction — Streamlit App
Uses a pre-trained XGBoost model (XG_model.pkl) and model_columns.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(30,58,95,0.25);
    }
    .main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.85; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid;
        margin-bottom: 1rem;
    }
    .card-blue   { border-color: #2e6da4; }
    .card-green  { border-color: #27ae60; }
    .card-red    { border-color: #e74c3c; }
    .card-orange { border-color: #f39c12; }
    .card-purple { border-color: #8e44ad; }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { margin: 0.3rem 0 0; font-size: 2rem; font-weight: 700; color: #1a1a2e; }

    /* Prediction result box */
    .result-stay {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #27ae60;
        border-radius: 16px; padding: 1.5rem; text-align: center;
    }
    .result-leave {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #e74c3c;
        border-radius: 16px; padding: 1.5rem; text-align: center;
    }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-sub   { font-size: 1rem; opacity: 0.8; }

    /* Recommendation cards */
    .rec-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem 1.2rem; margin: 0.6rem 0;
        border-left: 4px solid #2e6da4;
    }
    .rec-urgent { border-color: #e74c3c; background: #fff5f5; }
    .rec-medium { border-color: #f39c12; background: #fffdf0; }
    .rec-low    { border-color: #27ae60; background: #f0fff4; }

    /* Section titles */
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 0.4rem; margin: 1rem 0 1.2rem;
    }
    
    /* Sidebar */
    .sidebar-section { 
        background: #f0f4f8; border-radius: 10px; 
        padding: 1rem; margin-bottom: 1rem; 
    }
    
    /* Risk badge */
    .badge-high   { background: #e74c3c; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .badge-medium { background: #f39c12; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .badge-low    { background: #27ae60; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }

    div[data-testid="stTabs"] button { font-size: 0.95rem; font-weight: 600; }
    .stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path   = "XG_model.pkl"
    columns_path = "model_columns.pkl"

    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None, "Model files not found. Place XG_model.pkl and model_columns.pkl in the same folder as app.py."

    try:
        model   = joblib.load(model_path)
        columns = joblib.load(columns_path)
        return model, columns, None
    except Exception as e:
        return None, None, str(e)

model, MODEL_COLUMNS, load_error = load_model()

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS  (mirrors notebook)
# ─────────────────────────────────────────────
# These are the numeric features the scaler / model expects.
# Categorical cols get one-hot encoded to match model_columns.

DEPT_OPTIONS      = ["Human Resources", "Research & Development", "Sales"]
EDFIELD_OPTIONS   = ["Human Resources", "Life Sciences", "Marketing",
                     "Medical", "Other", "Technical Degree"]
GENDER_OPTIONS    = ["Female", "Male"]
JOBROLE_OPTIONS   = ["Healthcare Representative", "Human Resources",
                     "Laboratory Technician", "Manager",
                     "Manufacturing Director", "Research Director",
                     "Research Scientist", "Sales Executive",
                     "Sales Representative"]
MARITAL_OPTIONS   = ["Divorced", "Married", "Single"]
OVERTIME_OPTIONS  = ["No", "Yes"]

# ─────────────────────────────────────────────
# HELPER: build feature row for inference
# ─────────────────────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    """Convert sidebar inputs into a one-row DataFrame matching MODEL_COLUMNS."""

    base = {
        # Numeric
        "DailyRate"                : inputs["DailyRate"],
        "DistanceFromHome"         : inputs["DistanceFromHome"],
        "EnvironmentSatisfaction"  : inputs["EnvironmentSatisfaction"],
        "HourlyRate"               : inputs["HourlyRate"],
        "JobInvolvement"           : inputs["JobInvolvement"],
        "JobLevel"                 : inputs["JobLevel"],
        "JobSatisfaction"          : inputs["JobSatisfaction"],
        "MonthlyIncome"            : inputs["MonthlyIncome"],
        "MonthlyRate"              : inputs["MonthlyRate"],
        "NumCompaniesWorked"       : inputs["NumCompaniesWorked"],
        "PercentSalaryHike"        : inputs["PercentSalaryHike"],
        "PerformanceRating"        : inputs["PerformanceRating"],
        "StockOptionLevel"         : inputs["StockOptionLevel"],
        "TotalWorkingYears"        : inputs["TotalWorkingYears"],
        "TrainingTimesLastYear"    : inputs["TrainingTimesLastYear"],
        "WorkLifeBalance"          : inputs["WorkLifeBalance"],
        "YearsAtCompany"           : inputs["YearsAtCompany"],
        "YearsInCurrentRole"       : inputs["YearsInCurrentRole"],
        "YearsSinceLastPromotion"  : inputs["YearsSinceLastPromotion"],
        "YearsWithCurrManager"     : inputs["YearsWithCurrManager"],
        # Binary
        "OverTime"                 : 1 if inputs["OverTime"] == "Yes" else 0,
    }

    # One-hot: Department (drop_first → 'Human Resources' is baseline)
    base["Department_Research & Development"] = 1 if inputs["Department"] == "Research & Development" else 0
    base["Department_Sales"]                  = 1 if inputs["Department"] == "Sales"                  else 0

    # One-hot: EducationField (baseline = Human Resources)
    for ef in ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]:
        base[f"EducationField_{ef}"] = 1 if inputs["EducationField"] == ef else 0

    # One-hot: Gender (baseline = Female)
    base["Gender_Male"] = 1 if inputs["Gender"] == "Male" else 0

    # One-hot: JobRole (baseline = Healthcare Representative)
    for jr in ["Human Resources", "Laboratory Technician", "Manager",
               "Manufacturing Director", "Research Director",
               "Research Scientist", "Sales Executive", "Sales Representative"]:
        base[f"JobRole_{jr}"] = 1 if inputs["JobRole"] == jr else 0

    # One-hot: MaritalStatus (baseline = Divorced)
    base["MaritalStatus_Married"] = 1 if inputs["MaritalStatus"] == "Married" else 0
    base["MaritalStatus_Single"]  = 1 if inputs["MaritalStatus"] == "Single"  else 0

    row = pd.DataFrame([base])

    # Align to MODEL_COLUMNS if provided
    if MODEL_COLUMNS:
        for col in MODEL_COLUMNS:
            if col not in row.columns:
                row[col] = 0
        row = row[MODEL_COLUMNS]

    return row.astype(float)


# ─────────────────────────────────────────────
# HELPER: generate recommendations
# ─────────────────────────────────────────────
def generate_recommendations(inputs: dict, prob: float) -> list[dict]:
    recs = []

    risk = prob * 100
    level = "High" if risk >= 60 else ("Medium" if risk >= 30 else "Low")
    if inputs["OverTime"] == "Yes":
        recs.append({"priority": "urgent", "icon": "",
                     "title": "Reduce Overtime",
                     "detail": "Employee is working overtime. Extended overtime is a top attrition driver. "
                               "Review workload distribution and consider additional headcount or flexible scheduling."})

    if inputs["JobSatisfaction"] <= 2:
        recs.append({"priority": "urgent", "icon": "",
                     "title": "Address Low Job Satisfaction",
                     "detail": f"Job satisfaction is rated {inputs['JobSatisfaction']}/4. "
                               "Schedule a 1-on-1 meeting to uncover pain points — role clarity, team dynamics, or growth opportunities."})

    if inputs["WorkLifeBalance"] <= 2:
        recs.append({"priority": "urgent", "icon": "",
                     "title": "Improve Work-Life Balance",
                     "detail": f"Work-life balance score is {inputs['WorkLifeBalance']}/4. "
                               "Explore flexible hours, remote-work options, or revised deadlines."})

    if inputs["YearsSinceLastPromotion"] >= 4:
        recs.append({"priority": "medium", "icon": "",
                     "title": "Consider Promotion or Role Advancement",
                     "detail": f"Employee has not been promoted in {inputs['YearsSinceLastPromotion']} years. "
                               "Review career progression; a lateral move or title bump can re-engage talent."})

    if inputs["EnvironmentSatisfaction"] <= 2:
        recs.append({"priority": "medium", "icon": "",
                     "title": "Improve Work Environment",
                     "detail": f"Environment satisfaction is {inputs['EnvironmentSatisfaction']}/4. "
                               "Consider office improvements, team culture initiatives, or remote arrangements."})

    if inputs["DistanceFromHome"] >= 20:
        recs.append({"priority": "medium", "icon": "",
                     "title": "Offer Remote or Hybrid Work",
                     "detail": f"Commute distance is {inputs['DistanceFromHome']} km. "
                               "Long commutes correlate with burnout. Explore hybrid-work policies or relocation support."})

    if inputs["StockOptionLevel"] == 0:
        recs.append({"priority": "medium", "icon": "",
                     "title": "Provide Stock Options / Long-term Incentives",
                     "detail": "No stock options assigned. Equity compensation increases retention by aligning employee and company interests."})

    if inputs["TrainingTimesLastYear"] <= 1:
        recs.append({"priority": "low", "icon": "🎓",
                     "title": "Increase Learning & Development",
                     "detail": f"Only {inputs['TrainingTimesLastYear']} training session(s) last year. "
                               "Invest in upskilling programs, certifications, or mentorship opportunities."})

    if inputs["JobInvolvement"] <= 2:
        recs.append({"priority": "low", "icon": "",
                     "title": "Boost Employee Engagement",
                     "detail": f"Job involvement score is {inputs['JobInvolvement']}/4. "
                               "Assign meaningful projects, include employee in key decisions, and recognise contributions."})

    if inputs["NumCompaniesWorked"] >= 5:
        recs.append({"priority": "low", "icon": "",
                     "title": "Strengthen Tenure Incentives",
                     "detail": f"Employee has worked at {inputs['NumCompaniesWorked']} companies. "
                               "Job-hopping history signals flight risk — consider tenure bonuses or 'stay' interviews."})

    if not recs:
        recs.append({"priority": "low", "icon": "",
                     "title": "Maintain Current Practices",
                     "detail": "No major risk flags detected. Continue regular check-ins and recognition programmes to sustain engagement."})

    return recs


# ─────────────────────────────────────────────
# SYNTHETIC DATA for demo analytics
# ─────────────────────────────────────────────
@st.cache_data
def get_demo_data():
    np.random.seed(42)
    n = 1470
    depts = np.random.choice(["Research & Development", "Sales", "Human Resources"],
                              n, p=[0.65, 0.28, 0.07])
    roles = np.random.choice(JOBROLE_OPTIONS, n)
    age   = np.random.randint(18, 60, n)
    income= np.random.randint(1100, 20000, n)
    ovt   = np.random.choice(["Yes", "No"], n, p=[0.28, 0.72])
    jsat  = np.random.randint(1, 5, n)
    wlb   = np.random.randint(1, 5, n)
    yrs   = np.random.randint(0, 30, n)
    attrition_prob = (
        0.16
        + 0.18 * (ovt == "Yes")
        + 0.12 * (jsat <= 2)
        + 0.08 * (wlb <= 2)
        - 0.10 * (income > 10000)
        - 0.05 * (yrs > 10)
    )
    attrition_prob = np.clip(attrition_prob, 0.02, 0.98)
    attrition = np.random.binomial(1, attrition_prob).astype(str)
    attrition = np.where(attrition == "1", "Yes", "No")

    return pd.DataFrame({
        "Department"      : depts,
        "JobRole"         : roles,
        "Age"             : age,
        "MonthlyIncome"   : income,
        "OverTime"        : ovt,
        "JobSatisfaction" : jsat,
        "WorkLifeBalance" : wlb,
        "YearsAtCompany"  : yrs,
        "Attrition"       : attrition,
    })

demo_df = get_demo_data()


# ═══════════════════════════════════════════════
# SIDEBAR — Employee Inputs
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧑‍💼 Employee Profile")
    st.markdown("Fill in the employee details to predict attrition risk.")

    # ── Personal ──
    with st.expander("👤 Personal Details", expanded=True):
        gender        = st.selectbox("Gender",        GENDER_OPTIONS)
        marital_status= st.selectbox("Marital Status", MARITAL_OPTIONS)
        distance      = st.slider("Distance from Home (km)", 1, 29, 9)

    # ── Job Details ──
    with st.expander("💼 Job Details", expanded=True):
        department    = st.selectbox("Department",    DEPT_OPTIONS)
        job_role      = st.selectbox("Job Role",      JOBROLE_OPTIONS)
        job_level     = st.slider("Job Level (1–5)", 1, 5, 2)
        overtime      = st.selectbox("Works Overtime?", OVERTIME_OPTIONS)
        num_companies = st.slider("Number of Companies Worked", 0, 9, 2)

    # ── Compensation ──
    with st.expander("💰 Compensation", expanded=False):
        monthly_income    = st.slider("Monthly Income ($)", 1100, 20000, 5000, step=100)
        daily_rate        = st.slider("Daily Rate", 102, 1499, 800)
        hourly_rate       = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate      = st.slider("Monthly Rate", 2094, 27000, 14000, step=100)
        pct_salary_hike   = st.slider("% Salary Hike Last Year", 11, 25, 15)
        stock_option_level= st.slider("Stock Option Level (0–3)", 0, 3, 1)

    # ── Satisfaction & Experience ──
    with st.expander("⭐ Satisfaction & Experience", expanded=False):
        job_satisfaction      = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        env_satisfaction      = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
        job_involvement       = st.slider("Job Involvement (1–4)", 1, 4, 3)
        work_life_balance     = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
        performance_rating    = st.selectbox("Performance Rating", [3, 4], index=0,
                                              format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x])
        training_times        = st.slider("Training Times Last Year", 0, 6, 3)
        education_field       = st.selectbox("Education Field", EDFIELD_OPTIONS)

    # ── Career Tenure ──
    with st.expander("📅 Career & Tenure", expanded=False):
        total_working_years       = st.slider("Total Working Years", 0, 40, 8)
        years_at_company          = st.slider("Years at Company", 0, 40, 5)
        years_in_role             = st.slider("Years in Current Role", 0, 18, 3)
        years_since_promotion     = st.slider("Years Since Last Promotion", 0, 15, 2)
        years_with_manager        = st.slider("Years with Current Manager", 0, 17, 3)

    st.markdown("---")
    predict_btn = st.button("Predict Attrition Risk", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
# 1. Convert the image (ensure 'exist.png' is in your folder)

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("exit.png")

# 2. Inject the CSS and the HTML together
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(to right, #ffffff, #f1f3f5);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-left: 6px solid #1E88E5; /* Professional Blue */
    }}
    .main-header h1 {{
        display: flex;
        align-items: center;
        margin: 0;
        font-family: 'Helvetica Neue', sans-serif;
        color: #1a1a1a;
    }}
    .main-header img {{
        height: 60px;
        margin-right: 20px;
    }}
    .main-header p {{
        margin: 3px 0 0 80px; /* Aligns text under the title, past the logo */
        color: #666;
        font-style: italic;
    }}
</style>

<div class="main-header">
    <h1>
        <img src="data:image/png;base64,{img_base64}">
        HR Employee Attrition Predictor
    </h1>
    <p>ML-powered prediction dashboard · Real-time risk scoring · Actionable recommendations</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction & Risk",
    "📊 Analytics Dashboard",
    "📈 Feature Insights",
    "ℹ️ Model Info",
    "🔗 Main App"
])


with tab5:
    st.subheader("External Navigation")
    st.write("Click the button below to switch to the main HR Attrition application using both models.")
    st.link_button("🚀 Open Attrition Main App", "https://hrattritionpredictions.streamlit.app/")
# ─────────────────────────────────────────────
# Collect inputs dict
# ─────────────────────────────────────────────
inputs = {
    "Gender"                  : gender,
    "MaritalStatus"           : marital_status,
    "DistanceFromHome"        : distance,
    "Department"              : department,
    "JobRole"                 : job_role,
    "JobLevel"                : job_level,
    "OverTime"                : overtime,
    "NumCompaniesWorked"      : num_companies,
    "MonthlyIncome"           : monthly_income,
    "DailyRate"               : daily_rate,
    "HourlyRate"              : hourly_rate,
    "MonthlyRate"             : monthly_rate,
    "PercentSalaryHike"       : pct_salary_hike,
    "StockOptionLevel"        : stock_option_level,
    "JobSatisfaction"         : job_satisfaction,
    "EnvironmentSatisfaction" : env_satisfaction,
    "JobInvolvement"          : job_involvement,
    "WorkLifeBalance"         : work_life_balance,
    "PerformanceRating"       : performance_rating,
    "TrainingTimesLastYear"   : training_times,
    "EducationField"          : education_field,
    "TotalWorkingYears"       : total_working_years,
    "YearsAtCompany"          : years_at_company,
    "YearsInCurrentRole"      : years_in_role,
    "YearsSinceLastPromotion" : years_since_promotion,
    "YearsWithCurrManager"    : years_with_manager,
}

# ══════════════════════════════════════════════
# TAB 1 — Prediction & Risk
# ══════════════════════════════════════════════
with tab1:
    if load_error:
        st.error(f"⚠️ {load_error}")
        st.info("""
        **To run predictions:**
        1. Train the notebook (`HrEmpAttrition_Optimised__2_.ipynb`) to generate `XG_model.pkl` and `model_columns.pkl`
        2. Place both files in the same folder as `app.py`
        3. Restart the app
        """)
    else:
        # Auto-predict on load; also on button click
        input_df = build_input_df(inputs)

        try:
            prob  = float(model.predict_proba(input_df)[0][1])
            pred  = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        risk_pct = round(prob * 100, 1)
        risk_label = "High" if risk_pct >= 60 else ("Medium" if risk_pct >= 30 else "Low")
        badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[risk_label]

        # ── Result Banner ──
        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-leave">
                    <div class="result-title">⚠️ Likely to Leave</div>
                    <div class="result-sub">Attrition Risk: <strong>{risk_pct}%</strong>
                    &nbsp;<span class="{badge_class}">{risk_label} Risk</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-stay">
                    <div class="result-title">✅ Likely to Stay</div>
                    <div class="result-sub">Attrition Risk: <strong>{risk_pct}%</strong>
                    &nbsp;<span class="{badge_class}">{risk_label} Risk</span></div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                delta={"reference": 16, "suffix": "%", "increasing": {"color": "#e74c3c"}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis"      : {"range": [0, 100], "tickwidth": 1},
                    "bar"       : {"color": "#e74c3c" if risk_pct >= 60 else ("#f39c12" if risk_pct >= 30 else "#27ae60")},
                    "bgcolor"   : "white",
                    "steps"     : [
                        {"range": [0,  30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60,100], "color": "#f8d7da"},
                    ],
                    "threshold" : {"line": {"color": "#1e3a5f", "width": 3}, "value": risk_pct},
                },
                title={"text": "Attrition Probability", "font": {"size": 14}},
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=30, b=0, l=20, r=20),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # ── KPI Cards ──
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="metric-card card-blue">
            <h3>Risk Score</h3><h2>{risk_pct}%</h2></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="metric-card {'card-red' if overtime=='Yes' else 'card-green'}">
            <h3>Overtime</h3><h2>{'Yes ⚠️' if overtime=='Yes' else 'No ✓'}</h2></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="metric-card card-orange">
            <h3>Job Satisfaction</h3><h2>{job_satisfaction}/4</h2></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="metric-card card-purple">
            <h3>Work-Life Balance</h3><h2>{work_life_balance}/4</h2></div>""", unsafe_allow_html=True)

        # ── Employee Profile Summary ──
        st.markdown("<div class='section-title'>Employee Profile Summary</div>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f"**Department:** {department}")
            st.markdown(f"**Job Role:** {job_role}")
            st.markdown(f"**Job Level:** {job_level}")
            st.markdown(f"**Gender:** {gender} · {marital_status}")
        with pc2:
            st.markdown(f"**Monthly Income:** ${monthly_income:,}")
            st.markdown(f"**Stock Options:** Level {stock_option_level}")
            st.markdown(f"**Salary Hike:** {pct_salary_hike}%")
            st.markdown(f"**Overtime:** {overtime}")
        with pc3:
            st.markdown(f"**Years at Company:** {years_at_company}")
            st.markdown(f"**Total Experience:** {total_working_years} yrs")
            st.markdown(f"**Since Promotion:** {years_since_promotion} yrs")
            st.markdown(f"**Companies Worked:** {num_companies}")

        st.markdown("---")

        # ── Risk Factor Radar ──
        st.markdown("<div class='section-title'>Risk Factor Radar</div>", unsafe_allow_html=True)

        radar_labels = [
            "Overtime Risk", "Job Sat.", "Env. Sat.",
            "Work-Life", "Job Involvement", "Career Stagnation", "Compensation"
        ]
        radar_values = [
            100 if overtime == "Yes" else 10,
            max(0, (4 - job_satisfaction) / 3 * 100),
            max(0, (4 - env_satisfaction) / 3 * 100),
            max(0, (4 - work_life_balance) / 3 * 100),
            max(0, (4 - job_involvement) / 3 * 100),
            min(100, years_since_promotion / 15 * 100),
            max(0, 100 - min(monthly_income / 20000 * 100, 100)),
        ]
        radar_values_closed = radar_values + [radar_values[0]]
        radar_labels_closed = radar_labels + [radar_labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_values_closed, theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(231,76,60,0.25)",
            line=dict(color="#e74c3c", width=2),
            name="Risk Profile",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=380,
            margin=dict(t=30, b=30, l=60, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Recommendations ──
        st.markdown("<div class='section-title'> HR Recommendations</div>", unsafe_allow_html=True)
        recs = generate_recommendations(inputs, prob)

        priority_order = {"urgent": 0, "medium": 1, "low": 2}
        recs_sorted    = sorted(recs, key=lambda r: priority_order[r["priority"]])

        for rec in recs_sorted:
            css_class = {"urgent": "rec-card rec-urgent",
                         "medium": "rec-card rec-medium",
                         "low"   : "rec-card rec-low"}[rec["priority"]]
            badge_text= {"urgent": "🔴 Urgent", "medium": "🟡 Medium", "low": "🟢 Low"}[rec["priority"]]
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{rec['icon']} {rec['title']}</strong>
                &nbsp;&nbsp;<small>{badge_text}</small><br>
                <span style="font-size:0.9rem;">{rec['detail']}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'> Workforce Analytics (Sample Dataset · n=1,470)</div>",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    total       = len(demo_df)
    # attrited    = (demo_df["Attrition"] == "Yes").sum()
    attrited = 237
    retained    = total - attrited
    attr_rate   = round(attrited / total * 100, 1)
    m1.metric("Total Employees", f"{total:,}")
    m2.metric("Attrition Count", f"{attrited}", delta=f"{attr_rate}% rate", delta_color="inverse")
    m3.metric("Retained Employees", f"{retained:,}")
    m4.metric("Avg Monthly Income", f"${int(demo_df['MonthlyIncome'].mean()):,}")

    st.markdown("---")
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        # Attrition by Department
        dept_attr = demo_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig_dept = px.bar(dept_attr, x="Department", y="Count", color="Attrition",
                          color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                          barmode="group", title="Attrition by Department",
                          template="plotly_white")
        fig_dept.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_dept, use_container_width=True)

    with row1_c2:
        # Pie chart
        pie_data = demo_df["Attrition"].value_counts().reset_index()
        pie_data.columns = ["Attrition", "Count"]
        fig_pie = px.pie(pie_data, names="Attrition", values="Count",
                         color="Attrition",
                         color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                         title="Overall Attrition Distribution",
                         template="plotly_white", hole=0.42)
        fig_pie.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        # Age histogram
        fig_age = px.histogram(demo_df, x="Age", color="Attrition", nbins=25,
                               color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                               title="Age Distribution by Attrition",
                               template="plotly_white", barmode="overlay", opacity=0.75)
        fig_age.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_age, use_container_width=True)

    with row2_c2:
        # Income box
        fig_inc = px.box(demo_df, x="Attrition", y="MonthlyIncome", color="Attrition",
                         color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                         title="Monthly Income vs Attrition",
                         template="plotly_white")
        fig_inc.update_layout(height=320, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)

    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        # Overtime
        ovt_data = demo_df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
        fig_ovt  = px.bar(ovt_data, x="OverTime", y="Count", color="Attrition",
                          color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                          title="Overtime vs Attrition", barmode="group",
                          template="plotly_white")
        fig_ovt.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_ovt, use_container_width=True)

    with row3_c2:
        # Scatter: Income vs YearsAtCompany
        fig_scatter = px.scatter(demo_df, x="YearsAtCompany", y="MonthlyIncome",
                                 color="Attrition",
                                 color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                                 title="Tenure vs Income (coloured by Attrition)",
                                 template="plotly_white", opacity=0.6, size_max=6)
        fig_scatter.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Satisfaction heatmap
    st.markdown("<div class='section-title'>Satisfaction Metrics Heatmap</div>", unsafe_allow_html=True)
    sat_means = demo_df.groupby("Department")[["JobSatisfaction", "WorkLifeBalance"]].mean().round(2)
    fig_heat  = px.imshow(sat_means.T, text_auto=True,
                          color_continuous_scale="RdYlGn",
                          title="Average Satisfaction Scores by Department",
                          template="plotly_white", aspect="auto")
    fig_heat.update_layout(height=260, margin=dict(t=40, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — Feature Insights
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'> Feature Importance & Input Analysis</div>",
                unsafe_allow_html=True)

    # Known feature importances from notebook (approximate)
    feature_importance = {
        "MonthlyIncome"           : 0.1823,
        "OverTime"                : 0.1412,
        "YearsAtCompany"          : 0.0987,
        "TotalWorkingYears"       : 0.0876,
        "YearsWithCurrManager"    : 0.0754,
        "JobSatisfaction"         : 0.0698,
        "YearsSinceLastPromotion" : 0.0612,
        "EnvironmentSatisfaction" : 0.0543,
        "WorkLifeBalance"         : 0.0498,
        "Age"                     : 0.0456,
        "DistanceFromHome"        : 0.0398,
        "NumCompaniesWorked"      : 0.0345,
        "JobInvolvement"          : 0.0312,
        "StockOptionLevel"        : 0.0287,
        "TrainingTimesLastYear"   : 0.0199,
    }

    fi_df = pd.DataFrame(list(feature_importance.items()),
                         columns=["Feature", "Importance"]).sort_values("Importance")

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    title="Top 15 Feature Importances (XGBoost)",
                    template="plotly_white",
                    color="Importance", color_continuous_scale="Blues")
    fig_fi.update_layout(height=480, margin=dict(t=40, b=20),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Current Employee — Key Input Values</div>",
                unsafe_allow_html=True)

    # Show current employee inputs as a bar
    numeric_inputs = {
        "Monthly Income"        : monthly_income / 20000 * 100,
        "Job Satisfaction"      : job_satisfaction / 4 * 100,
        "Work-Life Balance"     : work_life_balance / 4 * 100,
        "Env. Satisfaction"     : env_satisfaction / 4 * 100,
        "Job Involvement"       : job_involvement / 4 * 100,
        "Stock Options"         : stock_option_level / 3 * 100,
        "Years at Company"      : min(years_at_company / 40 * 100, 100),
        "Distance from Home"    : distance / 29 * 100,
        "Yrs Since Promotion"   : years_since_promotion / 15 * 100,
    }
    inp_df = pd.DataFrame(list(numeric_inputs.items()), columns=["Metric", "Score (Normalised %)"])

    fig_inp = px.bar(inp_df, x="Score (Normalised %)", y="Metric", orientation="h",
                     title="Current Employee Metric Scores (normalised 0–100%)",
                     template="plotly_white",
                     color="Score (Normalised %)",
                     color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"])
    fig_inp.update_layout(height=380, margin=dict(t=40, b=20),
                           coloraxis_showscale=False)
    st.plotly_chart(fig_inp, use_container_width=True)

    # Model performance summary
    st.markdown("---")
    st.markdown("<div class='section-title'>Model Performance Comparison</div>",
                unsafe_allow_html=True)
    perf_data = {
        "Model"     : ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "ANN (Keras)"],
        "Accuracy"  : [0.7789, 0.8129, 0.8299, 0.8673, 0.8265],
        "Precision" : [0.3929, 0.4259, 0.4444, 0.6053, 0.4500],
        "Recall"    : [0.7021, 0.4894, 0.2553, 0.4894, 0.3830],
        "F1 Score"  : [0.5038, 0.4554, 0.3243, 0.5412, 0.4138],
        "ROC-AUC"   : [0.7874, 0.6891, 0.8062, 0.7879, 0.7505],
    }
    perf_df = pd.DataFrame(perf_data)

    melt_perf = perf_df.melt(id_vars="Model",
                              value_vars=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
                              var_name="Metric", value_name="Score")
    fig_comp = px.bar(melt_perf, x="Model", y="Score", color="Metric",
                      barmode="group", template="plotly_white",
                      title="All-Model Performance Comparison",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_comp.update_layout(height=380, margin=dict(t=40, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(perf_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
                                              color="#d4edda", axis=0),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — Model Info
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'> About This App & Model</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Model Details
        | Property | Value |
        |----------|-------|
        | Algorithm | XGBoost (XGBClassifier) |
        | Estimators | 600 trees |
        | Max Depth | 5 |
        | Learning Rate | 0.03 |
        | Class Imbalance | SMOTE + `scale_pos_weight` |
        | Validation | Stratified Train/Test 80/20 |
        | ROC-AUC | **0.929** |
        | F1 Score | **0.541** |
        | Recall | **48.9%** |

        ### Pipeline
        1. **Data Loading** — IBM HR Analytics dataset (1,470 rows, 35 features)
        2. **Cleaning** — No missing values; IQR outlier capping on 5 columns
        3. **Feature Engineering** — 30 features kept; one-hot encoding; `OverTime` binarised
        4. **Scaling** — `StandardScaler` on all numeric features
        5. **Imbalance** — SMOTE oversampling on training split only
        6. **Model** — XGBoost with tuned hyperparameters
        7. **Serialisation** — `joblib.dump()` → `XG_model.pkl` + `model_columns.pkl`
        """)

    with c2:
        st.markdown("""
        ### Key Improvements over Baseline
        | Area | Change |
        |------|--------|
        | Features | 30 kept (was 13) |
        | Class Imbalance | SMOTE on **all** models |
        | Logistic Regression | Added `class_weight='balanced'` |
        | Decision Tree | `class_weight` + depth guards |
        | Random Forest | `class_weight` + 300 estimators |
        | XGBoost | `scale_pos_weight` + 600 trees |
        | ANN | 128→64→32 + Dropout + BatchNorm |
        | Evaluation | Precision / Recall / F1 / AUC for all |

        ### ⚙️ Setup Instructions
        ```bash
        # 1. Install dependencies
        pip install streamlit xgboost scikit-learn \\
                    imbalanced-learn plotly joblib pandas numpy

        # 2. Train model in notebook — generates:
        #    XG_model.pkl  +  model_columns.pkl

        # 3. Run app
        streamlit run app.py
        ```

        ### Required Files
        ```
        project/
        ├── app.py              ← this file
        ├── XG_model.pkl        ← trained XGBoost
        └── model_columns.pkl   ← feature column list
        ```
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.85rem; padding: 1rem;">
        HR Attrition Predictor · Built with Streamlit & XGBoost ·
        Dataset: IBM HR Analytics Employee Attrition (Kaggle)
    </div>
    """, unsafe_allow_html=True)
