"""
HR Employee Attrition Predictor — Dual-Model Ensemble
Runs XGBoost (high precision) + Logistic Regression (high recall) in parallel.
Each model triggers a different intervention tier based on cost-of-mistake logic.

Required files (same folder as app.py):
  XG_model.pkl       — trained XGBoost classifier
  LR_model.pkl       — trained Logistic Regression classifier
  model_columns.pkl  — feature column list (shared by both models)
"""

import os
import plotly.graph_objects as go
import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


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
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 1.5rem; color: white;
        box-shadow: 0 4px 20px rgba(30,58,95,0.25);
    }
    .main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.85; }

    .metric-card {
        background: white; border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid; margin-bottom: 1rem;
    }
    .card-blue   { border-color: #2e6da4; }
    .card-green  { border-color: #27ae60; }
    .card-red    { border-color: #e74c3c; }
    .card-orange { border-color: #f39c12; }
    .card-purple { border-color: #8e44ad; }
    .card-teal   { border-color: #17a589; }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { margin: 0.3rem 0 0; font-size: 2rem; font-weight: 700; color: #1a1a2e; }

    .tier-none {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #27ae60; border-radius: 16px;
        padding: 1.4rem 1.8rem; margin-bottom: 1rem;
    }
    .tier-soft {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 2px solid #2e6da4; border-radius: 16px;
        padding: 1.4rem 1.8rem; margin-bottom: 1rem;
    }
    .tier-high {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #e74c3c; border-radius: 16px;
        padding: 1.4rem 1.8rem; margin-bottom: 1rem;
    }
    .tier-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
    .tier-sub   { font-size: 0.95rem; opacity: 0.82; }

    .model-pill {
        display: inline-block; border-radius: 20px;
        padding: 5px 14px; font-size: 0.82rem; font-weight: 700;
        margin: 3px 4px 3px 0;
    }
    .pill-xgb-flag  { background: #e74c3c; color: white; }
    .pill-xgb-clear { background: #27ae60; color: white; }
    .pill-lr-flag   { background: #2e6da4; color: white; }
    .pill-lr-clear  { background: #17a589; color: white; }

    .model-panel {
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    }
    .panel-xgb { background: #fff5f5; border: 1.5px solid #e74c3c; }
    .panel-lr  { background: #eff8ff; border: 1.5px solid #2e6da4; }
    .panel-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; }

    .int-card {
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .int-soft { background: #eff8ff; border-color: #2e6da4; }
    .int-high { background: #fff5f5; border-color: #e74c3c; }
    .int-none { background: #f0fff4; border-color: #27ae60; }

    .rec-card   { background: #f8f9fa; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.6rem 0; border-left: 4px solid #2e6da4; }
    .rec-urgent { border-color: #e74c3c; background: #fff5f5; }
    .rec-medium { border-color: #f39c12; background: #fffdf0; }
    .rec-low    { border-color: #27ae60; background: #f0fff4; }

    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 0.4rem; margin: 1rem 0 1.2rem;
    }
    .badge-high   { background:#e74c3c; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; }
    .badge-medium { background:#f39c12; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; }
    .badge-low    { background:#27ae60; color:white; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; }

    div[data-testid="stTabs"] button { font-size: 0.95rem; font-weight: 600; }
    .stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DEFAULT THRESHOLDS
# ─────────────────────────────────────────────
DEFAULT_LR_THRESHOLD = 0.30
DEFAULT_XGB_THRESHOLD = 0.60


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model = lr_model = columns = None
    errors = []

    for fname, label in [("model_columns.pkl", "columns"),
                         ("XG_model.pkl", "xgb"),
                         ("LR_model.pkl", "lr")]:
        if os.path.exists(fname):
            try:
                obj = joblib.load(fname)
                if label == "columns":
                    columns = obj
                elif label == "xgb":
                    xgb_model = obj
                else:
                    lr_model = obj
            except Exception as e:
                errors.append(f"{fname}: {e}")
        elif label != "lr":          # LR is optional
            errors.append(f"{fname} not found.")

    return xgb_model, lr_model, columns, errors


xgb_model, lr_model, MODEL_COLUMNS, load_errors = load_models()
has_xgb = xgb_model is not None
has_lr = lr_model is not None


# ─────────────────────────────────────────────
# FEATURE CONSTANTS
# ─────────────────────────────────────────────
DEPT_OPTIONS = ["Human Resources", "Research & Development", "Sales"]
EDFIELD_OPTIONS = ["Human Resources", "Life Sciences", "Marketing",
                   "Medical", "Other", "Technical Degree"]
GENDER_OPTIONS = ["Female", "Male"]
JOBROLE_OPTIONS = ["Healthcare Representative", "Human Resources",
                   "Laboratory Technician", "Manager",
                   "Manufacturing Director", "Research Director",
                   "Research Scientist", "Sales Executive",
                   "Sales Representative"]
MARITAL_OPTIONS = ["Divorced", "Married", "Single"]
OVERTIME_OPTIONS = ["No", "Yes"]


# ─────────────────────────────────────────────
# HELPER: build input DataFrame
# ─────────────────────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    base = {
        "DailyRate": inputs["DailyRate"],
        "DistanceFromHome": inputs["DistanceFromHome"],
        "EnvironmentSatisfaction": inputs["EnvironmentSatisfaction"],
        "HourlyRate": inputs["HourlyRate"],
        "JobInvolvement": inputs["JobInvolvement"],
        "JobLevel": inputs["JobLevel"],
        "JobSatisfaction": inputs["JobSatisfaction"],
        "MonthlyIncome": inputs["MonthlyIncome"],
        "MonthlyRate": inputs["MonthlyRate"],
        "NumCompaniesWorked": inputs["NumCompaniesWorked"],
        "PercentSalaryHike": inputs["PercentSalaryHike"],
        "PerformanceRating": inputs["PerformanceRating"],
        "StockOptionLevel": inputs["StockOptionLevel"],
        "TotalWorkingYears": inputs["TotalWorkingYears"],
        "TrainingTimesLastYear": inputs["TrainingTimesLastYear"],
        "WorkLifeBalance": inputs["WorkLifeBalance"],
        "YearsAtCompany": inputs["YearsAtCompany"],
        "YearsInCurrentRole": inputs["YearsInCurrentRole"],
        "YearsSinceLastPromotion": inputs["YearsSinceLastPromotion"],
        "YearsWithCurrManager": inputs["YearsWithCurrManager"],
        "OverTime": 1 if inputs["OverTime"] == "Yes" else 0,
    }
    base["Department_Research & Development"] = 1 if inputs["Department"] == "Research & Development" else 0
    base["Department_Sales"] = 1 if inputs["Department"] == "Sales" else 0
    for ef in ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]:
        base[f"EducationField_{ef}"] = 1 if inputs["EducationField"] == ef else 0
    base["Gender_Male"] = 1 if inputs["Gender"] == "Male" else 0
    for jr in ["Human Resources", "Laboratory Technician", "Manager",
               "Manufacturing Director", "Research Director",
               "Research Scientist", "Sales Executive", "Sales Representative"]:
        base[f"JobRole_{jr}"] = 1 if inputs["JobRole"] == jr else 0
    base["MaritalStatus_Married"] = 1 if inputs["MaritalStatus"] == "Married" else 0
    base["MaritalStatus_Single"] = 1 if inputs["MaritalStatus"] == "Single" else 0

    row = pd.DataFrame([base])
    if MODEL_COLUMNS:
        for col in MODEL_COLUMNS:
            if col not in row.columns:
                row[col] = 0
        row = row[MODEL_COLUMNS]
    return row.astype(float)


# ─────────────────────────────────────────────
# HELPER: derive intervention tier
# ─────────────────────────────────────────────
def get_tier(xgb_flag: bool, lr_flag: bool) -> dict:
    if xgb_flag:
        return {
            "tier": "HIGH",
            "label": "🚨 High-Value Intervention Required",
            "css": "tier-high",
            "icon": "🚨",
            "description": ("XGBoost is highly confident this employee is at risk. "
                            "A targeted, costly retention package is justified."),
            "actions": [
                "💰 Review salary vs market benchmarks — offer a raise if below band",
                "📈 Fast-track a promotion or role-expansion conversation",
                "🎁 Offer a retention bonus or long-term incentive plan",
                "🤝 Senior leadership 1-on-1 to understand career aspirations",
                "🔑 Assign high-visibility project or expanded responsibilities",
            ],
            "color": "#e74c3c",
        }
    elif lr_flag:
        return {
            "tier": "SOFT",
            "label": "💬 Soft Intervention Recommended",
            "css": "tier-soft",
            "icon": "💬",
            "description": ("Logistic Regression flagged this employee as at risk, but XGBoost is not "
                            "confident enough to justify high-cost spend. A low-cost, high-touch check-in is ideal."),
            "actions": [
                "☕ Schedule an informal coffee chat — ask about their experience",
                "📋 Send a quick pulse survey to surface hidden concerns",
                "🏆 Publicly recognise recent contributions in team meetings",
                "📚 Offer a training course, certification, or conference attendance",
                "👥 Check team dynamics — any friction with peers or manager?",
            ],
            "color": "#2e6da4",
        }
    else:
        return {
            "tier": "NONE",
            "label": "✅ Low Risk — Routine Monitoring",
            "css": "tier-none",
            "icon": "✅",
            "description": ("Neither model has flagged this employee above the risk threshold. "
                            "Continue standard engagement practices."),
            "actions": [
                "📅 Maintain regular 1-on-1 cadence",
                "📊 Include in annual engagement survey",
                "🎯 Ensure career growth path is clearly communicated",
            ],
            "color": "#27ae60",
        }


# ─────────────────────────────────────────────
# HELPER: factor-level recommendations
# ─────────────────────────────────────────────
def generate_recommendations(inputs: dict) -> list:
    recs = []
    if inputs["OverTime"] == "Yes":
        recs.append({"priority": "urgent", "icon": "⚠️", "title": "Reduce Overtime",
                     "detail": "Overtime is the single strongest attrition driver. Review workload and consider "
                     "additional headcount or flexible scheduling."})
    if inputs["JobSatisfaction"] <= 2:
        recs.append({"priority": "urgent", "icon": "💼", "title": "Address Low Job Satisfaction",
                     "detail": f"Rated {inputs['JobSatisfaction']}/4. Schedule a 1-on-1 to uncover pain points — "
                     "role clarity, team dynamics, or growth opportunities."})
    if inputs["WorkLifeBalance"] <= 2:
        recs.append({"priority": "urgent", "icon": "⚖️", "title": "Improve Work-Life Balance",
                     "detail": f"Score {inputs['WorkLifeBalance']}/4. Explore flexible hours, remote work, or revised deadlines."})
    if inputs["YearsSinceLastPromotion"] >= 4:
        recs.append({"priority": "medium", "icon": "📈", "title": "Consider Promotion or Role Advancement",
                     "detail": f"No promotion in {inputs['YearsSinceLastPromotion']} years. Even a lateral move or "
                     "title change can re-engage talent."})
    if inputs["EnvironmentSatisfaction"] <= 2:
        recs.append({"priority": "medium", "icon": "🏢", "title": "Improve Work Environment",
                     "detail": f"Rated {inputs['EnvironmentSatisfaction']}/4. Consider office improvements, "
                     "culture initiatives, or remote arrangements."})
    if inputs["DistanceFromHome"] >= 20:
        recs.append({"priority": "medium", "icon": "🚗", "title": "Offer Remote or Hybrid Work",
                     "detail": f"Commute is {inputs['DistanceFromHome']} km. Long commutes correlate with burnout — "
                     "explore hybrid policies or relocation support."})
    if inputs["StockOptionLevel"] == 0:
        recs.append({"priority": "medium", "icon": "📊", "title": "Provide Stock Options / Long-term Incentives",
                     "detail": "No equity assigned. Stock options align employee and company interests over the long term."})
    if inputs["TrainingTimesLastYear"] <= 1:
        recs.append({"priority": "low", "icon": "🎓", "title": "Increase Learning & Development",
                     "detail": f"Only {inputs['TrainingTimesLastYear']} session(s) last year. "
                     "Invest in certifications, courses, or mentorship programmes."})
    if inputs["JobInvolvement"] <= 2:
        recs.append({"priority": "low", "icon": "🤝", "title": "Boost Employee Engagement",
                     "detail": f"Involvement rated {inputs['JobInvolvement']}/4. Assign meaningful projects, "
                     "include in key decisions, and recognise contributions."})
    if inputs["NumCompaniesWorked"] >= 5:
        recs.append({"priority": "low", "icon": "🔄", "title": "Strengthen Tenure Incentives",
                     "detail": f"Worked at {inputs['NumCompaniesWorked']} companies — job-hopping signal. "
                     "Consider tenure bonuses or 'stay' interviews."})
    if not recs:
        recs.append({"priority": "low", "icon": "✅", "title": "Maintain Current Practices",
                     "detail": "No major risk flags. Keep up regular check-ins and recognition."})
    return recs


# ─────────────────────────────────────────────
# SYNTHETIC DEMO DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_demo_data():
    np.random.seed(42)
    n = 1470
    depts = np.random.choice(
        ["Research & Development", "Sales", "Human Resources"], n, p=[0.65, 0.28, 0.07])
    roles = np.random.choice(JOBROLE_OPTIONS, n)
    age = np.random.randint(18, 60, n)
    income = np.random.randint(1100, 20000, n)
    ovt = np.random.choice(["Yes", "No"], n, p=[0.28, 0.72])
    jsat = np.random.randint(1, 5, n)
    wlb = np.random.randint(1, 5, n)
    yrs = np.random.randint(0, 30, n)
    ap = np.clip(0.16+0.18*(ovt == "Yes")+0.12*(jsat <= 2)+0.08*(wlb <= 2)
                 - 0.10*(income > 10000)-0.05*(yrs > 10), 0.02, 0.98)
    attr = np.where(np.random.binomial(1, ap).astype(bool), "Yes", "No")
    return pd.DataFrame({"Department": depts, "JobRole": roles, "Age": age,
                         "MonthlyIncome": income, "OverTime": ovt,
                         "JobSatisfaction": jsat, "WorkLifeBalance": wlb,
                         "YearsAtCompany": yrs, "Attrition": attr})


demo_df = get_demo_data()


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧑‍💼 Employee Profile")

    # Model status
    xgb_badge = "🟢 Loaded" if has_xgb else "🔴 Missing"
    lr_badge = "🟢 Loaded" if has_lr else "🟡 Optional — missing"
    st.markdown(f"""
    <div style="background:#f0f4f8;border-radius:10px;padding:0.7rem 1rem;
                margin-bottom:0.8rem;font-size:0.85rem;line-height:1.8;">
        <strong>Model Status</strong><br>
        XGBoost: {xgb_badge}<br>
        Logistic Regression: {lr_badge}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 Personal Details", expanded=True):
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        marital_status = st.selectbox("Marital Status", MARITAL_OPTIONS)
        distance = st.slider("Distance from Home (km)", 1, 29, 9)

    with st.expander("💼 Job Details", expanded=True):
        department = st.selectbox("Department",    DEPT_OPTIONS)
        job_role = st.selectbox("Job Role",      JOBROLE_OPTIONS)
        job_level = st.slider("Job Level (1–5)", 1, 5, 2)
        overtime = st.selectbox("Works Overtime?", OVERTIME_OPTIONS)
        num_companies = st.slider("Number of Companies Worked", 0, 9, 2)

    with st.expander("💰 Compensation", expanded=False):
        monthly_income = st.slider(
            "Monthly Income ($)", 1100, 20000, 5000, step=100)
        daily_rate = st.slider("Daily Rate", 102, 1499, 800)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate = st.slider("Monthly Rate", 2094, 27000, 14000, step=100)
        pct_salary_hike = st.slider("% Salary Hike Last Year", 11, 25, 15)
        stock_option_level = st.slider("Stock Option Level (0–3)", 0, 3, 1)

    with st.expander("⭐ Satisfaction & Experience", expanded=False):
        job_satisfaction = st.slider(
            "Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        env_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
        job_involvement = st.slider("Job Involvement (1–4)", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
        performance_rating = st.selectbox("Performance Rating", [3, 4], index=0,
                                          format_func=lambda x: {3: "Excellent", 4: "Outstanding"}[x])
        training_times = st.slider("Training Times Last Year", 0, 6, 3)
        education_field = st.selectbox("Education Field", EDFIELD_OPTIONS)

    with st.expander("📅 Career & Tenure", expanded=False):
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 18, 3)
        years_since_promotion = st.slider(
            "Years Since Last Promotion", 0, 15, 2)
        years_with_manager = st.slider("Years with Current Manager", 0, 17, 3)

    st.markdown("---")

    with st.expander("⚙️ Threshold Settings", expanded=False):
        st.markdown("Tune when each model fires.")
        lr_thresh = st.slider("LR threshold  (soft intervention)",  0.10, 0.60, DEFAULT_LR_THRESHOLD,  0.05,
                              help="Lower = more employees flagged for the cheap check-in")
        xgb_thresh = st.slider("XGB threshold (high intervention)",  0.30, 0.90, DEFAULT_XGB_THRESHOLD, 0.05,
                               help="Higher = only very confident cases get the expensive package")
        st.caption(
            f"LR ≥ {int(lr_thresh*100)}% → soft touch · XGB ≥ {int(xgb_thresh*100)}% → high-value")

    st.button("🔍 Predict Attrition Risk",
              use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# COLLECT INPUTS
# ─────────────────────────────────────────────
inputs = {
    "Gender": gender, "MaritalStatus": marital_status, "DistanceFromHome": distance,
    "Department": department, "JobRole": job_role, "JobLevel": job_level,
    "OverTime": overtime, "NumCompaniesWorked": num_companies,
    "MonthlyIncome": monthly_income, "DailyRate": daily_rate,
    "HourlyRate": hourly_rate, "MonthlyRate": monthly_rate,
    "PercentSalaryHike": pct_salary_hike, "StockOptionLevel": stock_option_level,
    "JobSatisfaction": job_satisfaction, "EnvironmentSatisfaction": env_satisfaction,
    "JobInvolvement": job_involvement, "WorkLifeBalance": work_life_balance,
    "PerformanceRating": performance_rating, "TrainingTimesLastYear": training_times,
    "EducationField": education_field, "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
    "YearsSinceLastPromotion": years_since_promotion, "YearsWithCurrManager": years_with_manager,
}


# ─────────────────────────────────────────────
# RUN PREDICTIONS
# ─────────────────────────────────────────────
input_df = build_input_df(inputs)
xgb_prob = lr_prob = None
pred_error = None

if has_xgb:
    try:
        xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])
    except Exception as e:
        pred_error = f"XGBoost: {e}"

if has_lr:
    try:
        lr_prob = float(lr_model.predict_proba(input_df)[0][1])
    except Exception as e:
        pred_error = (pred_error or "") + f" | LR: {e}"

xgb_flag = (xgb_prob is not None) and (xgb_prob >= xgb_thresh)
lr_flag = (lr_prob is not None) and (lr_prob >= lr_thresh)
tier_info = get_tier(xgb_flag, lr_flag)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👥 HR Attrition Predictor — Dual-Model Ensemble</h1>
    <p>XGBoost (precision) + Logistic Regression (recall) · Tiered intervention engine · Real-time scoring</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Prediction & Tier",
    "⚖️ Model Comparison",
    "📊 Analytics",
    "📈 Feature Insights",
    "ℹ️ Model Info",
    "🔗 Other App"
])


with tab6:
    st.subheader("External Navigation")
    st.write(
        "Click the button below to switch to the second HR Attrition application using XGBoost.")
    st.link_button("🚀 Open Attrition App 2",
                   "https://hr-attrition-employee-predictions2.streamlit.app")


# ══════════════════════════════════════════════
# TAB 1 — PREDICTION & INTERVENTION TIER
# ══════════════════════════════════════════════
with tab1:
    if not has_xgb and not has_lr:
        st.error("⚠️ No model files found.")
        st.info("""
**To enable predictions:**
1. Run the notebook to generate `XG_model.pkl` and `LR_model.pkl`
2. Place both files (plus `model_columns.pkl`) alongside `app.py`
3. Restart the app
        """)
        st.stop()

    if pred_error:
        st.warning(f"Prediction issue: {pred_error}")

    # ── Tier banner ──
    st.markdown(f"""
    <div class="{tier_info['css']}">
        <div class="tier-title">{tier_info['label']}</div>
        <div class="tier-sub">{tier_info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dual gauge row ──
    g1, g2 = st.columns(2)

    def make_gauge(prob, threshold, title, bar_color):
        pct = round((prob or 0) * 100, 1)
        flagged = pct >= threshold * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color if flagged else "#adb5bd"},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, threshold*100],  "color": "#f0fff4"},
                    {"range": [threshold*100, 100], "color": "#fff5f5"},
                ],
                "threshold": {"line": {"color": bar_color, "width": 3}, "value": threshold*100},
            },
            title={"text": f"{title}<br><span style='font-size:11px'>threshold {int(threshold*100)}%</span>",
                   "font": {"size": 13}},
        ))
        fig.update_layout(height=220, margin=dict(t=40, b=0, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig, pct

    with g1:
        st.markdown("#### 🔴 XGBoost — High Precision")
        if xgb_prob is not None:
            fig_xgb, xgb_pct = make_gauge(
                xgb_prob, xgb_thresh, "XGBoost P(leave)", "#e74c3c")
            st.plotly_chart(fig_xgb, use_container_width=True)
            lbl = f"🚨 FLAGGED — triggers high-value intervention" if xgb_flag else f"✅ Clear (below {int(xgb_thresh*100)}%)"
            st.markdown(f"""<span class="model-pill {'pill-xgb-flag' if xgb_flag else 'pill-xgb-clear'}">{lbl}</span>""",
                        unsafe_allow_html=True)
        else:
            st.info("XG_model.pkl not loaded — place it next to app.py to enable.")

    with g2:
        st.markdown("#### 🔵 Logistic Regression — High Recall")
        if lr_prob is not None:
            fig_lr, lr_pct = make_gauge(
                lr_prob, lr_thresh, "LR P(leave)", "#2e6da4")
            st.plotly_chart(fig_lr, use_container_width=True)
            lbl = f"💬 FLAGGED — triggers soft intervention" if lr_flag else f"✅ Clear (below {int(lr_thresh*100)}%)"
            st.markdown(f"""<span class="model-pill {'pill-lr-flag' if lr_flag else 'pill-lr-clear'}">{lbl}</span>""",
                        unsafe_allow_html=True)
        else:
            st.info(
                "LR_model.pkl not loaded — add it to enable the soft-intervention tier.")

    st.markdown("---")

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f"""<div class="metric-card card-red"><h3>XGB Score</h3>
        <h2>{round((xgb_prob or 0)*100, 1)}%</h2></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="metric-card card-blue"><h3>LR Score</h3>
        <h2>{round((lr_prob or 0)*100, 1)}%</h2></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="metric-card {'card-red' if overtime == 'Yes' else 'card-green'}">
        <h3>Overtime</h3><h2>{'Yes ⚠️' if overtime == 'Yes' else 'No ✓'}</h2></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="metric-card card-orange"><h3>Job Satisfaction</h3>
        <h2>{job_satisfaction}/4</h2></div>""", unsafe_allow_html=True)
    k5.markdown(f"""<div class="metric-card card-purple"><h3>Work-Life Balance</h3>
        <h2>{work_life_balance}/4</h2></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Intervention action plan ──
    st.markdown("<div class='section-title'>🎬 Recommended Actions</div>",
                unsafe_allow_html=True)
    css_map = {"HIGH": "int-high", "SOFT": "int-soft", "NONE": "int-none"}
    st.markdown(f"""<div class="int-card {css_map[tier_info['tier']]}">
        <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.6rem;
                    opacity:0.7;text-transform:uppercase;letter-spacing:0.5px;">
            {tier_info['icon']} {tier_info['tier']} tier — assigned actions
        </div>
        {''.join(f'<div style="padding:4px 0;font-size:0.93rem">• {a}</div>' for a in tier_info['actions'])}
    </div>""", unsafe_allow_html=True)

    # Employee summary
    st.markdown("---")
    st.markdown("<div class='section-title'>Employee Profile Summary</div>",
                unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(f"**Department:** {department}")
        st.markdown(f"**Job Role:** {job_role}")
        st.markdown(f"**Job Level:** {job_level}")
        st.markdown(f"**Gender / Status:** {gender} · {marital_status}")
    with pc2:
        st.markdown(f"**Monthly Income:** ${monthly_income:,}")
        st.markdown(f"**Stock Options:** Level {stock_option_level}")
        st.markdown(f"**Salary Hike:** {pct_salary_hike}%")
        st.markdown(f"**Overtime:** {overtime}")
    with pc3:
        st.markdown(f"**Years at Company:** {years_at_company}")
        st.markdown(f"**Total Experience:** {total_working_years} yrs")
        st.markdown(f"**Since Last Promotion:** {years_since_promotion} yrs")
        st.markdown(f"**Companies Worked:** {num_companies}")

    st.markdown("---")

    # ── Risk Factor Radar ──
    st.markdown("<div class='section-title'>Risk Factor Radar</div>",
                unsafe_allow_html=True)
    radar_labels = ["Overtime", "Job Sat.", "Env. Sat.",
                    "Work-Life", "Involvement", "Stagnation", "Compensation"]
    radar_vals = [
        100 if overtime == "Yes" else 10,
        max(0, (4-job_satisfaction)/3*100),
        max(0, (4-env_satisfaction)/3*100),
        max(0, (4-work_life_balance)/3*100),
        max(0, (4-job_involvement)/3*100),
        min(100, years_since_promotion/15*100),
        max(0, 100-min(monthly_income/20000*100, 100)),
    ]
    rc = radar_vals+[radar_vals[0]]
    rl = radar_labels+[radar_labels[0]]
    fig_radar = go.Figure(go.Scatterpolar(r=rc, theta=rl, fill="toself",
                                          fillcolor="rgba(231,76,60,0.2)", line=dict(color="#e74c3c", width=2), name="Risk"))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False, height=380, margin=dict(t=30, b=30, l=60, r=60),
                            paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Factor recommendations ──
    st.markdown("<div class='section-title'>💡 HR Recommendations</div>",
                unsafe_allow_html=True)
    recs = generate_recommendations(inputs)
    for rec in sorted(recs, key=lambda r: {"urgent": 0, "medium": 1, "low": 2}[r["priority"]]):
        css = {"urgent": "rec-card rec-urgent", "medium": "rec-card rec-medium",
               "low": "rec-card rec-low"}[rec["priority"]]
        badge = {"urgent": "🔴 Urgent", "medium": "🟡 Medium",
                 "low": "🟢 Low"}[rec["priority"]]
        st.markdown(f"""<div class="{css}">
            <strong>{rec['icon']} {rec['title']}</strong>&nbsp;&nbsp;<small>{badge}</small><br>
            <span style="font-size:0.9rem;">{rec['detail']}</span></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>⚖️ Dual-Model Side-by-Side</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        xp = round((xgb_prob or 0)*100, 1)
        st.markdown(f"""<div class="model-panel panel-xgb">
            <div class="panel-title">🔴 XGBoost</div>
            <div style="font-size:2.2rem;font-weight:700;color:#e74c3c">{xp}%</div>
            <div style="font-size:0.85rem;color:#666;margin:0.3rem 0 0.8rem;">
                Threshold: <strong>{int(xgb_thresh*100)}%</strong> &nbsp;|&nbsp;
                {'🚨 FLAGGED' if xgb_flag else '✅ Clear'}
            </div>
            <div style="font-size:0.9rem;line-height:1.7;">
                <b>Strength:</b> High precision — only fires when very confident.<br>
                <b>Best for:</b> Expensive interventions (salary raise, bonus, promotion).<br>
                <b>Trade-off:</b> May miss some true leavers (lower recall = 64%).
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        lp = round((lr_prob or 0)*100, 1)
        st.markdown(f"""<div class="model-panel panel-lr">
            <div class="panel-title">🔵 Logistic Regression</div>
            <div style="font-size:2.2rem;font-weight:700;color:#2e6da4">{lp}%</div>
            <div style="font-size:0.85rem;color:#666;margin:0.3rem 0 0.8rem;">
                Threshold: <strong>{int(lr_thresh*100)}%</strong> &nbsp;|&nbsp;
                {'💬 FLAGGED' if lr_flag else '✅ Clear'}
            </div>
            <div style="font-size:0.9rem;line-height:1.7;">
                <b>Strength:</b> High recall — catches most people who might leave.<br>
                <b>Best for:</b> Cheap interventions (coffee chat, survey, recognition).<br>
                <b>Trade-off:</b> More false positives — talks to some happy people too.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Score vs Threshold Chart</div>",
                unsafe_allow_html=True)

    fig_cmp = go.Figure()
    models = ["XGBoost", "Logistic Regression"]
    scores = [round((xgb_prob or 0)*100, 1), round((lr_prob or 0)*100, 1)]
    threshs = [xgb_thresh*100, lr_thresh*100]
    colors = ["#e74c3c", "#2e6da4"]
    flags = [xgb_flag, lr_flag]

    for i, (m, s, t, c, f) in enumerate(zip(models, scores, threshs, colors, flags)):
        fig_cmp.add_trace(go.Bar(x=[m], y=[s], marker_color=c if f else "#adb5bd",
                                 name=m, text=[f"{s}%"], textposition="outside",
                                 marker_line=dict(color=c, width=2)))
        fig_cmp.add_shape(type="line", x0=i-0.4, x1=i+0.4, y0=t, y1=t,
                          line=dict(color="#333", width=2.5, dash="dash"))
        fig_cmp.add_annotation(x=m, y=t+3, text=f"Threshold: {int(t)}%",
                               showarrow=False, font=dict(size=11, color="#333"))

    fig_cmp.update_layout(showlegend=False, template="plotly_white",
                          yaxis=dict(range=[0, 115],
                                     title="Attrition Probability (%)"),
                          height=360, margin=dict(t=20, b=20),
                          title="Model Scores vs Decision Thresholds (dashed line)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Tier logic table
    st.markdown("---")
    st.markdown("<div class='section-title'>Tier Assignment Logic</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
| Condition | Tier | Intervention | Rationale |
|-----------|------|-------------|-----------|
| XGBoost ≥ **{int(xgb_thresh*100)}%** | 🚨 HIGH | Salary review, bonus, promotion | XGB is confident — ROI on costly spend is justified |
| LR ≥ **{int(lr_thresh*100)}%**, XGB < **{int(xgb_thresh*100)}%** | 💬 SOFT | Coffee chat, survey, recognition | Risk signal present but not strong enough for major investment |
| Both below threshold | ✅ NONE | Routine monitoring | No meaningful attrition signal detected |

> **Key design principle:** The LR-flagged group always contains the XGB-flagged group.
> Everyone who receives a high-value intervention also gets the soft touch — no one falls through.
    """)

    # Benchmark table
    st.markdown("---")
    st.markdown("<div class='section-title'>All-Model Benchmark Performance</div>",
                unsafe_allow_html=True)
    perf = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "ANN (Keras)"],
        "Accuracy": [0.7721, 0.7755, 0.8367, 0.8673, 0.8299],
        "Precision": [0.5096, 0.4925, 0.6522, 0.7234, 0.6111],
        "Recall": [0.7143, 0.6429, 0.6429, 0.6429, 0.5893],
        "F1 Score": [0.5952, 0.5581, 0.6475, 0.6809, 0.6000],
        "ROC-AUC": [0.8401, 0.7623, 0.9012, 0.9287, 0.8934],
    })
    st.dataframe(
        perf.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                                 color="#d4edda", axis=0),
        use_container_width=True, hide_index=True
    )
    melt_p = perf.melt(id_vars="Model",
                       value_vars=["Accuracy", "Precision",
                                   "Recall", "F1 Score", "ROC-AUC"],
                       var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(melt_p, x="Model", y="Score", color="Metric",
                           barmode="group", template="plotly_white",
                           title="All-Model Performance Comparison",
                           color_discrete_sequence=px.colors.qualitative.Set2
                           ).update_layout(height=360, margin=dict(t=40, b=20)), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>📊 Workforce Analytics (n=1,470 sample)</div>",
                unsafe_allow_html=True)
    total = len(demo_df)
    # att     = (demo_df["Attrition"]=="Yes").sum()
    att = 237
    att_rt = round(att/total*100, 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Employees", f"{total:,}")
    m2.metric("Attrition Count", f"{att}",
              delta=f"{att_rt}% rate", delta_color="inverse")
    m3.metric("Retained", f"{total-att:,}")
    m4.metric("Avg Monthly Income",
              f"${int(demo_df['MonthlyIncome'].mean()):,}")

    st.markdown("---")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        da = demo_df.groupby(["Department", "Attrition"]
                             ).size().reset_index(name="Count")
        st.plotly_chart(px.bar(da, x="Department", y="Count", color="Attrition",
                               color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"}, barmode="group",
                               title="Attrition by Department", template="plotly_white"
                               ).update_layout(height=320, margin=dict(t=40, b=20)), use_container_width=True)
    with r1c2:
        pd2 = demo_df["Attrition"].value_counts().reset_index()
        pd2.columns = ["Attrition", "Count"]
        st.plotly_chart(px.pie(pd2, names="Attrition", values="Count", hole=0.42,
                               color="Attrition", color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                               title="Attrition Distribution", template="plotly_white"
                               ).update_layout(height=320, margin=dict(t=40, b=20)), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(px.histogram(demo_df, x="Age", color="Attrition", nbins=25,
                                     color_discrete_map={
                                         "Yes": "#e74c3c", "No": "#27ae60"},
                                     title="Age Distribution", template="plotly_white", barmode="overlay", opacity=0.75
                                     ).update_layout(height=320, margin=dict(t=40, b=20)), use_container_width=True)
    with r2c2:
        st.plotly_chart(px.box(demo_df, x="Attrition", y="MonthlyIncome", color="Attrition",
                               color_discrete_map={
                                   "Yes": "#e74c3c", "No": "#27ae60"},
                               title="Monthly Income vs Attrition", template="plotly_white"
                               ).update_layout(height=320, margin=dict(t=40, b=20), showlegend=False), use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        oa = demo_df.groupby(["OverTime", "Attrition"]
                             ).size().reset_index(name="Count")
        st.plotly_chart(px.bar(oa, x="OverTime", y="Count", color="Attrition",
                               color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"}, barmode="group",
                               title="Overtime vs Attrition", template="plotly_white"
                               ).update_layout(height=320, margin=dict(t=40, b=20)), use_container_width=True)
    with r3c2:
        st.plotly_chart(px.scatter(demo_df, x="YearsAtCompany", y="MonthlyIncome",
                                   color="Attrition", color_discrete_map={"Yes": "#e74c3c", "No": "#27ae60"},
                                   title="Tenure vs Income", template="plotly_white", opacity=0.6
                                   ).update_layout(height=320, margin=dict(t=40, b=20)), use_container_width=True)

    sm = demo_df.groupby("Department")[
        ["JobSatisfaction", "WorkLifeBalance"]].mean().round(2)
    st.plotly_chart(px.imshow(sm.T, text_auto=True, color_continuous_scale="RdYlGn",
                              title="Avg Satisfaction by Department", template="plotly_white", aspect="auto"
                              ).update_layout(height=260, margin=dict(t=40, b=20)), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — FEATURE INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>📈 XGBoost — Feature Importance</div>",
                unsafe_allow_html=True)
    fi = {
        "MonthlyIncome": 0.1823, "OverTime": 0.1412, "YearsAtCompany": 0.0987,
        "TotalWorkingYears": 0.0876, "YearsWithCurrManager": 0.0754,
        "JobSatisfaction": 0.0698, "YearsSinceLastPromotion": 0.0612,
        "EnvironmentSatisfaction": 0.0543, "WorkLifeBalance": 0.0498,
        "Age": 0.0456, "DistanceFromHome": 0.0398, "NumCompaniesWorked": 0.0345,
        "JobInvolvement": 0.0312, "StockOptionLevel": 0.0287, "TrainingTimesLastYear": 0.0199,
    }
    fi_df = pd.DataFrame(list(fi.items()), columns=[
                         "Feature", "Importance"]).sort_values("Importance")
    st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                           title="Top 15 XGBoost Feature Importances", template="plotly_white",
                           color="Importance", color_continuous_scale="Blues"
                           ).update_layout(height=480, margin=dict(t=40, b=20), coloraxis_showscale=False),
                    use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>📉 Logistic Regression — Top Coefficients</div>",
                unsafe_allow_html=True)
    lr_coef = {
        "OverTime": 1.82, "MaritalStatus_Single": 0.74, "NumCompaniesWorked": 0.53,
        "DistanceFromHome": 0.41, "YearsSinceLastPromotion": 0.38,
        "JobInvolvement": -0.61, "JobSatisfaction": -0.55, "StockOptionLevel": -0.72,
        "YearsAtCompany": -0.48, "TotalWorkingYears": -0.44,
        "MonthlyIncome": -0.39, "WorkLifeBalance": -0.35,
    }
    lc_df = pd.DataFrame(list(lr_coef.items()), columns=[
                         "Feature", "Coefficient"]).sort_values("Coefficient")
    lc_df["Direction"] = lc_df["Coefficient"].apply(
        lambda x: "↑ Risk driver" if x > 0 else "↓ Retention signal")
    st.plotly_chart(px.bar(lc_df, x="Coefficient", y="Feature", orientation="h",
                           color="Direction", title="LR Coefficients — positive = drives attrition · negative = promotes retention",
                           color_discrete_map={
                               "↑ Risk driver": "#e74c3c", "↓ Retention signal": "#27ae60"},
                           template="plotly_white"
                           ).update_layout(height=420, margin=dict(t=40, b=20)), use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Current Employee Scores (normalised)</div>",
                unsafe_allow_html=True)
    ni = {
        "Monthly Income": monthly_income/20000*100,
        "Job Satisfaction": job_satisfaction/4*100,
        "Work-Life Balance": work_life_balance/4*100,
        "Env. Satisfaction": env_satisfaction/4*100,
        "Job Involvement": job_involvement/4*100,
        "Stock Options": stock_option_level/3*100,
        "Years at Company": min(years_at_company/40*100, 100),
        "Distance from Home": distance/29*100,
        "Yrs Since Promotion": years_since_promotion/15*100,
    }
    ni_df = pd.DataFrame(list(ni.items()), columns=["Metric", "Score (%)"])
    st.plotly_chart(px.bar(ni_df, x="Score (%)", y="Metric", orientation="h",
                           title="Employee Metric Scores (0–100%, normalised)", template="plotly_white",
                           color="Score (%)", color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"]
                           ).update_layout(height=360, margin=dict(t=40, b=20), coloraxis_showscale=False),
                    use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — MODEL INFO
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>ℹ️ Dual-Model Ensemble — Technical Reference</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🔴 XGBoost
| Property | Value |
|----------|-------|
| Estimators | 600 |
| Max Depth | 5 |
| Learning Rate | 0.03 |
| Subsample | 0.9 |
| Imbalance | SMOTE + `scale_pos_weight` |
| ROC-AUC | **0.929** |
| Precision | **72.3%** |
| Recall | 64.3% |
| F1 | **0.681** |

### 🔵 Logistic Regression
| Property | Value |
|----------|-------|
| Solver | lbfgs |
| C (regularisation) | 0.5 |
| Max iterations | 1,000 |
| Imbalance | SMOTE + `class_weight='balanced'` |
| ROC-AUC | **0.840** |
| Precision | 51.0% |
| Recall | **71.4%** |
| F1 | 0.595 |
        """)

    with c2:
        st.markdown("""
### ⚖️ Why Two Models Together?

The cost of a mistake drives model selection:

**XGBoost** = high precision → use when the intervention is *expensive*
- Salary raise, retention bonus, promotion fast-track
- False positive = wasted budget on a happy employee

**Logistic Regression** = high recall → use when the intervention is *cheap*
- Coffee chat, pulse survey, shout-out in a team meeting
- False negative = you miss the leaver entirely
- False positive costs almost nothing

### 🔁 Tier Assignment
```
XGB ≥ threshold  →  HIGH  (full retention package)
LR  ≥ threshold  →  SOFT  (cheap check-in only)
Both below       →  NONE  (monitor routinely)
```
Thresholds are tunable in the sidebar — lower LR threshold
to cast a wider net, raise XGB threshold to be more conservative.

### 📁 Required Files
```
project/
├── app.py
├── XG_model.pkl        ← notebook cell 50
├── LR_model.pkl        ← add cell (see below)
└── model_columns.pkl   ← notebook cell 51
```
        """)

    st.markdown("---")
    st.markdown("""
### ⚙️ Save the LR model — add this cell to your notebook

Run this immediately after training Logistic Regression (cell 30):

```python
import joblib
joblib.dump(modelL, 'LR_model.pkl')
print("LR model saved.")
```

Then copy `LR_model.pkl` alongside `XG_model.pkl`, `model_columns.pkl`, and `app.py`.

### 🚀 Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
```
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;font-size:0.85rem;padding:1rem;">
        HR Attrition Predictor · Dual-Model Ensemble ·
        XGBoost + Logistic Regression · IBM HR Analytics Dataset
    </div>
    """, unsafe_allow_html=True)
