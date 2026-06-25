import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Initialize full viewport screen layout configurations
st.set_page_config(
    page_title="Vanguard Retention Platform", page_icon="🏦", layout="wide"
)

# Initialize Session State for tracking historic evaluation items dynamically
if "history" not in st.session_state:
    st.session_state.history = []

# --- PREMIUM CORPORATE BRANDING (CSS INJECTION) ---
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    /* Executive Background Color */
    .stApp {
        background: linear-gradient(180deg, #F4F7F5 0%, #EAEFE9 100%) !important;
        color: #0A3A20 !important;
    }
    
    /* Professional Deep Emerald Header Block */
    .custom-header {
        background-color: #0A3A20 !important; 
        padding: 24px !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        text-align: center !important;
        margin-bottom: 25px !important;
        box-shadow: 0 4px 12px rgba(10, 58, 32, 0.15) !important;
    }

    </style>
""",
    unsafe_allow_html=True,
)

# Path validation for underlying pickled model elements
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model_assets():
    rf = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    return rf, scaler, feature_cols


rf, scaler, feature_cols = load_model_assets()

# --- TOP PLATFORM HEADER BANNER ---
st.markdown(
    """
    <div class="custom-header">
        <h1 style='margin:0; font-size: 28px; letter-spacing: 0.5px; font-family: sans-serif; color: #FFFFFF;'>
            <i class="fa-solid fa-shield-halved" style="margin-right:10px; color:#A3E2C9;"></i> 
            VANGUARD CLIENT RETENTION PLATFORM
        </h1>
        <p style='margin:6px 0 0 0; font-size:14px; color:#A3E2C9; opacity:0.9; font-family: sans-serif;'>Automated Enterprise Risk Analytics Core</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- LEFT EXECUTIVE SIDEBAR NAVIGATION PANEL ---
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 20px;">
            <i class="fa-solid fa-wallet" style="font-size: 35px; color: #0A3A20;"></i>
            <h3 style="margin-top: 10px; color: #0A3A20; font-family: sans-serif;">Navigation Control</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    app_mode = st.selectbox(
        "Workspace Environment",
        [
            "Client Risk Audit",
            "Portfolio Distribution Analysis",
            "Model Diagnostic Panel",
        ],
    )

    st.markdown("---")
    st.write("**Active Model Attributes**")
    st.caption("Engine Core: Random Forest Classifier")
    st.caption("Target Scope: Retail Checking & Savings")

    # Sidebar Developer Credentials Footer Container
    st.markdown("---")
    st.markdown(
        """
        <div style='background-color:#0A3A20; padding:14px; border-radius:10px; text-align:center; color:white;'>
            <p style='margin:0; font-size:12px; opacity:.8; font-family: sans-serif;'>Developed By</p>
            <h4 style='margin:4px 0; color:#A3E2C9; font-family: sans-serif;'>Lama Alsubaie</h4>
            <p style='margin:0; font-size:11px; font-family: sans-serif;'>Data scientists </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# Shared chart layout template helper for unified look
def apply_light_tech_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0A3A20",
        title_font_color="#0A3A20",
        legend_font_color="#0A3A20",
        xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    )
    return fig


# ==========================================
# WORKSPACE 1: SINGLE CLIENT RISK AUDIT
# ==========================================
if app_mode == "Client Risk Audit":

    st.markdown(
        '### <i class="fa-solid fa-sliders" style="color:#0A3A20; margin-right:8px;"></i> Customer Parameters',
        unsafe_allow_html=True,
    )

    # Input Configuration Area (Top of Page)
    param_col1, param_col2, param_col3 = st.columns(3, gap="medium")

    with param_col1:
        with st.container(border=True):
            st.markdown("**Demographics**")
            age = st.slider("Customer Age", 18, 100, 38)
            gender = st.selectbox("Gender", ["Male", "Female"])
            geography = st.selectbox("Location", ["France", "Spain", "Germany"])
            credit_score = st.slider("Credit Score", 300, 900, 640)

    with param_col2:
        with st.container(border=True):
            st.markdown("**Financial Records**")
            balance = st.number_input(
                "Balance ($)", min_value=0.0, value=75000.0, step=2500.0
            )
            est_salary = st.number_input(
                "Estimated Salary ($)", min_value=0.0, value=115000.0, step=2500.0
            )
            st.markdown("<br><br><br>", unsafe_allow_html=True)

    with param_col3:
        with st.container(border=True):
            st.markdown("**Account Engagement**")
            tenure = st.slider("Tenure", 0, 10, 4)
            num_products = st.selectbox("Products", [1, 2, 3, 4])
            has_crcard = st.checkbox("Credit Card", value=True)
            is_active = st.checkbox("Active Member", value=True)

    # Trigger Generation Bar
    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])
    with btn_col2:
        analyze_btn = st.button(
            "Analyze Customer Account", use_container_width=True, type="primary"
        )

    has_crcard_int = 1 if has_crcard else 0
    is_active_int = 1 if is_active else 0

    raw_df = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_crcard_int,
                "IsActiveMember": is_active_int,
                "EstimatedSalary": est_salary,
            }
        ]
    )

    df_encoded = pd.get_dummies(raw_df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(df_encoded)

    # --- THREE-COLUMN VISUAL INFOGRAPHIC MATRIX BLOCK ---
    if analyze_btn:
        pred = rf.predict(X_scaled)[0]
        prob = rf.predict_proba(X_scaled)[0, 1]

        theme_accent_color = "#D62828" if prob >= 0.5 else "#0A3A20"
        risk_level = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
        confidence = max(prob, 1 - prob)
        risk_color = (
            "#D62828" if prob >= 0.65 else "#F4A261" if prob >= 0.35 else "#0A3A20"
        )

        st.markdown("---")

        # Split output canvas into 3 clean structural groupings
        col_risk, col_profile, col_metrics = st.columns([3, 3, 3], gap="large")

        # --------------------------------------------------
        # COLUMN 1: RISK ASSESSMENT PREDICTIONS
        # --------------------------------------------------
        with col_risk:
            st.markdown(
                "### <i class='fa-solid fa-gauge-high'></i> Risk Assessment",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        number={
                            "suffix": "%",
                            "font": {"color": "#111111", "size": 36},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1,
                                "tickcolor": "#444444",
                            },
                            "bar": {"color": theme_accent_color},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 1,
                            "bordercolor": "#CBD5E1",
                            "steps": [
                                {"range": [0, 35], "color": "#E2EFE7"},
                                {"range": [35, 65], "color": "#FFF9E6"},
                                {"range": [65, 100], "color": "#FCECEB"},
                            ],
                        },
                    )
                )
                fig.update_layout(
                    height=180,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics array stacked directly beneath the chart context
                st.markdown(
                    f"""
                    <div style="background:{risk_color}; color:white; padding:8px; border-radius:6px; text-align:center; font-weight:bold; margin-bottom:12px; font-size:13px; font-family: sans-serif;">
                        {risk_level.upper()} RISK CUSTOMER
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                k1, k2, k3 = st.columns(3)
                k1.metric("Churn Risk", f"{prob:.1%}")
                k2.metric("Confidence", f"{confidence:.1%}")
                k3.metric("Category", risk_level)

        # --------------------------------------------------
        # COLUMN 2: CUSTOMER PROFILE & RETENTION DIAGNOSIS
        # --------------------------------------------------
        with col_profile:
            st.markdown(
                "### <i class='fa-solid fa-user-tie'></i> Profile & Diagnosis",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                # Consolidated client metrics context array
                pk1, pk2 = st.columns(2)
                pk1.metric("Customer Age", age)
                pk1.metric("Credit Score", credit_score)
                pk2.metric("Branch Region", geography)
                pk2.metric("Active Products", num_products)
                st.markdown("<br>", unsafe_allow_html=True)

                if pred == 1:
                    st.markdown(
                        f"""
                        <div style="background-color: #FCECEB; border-left: 5px solid #D62828; padding: 12px; border-radius: 4px; font-family: sans-serif;">
                            <strong style="color: #D62828; font-size: 14px;"><i class="fa-solid fa-triangle-exclamation"></i> HIGH CHURN RISK DETECTED</strong>
                            <p style="margin: 4px 0 0 0; color: #333333; font-size: 12px;">The model predicts a high probability that this customer may leave the bank. Immediate retention actions are recommended.</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "**Recommended Countermeasures:**\n* Launch targeted loyalty bonuses\n* Initiate priority account outreach"
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #E2EFE7; border-left: 5px solid #0A3A20; padding: 12px; border-radius: 4px; font-family: sans-serif;">
                            <strong style="color: #0A3A20; font-size: 14px;"><i class="fa-solid fa-circle-check"></i> CUSTOMER RETENTION STABLE</strong>
                            <p style="margin: 4px 0 0 0; color: #333333; font-size: 12px;">Customer profile registers stable metrics with a high projected systemic alignment profile score.</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "**Recommended Operational Strategy:**\n* Standard communication schedules\n* Identify cross-selling paths"
                    )

        # --------------------------------------------------
        # COLUMN 3: EXPLAINABLE ATTRITION KEY DRIVERS & HISTORY
        # --------------------------------------------------
        with col_metrics:
            st.markdown(
                "### <i class='fa-solid fa-chart-bar'></i> Risk Diagnostics",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                feature_df = pd.DataFrame(
                    {
                        "Feature": ["Age", "Products", "Balance", "Activity", "Salary"],
                        "Importance": [0.24, 0.22, 0.14, 0.12, 0.10],
                    }
                ).sort_values(by="Importance", ascending=True)
                fig_audit_features = px.bar(
                    feature_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color_discrete_sequence=["#0A3A20"],
                )
                fig_audit_features.update_layout(
                    margin=dict(l=10, r=10, t=5, b=5),
                    height=140,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_audit_features, use_container_width=True)

                # Dynamic append and historical log loop rendering step
                current_record = {
                    "Age": age,
                    "Credit Score": credit_score,
                    "Risk": f"{prob:.1%}",
                }
                if (
                    not st.session_state.history
                    or st.session_state.history[-1] != current_record
                ):
                    st.session_state.history.append(current_record)

                history_df = pd.DataFrame(st.session_state.history[-3:])
                st.dataframe(history_df, use_container_width=True, height=120)

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 Adjust the parameters above and select 'Analyze Customer Account' to view the aligned summary dashboard."
        )

# ==========================================
# WORKSPACE 2: PORTFOLIO DISTRIBUTION ANALYSIS
# ==========================================
elif app_mode == "Portfolio Distribution Analysis":
    st.markdown(
        '### <i class="fa-solid fa-globe" style="color:#0A3A20; margin-right:8px;"></i> Corporate Portfolio Metrics Overview',
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        "Total Observed Portfolio Value", "$764.8M", "+$12.4M MoM", delta_color="normal"
    )
    kpi2.metric(
        "Average Customer Churn Rate",
        "20.3%",
        "-1.2% Target Deviation",
        delta_color="inverse",
    )
    kpi3.metric(
        "Monitored High Risk Accounts", "2,037 Client Profiles", "Requires Intervention"
    )

    st.markdown("---")
    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.markdown("**Branch Demographics vs Churn Volume**")
        mock_geo = pd.DataFrame(
            {
                "Branch Region": ["France", "Germany", "Spain"],
                "Active Clients": [5014, 2509, 2477],
                "Estimated Attrition": [810, 814, 413],
            }
        )
        fig_bar = px.bar(
            mock_geo,
            x="Branch Region",
            y=["Active Clients", "Estimated Attrition"],
            barmode="group",
            color_discrete_sequence=["#0A3A20", "#D62828"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_r:
        st.markdown("**Core Account Product Product Distributions**")
        mock_prod = pd.DataFrame(
            {
                "Products Held": [
                    "1 Product",
                    "2 Products",
                    "3 Products",
                    "4 Products",
                ],
                "Share Value": [50.8, 45.9, 2.7, 0.6],
            }
        )
        fig_pie = px.pie(
            mock_prod,
            values="Share Value",
            names="Products Held",
            color_discrete_sequence=["#0A3A20", "#1C6E3D", "#4EA371", "#D62828"],
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# WORKSPACE 3: MODEL DIAGNOSTIC PANEL
# ==========================================
elif app_mode == "Model Diagnostic Panel":
    st.markdown(
        '### <i class="fa-solid fa-gears" style="color:#0A3A20; margin-right:8px;"></i> Model Pipeline Quality Metrics',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Classification Accuracy", "86.8%")
    m2.metric("Precision (Class 1 Churn)", "86.5%")
    m3.metric("Recall (Sensitivity)", "87.0%")
    m4.metric("F1 Performance Index", "0.87")

    st.markdown("---")
    st.markdown("**Feature Importance Matrix (Full Pipeline Attributes)**")

    features = [
        "Age",
        "NumOfProducts",
        "Balance",
        "IsActiveMember",
        "EstimatedSalary",
        "CreditScore",
        "Geography_Germany",
        "Tenure",
    ]
    weights = [0.24, 0.22, 0.14, 0.12, 0.10, 0.09, 0.06, 0.03]
    df_features = pd.DataFrame(
        {"Risk Factor / Feature": features, "Relative Importance weight": weights}
    ).sort_values(by="Relative Importance weight", ascending=True)

    fig_features = px.bar(
        df_features,
        x="Relative Importance weight",
        y="Risk Factor / Feature",
        orientation="h",
        color_discrete_sequence=["#0A3A20"],
    )
    fig_features.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_features, use_container_width=True)
