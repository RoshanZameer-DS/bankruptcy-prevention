import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stability Command Center", page_icon="🏦", layout="wide")

# --- PREMIUM CSS FOR PRESENTATION ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .decision-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        color: white;
    }
    .healthy { background-color: #28a745; border: 4px solid #1e7e34; }
    .warning { background-color: #ffc107; color: #212529; border: 4px solid #d39e00; }
    .critical { background-color: #dc3545; border: 4px solid #bd2130; }
    .reason-box {
        padding: 15px;
        background-color: white;
        border-left: 6px solid #008080;
        margin-bottom: 12px;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stSlider { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists('bankruptcy_model.pkl'):
        return joblib.load('bankruptcy_model.pkl')
    return None

model = load_model()

# --- HEADER ---
st.title("🏦 Business Stability Command Center")
st.markdown("### Final Decision Support System for Bankruptcy Prevention")
st.divider()

# --- 3-COLUMN PRESENTATION LAYOUT ---
col_input, col_decision, col_reason = st.columns([1.1, 1.4, 1.2], gap="large")

with col_input:
    st.subheader("🔍 Risk Profile Inputs")
    # Using sliders for risk factors
    ind = st.slider("Industrial Risk", 0.0, 1.0, 0.5, 0.5)
    man = st.slider("Management Risk", 0.0, 1.0, 0.5, 0.5)
    fin = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, 0.5)
    cre = st.slider("Credibility", 0.0, 1.0, 0.5, 0.5)
    com = st.slider("Competitiveness", 0.0, 1.0, 0.5, 0.5)
    ope = st.slider("Operating Risk", 0.0, 1.0, 0.5, 0.5)
    
    input_data = pd.DataFrame([[ind, man, fin, cre, com, ope]], 
                              columns=['industrial_risk', 'management_risk', 'financial_flexibility', 
                                       'credibility', 'competitiveness', 'operating_risk'])

# --- LOGIC & CALCULATION ---
# Bankruptcy risk logic: higher values in some features mean MORE risk, 
# while higher values in others (flexibility, credibility, competitiveness) mean LESS risk.
# Formula to normalize everything to a 'Risk Score' (0 = Good, 1 = Critical)
total_risk_score = (ind + man + (1-fin) + (1-cre) + (1-com) + ope) / 6

with col_decision:
    st.subheader("🎯 Final Decision")
    if model:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        # Determine Label based on the Risk Score
        if total_risk_score <= 0.35:
            st.markdown('<div class="decision-card healthy">FINANCIALLY HEALTHY</div>', unsafe_allow_html=True)
            st.markdown("## Recommendation: **GOOD CANDIDATE**")
        elif total_risk_score <= 0.65:
            st.markdown('<div class="decision-card warning">AT RISK</div>', unsafe_allow_html=True)
            st.markdown("## Recommendation: **NEEDS ATTENTION**")
        else:
            st.markdown('<div class="decision-card critical">CRITICAL RISK</div>', unsafe_allow_html=True)
            st.markdown("## Recommendation: **NOT RECOMMENDED**")
            
        st.write(f"**AI Confidence Level:** {max(proba)*100:.1f}%")
    else:
        st.info("Loading Predictive Engine...")

with col_reason:
    st.subheader("💡 Business Insights")
    
    # Automated reasoning based on input levels
    insights = []
    
    # Strong Points
    if com >= 0.75: insights.append(("🟢", "Market Leader: Strong Competitiveness"))
    if cre >= 0.75: insights.append(("🟢", "High Credibility: Market trust is high"))
    if fin >= 0.75: insights.append(("🟢", "Financial Strength: High Flexibility"))
    
    # Weak Points
    if man >= 0.75: insights.append(("🔴", "Governance Alert: High Management Risk"))
    if ope >= 0.75: insights.append(("🔴", "Efficiency Alert: High Operating Risk"))
    if total_risk_score > 0.6: insights.append(("🔴", "Combined Risk: Multiple factors are critical"))

    if insights:
        for icon, text in insights:
            st.markdown(f'<div class="reason-box">{icon} {text}</div>', unsafe_allow_html=True)
    else:
        st.write("The company shows balanced risk indicators. No critical outliers detected.")

st.divider()
st.caption("This professional dashboard is designed for executive decision-making and project presentation.")
