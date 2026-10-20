"""Streamlit app: score a company's bankruptcy risk from six qualitative risk factors."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "industrial_risk": "Industrial risk",
    "management_risk": "Management risk",
    "financial_flexibility": "Financial flexibility",
    "credibility": "Credibility",
    "competitiveness": "Competitiveness",
    "operating_risk": "Operating risk",
}
LEVELS = {0.0: "Low (0)", 0.5: "Medium (0.5)", 1.0: "High (1)"}

st.set_page_config(page_title="Bankruptcy Risk Assessment", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    path = ROOT / "models" / "bankruptcy_model.joblib"
    return joblib.load(path) if path.exists() else None


model = load_model()
if model is None:
    st.error("Model not found. Train it first:  `python src/train.py`")
    st.stop()

st.title("🏦 Bankruptcy Risk Assessment")
st.caption("Random Forest trained on 250 firms rated on six qualitative factors (0 = low, 0.5 = medium, 1 = high).")

left, middle, right = st.columns([1.1, 1.3, 1.2], gap="large")

with left:
    st.subheader("Risk profile")
    values = {key: st.select_slider(label, options=list(LEVELS), value=0.5, format_func=LEVELS.get)
              for key, label in FEATURES.items()}

X = pd.DataFrame([values])[list(FEATURES)]
p_bankrupt = float(model.predict_proba(X)[0, list(model.classes_).index(1)])

with middle:
    st.subheader("Assessment")
    st.metric("Estimated bankruptcy probability", f"{p_bankrupt:.0%}")
    if p_bankrupt < 0.25:
        st.success("Low risk: model leans clearly towards a sound company")
    elif p_bankrupt < 0.6:
        st.warning("Elevated risk: needs a closer look")
    else:
        st.error("High risk: model leans towards bankruptcy")
    st.progress(p_bankrupt)

with right:
    st.subheader("What drives the score")
    drivers = pd.DataFrame({"Factor": list(FEATURES.values()), "Model weight": model.feature_importances_})
    st.bar_chart(drivers.set_index("Factor").sort_values("Model weight"), horizontal=True)
    st.caption("Competitiveness, financial flexibility and credibility carry almost all of the weight in this dataset.")

st.divider()
st.caption("Educational model on a small public dataset (UCI Qualitative Bankruptcy). Not financial advice.")
