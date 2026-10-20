import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib  # noqa: E402

import train  # noqa: E402


def test_cross_validated_scores_are_high():
    r = train.main()
    assert r["Random Forest"]["accuracy"] > 0.9


def test_strong_company_scores_lower_than_weak_company():
    model = joblib.load(ROOT / "models/bankruptcy_model.joblib")
    strong = dict(industrial_risk=0, management_risk=0, financial_flexibility=1, credibility=1, competitiveness=1, operating_risk=0)
    weak = dict(industrial_risk=1, management_risk=1, financial_flexibility=0, credibility=0, competitiveness=0, operating_risk=1)
    col = list(model.classes_).index(1)
    p = [model.predict_proba(pd.DataFrame([row])[train.FEATURES])[0, col] for row in (strong, weak)]
    assert p[0] < 0.2 and p[1] > 0.8


def test_app_runs():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
