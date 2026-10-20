"""Train the bankruptcy-risk classifier and report cross-validated scores.

    python src/train.py

The dataset (250 firms, six expert-rated risk factors, each 0 / 0.5 / 1) holds many identical rows
(147 of 250 are duplicates), so scores are computed on de-duplicated rows with stratified cross-validation;
a plain random split would put the same row in both train and test and flatter the result.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["industrial_risk", "management_risk", "financial_flexibility", "credibility", "competitiveness", "operating_risk"]


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/bankruptcy.csv")
    df["bankrupt"] = (df["class"] == "bankruptcy").astype(int)
    return df


def main() -> dict:
    df = load()
    unique = df.drop_duplicates(subset=FEATURES + ["bankrupt"])
    X, y = unique[FEATURES], unique["bankrupt"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    results = {"rows": len(df), "duplicate_rows": int(len(df) - len(unique)), "unique_rows": len(unique)}
    for name, model in models.items():
        s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
        results[name] = {k: round(float(s[f"test_{k}"].mean()), 3) for k in ("accuracy", "recall", "roc_auc")}
        print(name, results[name])

    final = RandomForestClassifier(n_estimators=200, random_state=42).fit(df[FEATURES], df["bankrupt"])
    results["importance"] = dict(zip(FEATURES, final.feature_importances_.round(3).tolist()))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models/bankruptcy_model.joblib")
    (ROOT / "models/metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
