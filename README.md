# Bankruptcy Prevention

Score a company's bankruptcy risk from six qualitative risk factors, with a Random Forest and a small Streamlit app.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/Roshanz/Bankruptcy_Prevention)

## Data

250 firms (107 bankrupt, 143 not) from the UCI *Qualitative Bankruptcy* data. Each firm is rated 0, 0.5 or 1 on industrial risk, management risk, financial flexibility, credibility, competitiveness and operating risk.

Only **103 of the 250 rows are unique**; the other 147 are exact repeats, because six three-level factors give few possible combinations. That matters for evaluation, so scores below use the unique rows only with 5-fold stratified cross-validation. A plain random split would put the same row in both train and test and overstate the result.

## Results (5-fold CV on unique rows)

| Model | Accuracy | Recall (bankrupt) | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.98 | 0.92 | 1.00 |
| Decision Tree (depth 4) | 0.98 | 0.96 | 0.97 |
| **Random Forest (200 trees)** | **0.99** | **0.96** | **1.00** |

The classes are almost perfectly separable on these factors, so every model scores high. Do not read this as real-world accuracy: it is a small, clean, expert-rated teaching dataset.

**What drives the prediction:** competitiveness carries 52% of the forest's weight, financial flexibility 21%, credibility 20%. Industrial, management and operating risk together carry about 7%.

## Run it

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

Tests: `pip install -r requirements-dev.txt && pytest`

## Layout

```
data/bankruptcy.csv     the 250 rows (converted from the original semicolon-separated Excel sheet)
src/train.py            cross-validated comparison, final model and metrics.json
app/streamlit_app.py    the web app
tests/                  tests for scores, direction of risk and the app
```

## Limitations

- Tiny dataset of qualitative ratings, not financial statements; real bankruptcy models use ratios (leverage, liquidity, profitability).
- The verdict thresholds in the app (25% and 60%) are chosen for readability, not calibrated.
- Educational use only. Not financial advice.
