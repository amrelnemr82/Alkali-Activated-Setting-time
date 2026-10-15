# Alkali-Activated-Setting-time

Explainable machine learning for the initial (IST) and final (FST) setting times of alkali-activated slag (AAS) pastes.
This repository accompanies the article *"Controlling the setting of alkali-activated slag: experimental activator study and explainable machine-learning design tool"* (A. ElNemr, A. Hassan, L. Goebel, C. Koenke).

**Web application:** https://alkali-activated-setting-time-ehqy3e5j6vvdxzjlhn8cji.streamlit.app/

## Contents

| Path | Purpose |
|---|---|
| `app.py` | Streamlit web application (Appendix A): forward prediction with 80/90 % multiplicative prediction intervals and an applicability-domain check, route-constrained inverse mix design, and a model card |
| `aas_hybrid_model.joblib` | Deployed 18-mix hybrid models (RF + GB + PSO-tuned ANN, Ridge meta-learner) for IST and FST |
| `domain.json` | Prediction-interval factors and applicability-domain statistics used by the app |
| `data.py` | Experimental dataset (Table 2 of the article) and feature engineering |
| `model.py` | PSO tuning, hybrid stacked model and metrics |
| `train_export.py` | Retrains the deployed models from `data.py` and rewrites the model bundle |
| `analysis/` | Scripts and result files for every analysis and figure in the article |

## Analysis scripts (`analysis/`)

Run from inside `analysis/`; figures are written to `analysis/figures/`.

| Script | Article section / figure |
|---|---|
| `run1.py <all19|excl16> <IST|FST>` | Standard LOOCV of base learners and hybrid (Table 5, Figs. 7–9) |
| `nest_chunk.py` | Fully nested LOOCV of the hybrid (Table 5, Fig. 9) |
| `bench.py <all19|excl16> <IST|FST>` | 15 benchmark learners and the log-linear equation (Tables 6–7, Figs. 10–11) |
| `yrand.py`, `fig_yrand.py` | y-randomisation (Fig. 12) |
| `xai.py`, `figs2.py` | Kernel SHAP, interactions, ICE/PDP (Figs. 13–17) and benchmark figures |
| `unc.py` | Conformal multiplicative prediction intervals and applicability domain (Section 4.10) |
| `ext2.py`, `figs3_v2.py` | Transferability to Choi and Lee (2019) with one-mix recalibration (Figs. 18–19) |
| `figs3.py`, `inv.py` | Design charts and inverse mix design (Figs. 20–21, Table 8) |
| `fig_corr.py`, `fig_kde.py`, `fig_exp.py`, `fig_wf.py`, `fig6.py` | Data distributions, experimental trends, workflow (Figs. 2, 3, 4, 6) |

Precomputed results (`*.json`, `*.csv`) and fitted hybrids (`H_*.pkl`) are included so that every figure can be regenerated without re-running the slow nested validation.

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Python 3.12, scikit-learn 1.8.0, SHAP 0.52.0, NumPy 2.4, pandas 3.0, XGBoost, LightGBM, CatBoost.

## Citation

Please cite the article and the archived software (Zenodo, https://doi.org/10.5281/zenodo.23133386).
