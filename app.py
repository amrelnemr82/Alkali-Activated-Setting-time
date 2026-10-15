"""Streamlit app: forward prediction with prediction intervals and an applicability-domain check,
plus route-constrained inverse mix design for alkali-activated slag (AAS) setting times."""
import json, joblib, numpy as np, pandas as pd, streamlit as st
from model import Hybrid, ScaledANN   # needed so joblib can load the models

st.set_page_config(page_title="AAS Setting Time Predictor", layout="wide")
B = joblib.load("aas_hybrid_model.joblib")
F, HI, HF = B["features"], B["models"]["IST"], B["models"]["FST"]
D = json.load(open("domain.json"))                       # interval factors + applicability domain
MU, SD, TZ, THR = np.array(D["mu"]), np.array(D["sd"]), np.array(D["train_z"]), D["threshold"]

# Activator routes: bounds observed in the training data (validity domain)
ROUTES = {
 "A: NaOH solution only": dict(Na2O_pct=(.136,.242), MS=(0,0), LS=(.26,.37), Alk_Bi=(.55,.65),
     has_Na2SiO3=(0,0), water_binder_ratio=(0,0), Na2SiO3_binder_ratio=(0,0)),
 "B: NaOH + Na2SiO3 (molar NaOH)": dict(Na2O_pct=(.107,.277), MS=(.3,1.3), LS=(.30,.47), Alk_Bi=(.65,1.1),
     has_Na2SiO3=(1,1), water_binder_ratio=(0,0), Na2SiO3_binder_ratio=(.23,.83)),
 "C: 48% Na2O NaOH + Na2SiO3 + extra water": dict(Na2O_pct=(.06,.08), MS=(1.0,1.1), LS=(.37,.42), Alk_Bi=(.57,.64),
     has_Na2SiO3=(1,1), water_binder_ratio=(.216,.283), Na2SiO3_binder_ratio=(.22,.30)),
}
LABELS = {"Na2O_pct":"Na2O dosage (fraction of slag)","MS":"Silicate modulus Ms","LS":"Water/solids L/S",
          "Alk_Bi":"Alkaline solution/slag","water_binder_ratio":"Extra water/slag","Na2SiO3_binder_ratio":"Na2SiO3 solution/slag"}

def predict(X):
    X = np.atleast_2d(X); return HI.predict(X), HF.predict(X)

def ad_distance(X):
    """Mean standardised Euclidean distance to the k = 3 nearest training mixes."""
    Z = (np.atleast_2d(X) - MU) / SD
    d = np.sqrt(((Z[:, None, :] - TZ[None, :, :])**2).sum(-1))
    return np.sort(d, 1)[:, :D["k"]].mean(1)

def interval(p, target, level):
    f = D["interval"][target][level]; p = max(p, 1.0); return p / f, p * f

st.title("Setting time of alkali-activated slag")
st.caption("Stacked RF + GB + PSO-tuned ANN with a Ridge meta-learner, trained on 18 laboratory mixes "
           "(one GGBS, 20 ± 2 °C). Final setting: nested LOOCV R² = 0.94. Initial setting is indicative only "
           "(nested R² = 0.19). Interval bounds are the prediction divided and multiplied by a factor calibrated on nested LOOCV residuals.")
tab1, tab2, tab3 = st.tabs(["Predict setting time", "Find a mix for a target", "Model card"])

with tab1:
    route = st.selectbox("Activator route", list(ROUTES), key="r1")
    level = st.radio("Prediction interval", ["80", "90"], horizontal=True, format_func=lambda s: f"{s} %")
    b = ROUTES[route]; x = {}
    for f in F:
        lo, hi = b[f]
        if lo == hi: x[f] = lo                     # fixed by the route (e.g. has_Na2SiO3 = 0 or 1)
        else: x[f] = st.slider(LABELS.get(f, f), float(lo), float(hi), float((lo+hi)/2), step=0.001, format="%.3f")
    xv = [x[f] for f in F]; ist, fst = predict(xv); dist = float(ad_distance(xv)[0])
    c1, c2, c3 = st.columns(3)
    li, hi_ = interval(ist[0], "IST", level); lf, hf = interval(fst[0], "FST", level)
    c1.metric("Initial setting time", f"{ist[0]:.0f} min"); c1.caption(f"{level} % interval: {li:.0f}–{hi_:.0f} min")
    c2.metric("Final setting time", f"{fst[0]:.0f} min");   c2.caption(f"{level} % interval: {lf:.0f}–{hf:.0f} min")
    c3.metric("Domain distance", f"{dist:.2f}", help=f"Inside the applicability domain if ≤ {THR:.2f}")
    c3.caption("Inside domain" if dist <= THR else "Outside domain: extrapolation")
    if dist > THR: st.error("This combination is far from every training mix; the prediction is an extrapolation.")
    if fst[0] <= ist[0] or ist[0] < 1: st.warning("Predicted FST ≤ IST or a non-positive time: this combination lies "
                                                  "outside the jointly observed domain.")

with tab2:
    c1, c2 = st.columns(2)
    ist_lo = c1.number_input("Min initial setting (min)", 1.0, 300.0, 20.0); ist_hi = c2.number_input("Max initial setting (min)", 1.0, 300.0, 30.0)
    fst_lo = c1.number_input("Min final setting (min)", 1.0, 400.0, 45.0); fst_hi = c2.number_input("Max final setting (min)", 1.0, 400.0, 60.0)
    n = st.select_slider("Candidates per route", [2000, 5000, 10000], 5000)
    if st.button("Search for feasible mixes"):
        tI, tF = (ist_lo+ist_hi)/2, (fst_lo+fst_hi)/2
        rng, rows = np.random.default_rng(0), []
        for name, b in ROUTES.items():
            C = np.column_stack([rng.uniform(*b[f], n) for f in F])
            pi, pf = predict(C); dd = ad_distance(C)
            s = np.abs(pi-tI)/tI + np.abs(pf-tF)/tF
            s[(pf <= pi) | (pi < 1) | (dd > THR)] = np.inf          # inconsistent or outside the domain
            inside = (pi>=ist_lo)&(pi<=ist_hi)&(pf>=fst_lo)&(pf<=fst_hi)&np.isfinite(s)
            k = int(np.argmin(s))
            rows.append({"Route": name, **{f: round(C[k, i], 3) for i, f in enumerate(F)},
                         "IST pred": round(pi[k],1), "FST pred": round(pf[k],1),
                         "FST 80 % interval": "%.0f–%.0f" % interval(pf[k], "FST", "80"),
                         "Domain distance": round(float(dd[k]),2), "Fit score": round(s[k],3),
                         "In window": bool(inside[k]), "Feasible candidates": int(inside.sum())})
        df = pd.DataFrame(rows).sort_values("Fit score")
        first = ["Route", "IST pred", "FST pred", "FST 80 % interval", "Domain distance", "Fit score", "In window", "Feasible candidates"]
        df = df[first + [c for c in df.columns if c not in first]]   # results first, mix features after
        st.dataframe(df, hide_index=True)
        if not df["In window"].any(): st.error("No route reaches this window within the validated domain.")
        st.info("Lower fit score = closer to the target. Proposed mixes must be verified by laboratory testing.")

with tab3:
    st.markdown(f"""
**Training data.** 18 alkali-activated slag pastes (Trial 16 withheld as a response outlier), one German GGBS,
Vicat needle, 20 ± 2 °C. Three activator routes (A, B, C) as listed in the selector.

**Validation (nested leave-one-out).** FST: R² = 0.94, RMSE = 10.9 min. IST: R² = 0.19, RMSE = 17.8 min.

**Prediction intervals.** Lower bound = prediction divided by a factor; upper bound = prediction multiplied by the same factor
(factors from nested residuals on the log scale). FST factor {D['interval']['FST']['80']:.2f} (80 %) and {D['interval']['FST']['90']:.2f} (90 %);
IST factor {D['interval']['IST']['80']:.2f} (80 %) and {D['interval']['IST']['90']:.2f} (90 %). Example: a predicted FST of 60 min has 80 % bounds of
{60/D['interval']['FST']['80']:.0f} and {60*D['interval']['FST']['80']:.0f} min.

**Applicability domain.** Mean standardised distance to the three nearest training mixes; limit = {THR:.2f}
(upper Tukey fence, Q3 + 1.5 IQR, of the leave-one-out distances of the training mixes). A mix inside this limit is *not* guaranteed to be
reliable: a different slag or temperature changes setting without changing these inputs.

**Transfer to another slag.** Raw predictions for an independent dataset were biased high, but the ranking was
preserved. Measuring one reference mix with the new slag and shifting all predictions by the log-ratio
(measured/predicted) reduced the FST RMSE from 126 to 12 min in our check (six mixes with 6–8 % Na₂O).
""")
