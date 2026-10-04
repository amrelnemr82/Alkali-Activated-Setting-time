"""Step 2: Streamlit app - forward prediction and route-constrained inverse mix design."""
import joblib, numpy as np, pandas as pd, streamlit as st
from model import Hybrid, ScaledANN   # needed so joblib can load the models

st.set_page_config(page_title="AAS Setting Time Predictor", layout="centered")
B = joblib.load("aas_hybrid_model.joblib")
F, HI, HF = B["features"], B["models"]["IST"], B["models"]["FST"]

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

st.title("Setting time of alkali-activated slag")
st.caption("PSO-tuned RF + GB + ANN stack with Ridge meta-learner. Valid only within the ranges of the 18 training mixes "
           "(one GGBS, laboratory temperature). Final setting: nested LOOCV R2 = 0.94; initial setting is less reliable.")
tab1, tab2 = st.tabs(["Predict setting time", "Find a mix for a target"])

with tab1:
    route = st.selectbox("Activator route", list(ROUTES), key="r1")
    b = ROUTES[route]; x = {}
    for f in F:
        lo, hi = b[f]
        if lo == hi: x[f] = lo                     # fixed by the route (e.g. has_Na2SiO3 = 0 or 1)
        else: x[f] = st.slider(LABELS.get(f, f), float(lo), float(hi), float((lo+hi)/2), step=0.001, format="%.3f")
    ist, fst = predict([x[f] for f in F])
    c1, c2 = st.columns(2)
    c1.metric("Initial setting time", f"{ist[0]:.0f} min"); c2.metric("Final setting time", f"{fst[0]:.0f} min")
    if fst[0] <= ist[0]: st.warning("Predicted FST <= IST: combination is outside the reliable domain.")

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
            pi, pf = predict(C)
            s = np.abs(pi-tI)/tI + np.abs(pf-tF)/tF
            s[pf <= pi] = np.inf                                      # physically inconsistent
            inside = (pi>=ist_lo)&(pi<=ist_hi)&(pf>=fst_lo)&(pf<=fst_hi)
            k = int(np.argmin(s))
            rows.append({"Route": name, **{f: round(C[k, i], 3) for i, f in enumerate(F)},
                         "IST pred": round(pi[k],1), "FST pred": round(pf[k],1), "Fit score": round(s[k],3),
                         "In window": bool(inside[k]), "Feasible candidates": int(inside.sum())})
        df = pd.DataFrame(rows).sort_values("Fit score")
        st.dataframe(df, hide_index=True)
        if not df["In window"].any(): st.error("No route reaches this window within the validated domain.")
        st.info("Lower fit score = closer to the target. Proposed mixes must be verified by laboratory testing.")
