"""Step 1: train the hybrid models on the validated domain (Trial 16 excluded) and save them."""
import joblib
from data import build, FEATURES
from model import pso, Hybrid, metrics

d = build()
v = d[d.valid & (d.trial != 16)].reset_index(drop=True)   # 18 mixes = deployed domain
X = v[FEATURES].values
bundle = {"features": FEATURES, "models": {}, "ranges": {}}
for tgt in ["IST", "FST"]:
    y = v[tgt].values.astype(float)
    params, cv_rmse, _ = pso(X, y)                    # PSO-tuned ANN
    H = Hybrid(params).fit(X, y)                      # RF + GB + ANN -> Ridge
    m = metrics(y, H.meta.predict(H.oof))             # LOOCV performance
    print(f"{tgt}: LOOCV R2={m['R2']:.3f}, RMSE={m['RMSE']:.1f} min")
    bundle["models"][tgt] = H
bundle["ranges"] = {f: (float(v[f].min()), float(v[f].max())) for f in FEATURES}
joblib.dump(bundle, "aas_hybrid_model.joblib")
print("Saved aas_hybrid_model.joblib")
