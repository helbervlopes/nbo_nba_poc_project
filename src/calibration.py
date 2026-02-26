import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEATURES = [
    "is_new","days_relationship","score_band",
    "recency_days","freq_90d","monetary_90d",
    "weekend_ratio","spend_std_90d",
    "app_interactions_30d","engagement_rate",
    "multimodal_90d","plan_active","plan_price",
    "reactivations_12m","inadimplencia",
]
CAT_FEATURES = ["score_band"]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True, choices=["churn_90d","convert_30d"])
    ap.add_argument("--cutoff", default="2025-06-30")
    ap.add_argument("--method", default="sigmoid", choices=["sigmoid","isotonic"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["ref_date"])
    cutoff = pd.Timestamp(args.cutoff)

    train_df = df[df["ref_date"] <= cutoff].copy().sort_values("ref_date")
    test_df  = df[df["ref_date"] > cutoff].copy()

    split_idx = int(len(train_df) * 0.8)
    tr = train_df.iloc[:split_idx].copy()
    cal = train_df.iloc[split_idx:].copy()

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    base = Pipeline([
        ("pre", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=16,
            random_state=42, n_jobs=-1
        ))
    ])

    X_tr, y_tr = tr[FEATURES], tr[args.target]
    X_cal, y_cal = cal[FEATURES], cal[args.target]
    X_te, y_te = test_df[FEATURES], test_df[args.target]

    # 1) Treina modelo base no conjunto "tr"
    base.fit(X_tr, y_tr)

    # 2) Calibra em "cal" usando o estimador já treinado (cv="prefit")
    # OBS: use 'estimator=' (novo nome). Isso evita warning e impede estimator=None.
    calib = CalibratedClassifierCV(estimator=base, method=args.method, cv="prefit")
    calib.fit(X_cal, y_cal)

    p_base = base.predict_proba(X_te)[:,1]
    p_cal  = calib.predict_proba(X_te)[:,1]

    report = {
        "target": args.target,
        "method": args.method,
        "auc_base": float(roc_auc_score(y_te, p_base)),
        "auc_cal": float(roc_auc_score(y_te, p_cal)),
        "brier_base": float(brier_score_loss(y_te, p_base)),
        "brier_cal": float(brier_score_loss(y_te, p_cal)),
        "n_train": int(len(tr)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test_df)),
    }

    frac_pos_base, mean_pred_base = calibration_curve(y_te, p_base, n_bins=10)
    frac_pos_cal,  mean_pred_cal  = calibration_curve(y_te, p_cal,  n_bins=10)

    plt.figure(figsize=(7,6))
    plt.plot(mean_pred_base, frac_pos_base, marker="o", label="Base (RF)")
    plt.plot(mean_pred_cal,  frac_pos_cal,  marker="o", label=f"Calibrado ({args.method})")
    plt.plot([0,1],[0,1], linestyle="--", label="Ideal")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Fração de positivos")
    plt.title(f"Curva de calibração – {args.target}")
    plt.legend()
    fig_path = os.path.join(args.outdir, f"calibration_{args.target}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    rep_path = os.path.join(args.outdir, f"calibration_{args.target}.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("OK:", rep_path, fig_path)

if __name__ == "__main__":
    main()
