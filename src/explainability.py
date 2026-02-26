import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "is_new",
    "days_relationship",
    "score_band",
    "recency_days",
    "freq_90d",
    "monetary_90d",
    "weekend_ratio",
    "spend_std_90d",
    "app_interactions_30d",
    "engagement_rate",
    "multimodal_90d",
    "plan_active",
    "plan_price",
    "reactivations_12m",
    "inadimplencia",
]

CAT_FEATURES = ["score_band"]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True, choices=["churn_90d", "convert_30d"])
    ap.add_argument("--cutoff", default="2025-06-30")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["ref_date"])
    cutoff = pd.Timestamp(args.cutoff)
    train_df = df[df["ref_date"] <= cutoff].copy()
    test_df = df[df["ref_date"] > cutoff].copy()

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )

    model = Pipeline(
        [
            ("pre", preprocess),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=140,
                    max_depth=14,
                    min_samples_leaf=16,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, y_train = train_df[FEATURES], train_df[args.target]
    X_test, y_test = test_df[FEATURES], test_df[args.target]

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    # Permutation importance é calculada permutando as COLUNAS DE X_test (antes do OneHot),
    # então os nomes corretos são os nomes originais das features.
    pi = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    feat_names = list(X_test.columns)

    if len(feat_names) != len(pi.importances_mean):
        raise ValueError(
            f"Mismatch: feat_names={len(feat_names)} vs importances={len(pi.importances_mean)}. "
            f"Colunas X_test: {feat_names}"
        )

    imp = (
        pd.DataFrame(
            {
                "feature": feat_names,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    imp_path = os.path.join(args.outdir, f"permutation_importance_{args.target}.csv")
    imp.to_csv(imp_path, index=False)

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (mean)")
    plt.title(f"Top-15 Importâncias – {args.target} (AUC={auc:.3f})")
    fig_path = os.path.join(args.outdir, f"permutation_importance_{args.target}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    report = {
        "target": args.target,
        "auc": auc,
        "csv": os.path.basename(imp_path),
        "png": os.path.basename(fig_path),
    }
    with open(
        os.path.join(args.outdir, f"explain_{args.target}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("OK:", imp_path, fig_path)


if __name__ == "__main__":
    main()
