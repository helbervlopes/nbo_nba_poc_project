import argparse, os, json
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, recall_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans

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
SEG_FEATURES = ["recency_days","freq_90d","monetary_90d","weekend_ratio","spend_std_90d","engagement_rate","multimodal_90d"]

def decide_nba(row, p_conv, p_churn, ltv_hat, policy):
    if row["inadimplencia"] == 1 or row["score_band"] == "<600":
        return ("NENHUMA", "Inelegível por risco (inadimplência/score)")
    if p_churn >= policy["churn_high"]:
        return ("RETENCAO_BENEFICIO", "Priorizar retenção (alto risco churn)")
    if (row["plan_active"]==0) and (ltv_hat >= policy["ltv_high"]) and (p_conv >= policy["conv_high"]):
        return ("UPSELL_PLANO", "Upsell (alto LTV e boa propensão)")
    if p_conv >= policy["conv_medium"]:
        return ("CASHBACK_LEVE", "Ação leve (propensão moderada)")
    return ("NENHUMA", "Baixa propensão; evitar contato")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cutoff", default="2025-06-30")
    ap.add_argument("--thr_churn", type=float, default=0.35)
    ap.add_argument("--thr_conv", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["ref_date"])
    cutoff = pd.Timestamp(args.cutoff)
    train_df = df[df["ref_date"] <= cutoff].copy()
    test_df  = df[df["ref_date"] > cutoff].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Split temporal gerou train/test vazio. Verifique cutoff e ref_date.")

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    X_train, X_test = train_df[FEATURES], test_df[FEATURES]

    # churn
    churn_model = Pipeline([
        ("pre", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=18,
            random_state=args.seed, n_jobs=-1
        ))
    ])
    y_train_ch, y_test_ch = train_df["churn_90d"], test_df["churn_90d"]
    churn_model.fit(X_train, y_train_ch)
    p_ch = churn_model.predict_proba(X_test)[:, 1]
    pred_ch = (p_ch >= args.thr_churn).astype(int)

    churn_metrics = {
        "auc": float(roc_auc_score(y_test_ch, p_ch)),
        "precision": float(precision_score(y_test_ch, pred_ch, zero_division=0)),
        "recall": float(recall_score(y_test_ch, pred_ch, zero_division=0)),
        "threshold": float(args.thr_churn),
    }

    # conversion
    conv_model = Pipeline([
        ("pre", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=140, max_depth=14, min_samples_leaf=16,
            random_state=args.seed, n_jobs=-1
        ))
    ])
    y_train_cv, y_test_cv = train_df["convert_30d"], test_df["convert_30d"]
    conv_model.fit(X_train, y_train_cv)
    p_cv = conv_model.predict_proba(X_test)[:, 1]
    pred_cv = (p_cv >= args.thr_conv).astype(int)

    conv_metrics = {
        "auc": float(roc_auc_score(y_test_cv, p_cv)),
        "precision": float(precision_score(y_test_cv, pred_cv, zero_division=0)),
        "recall": float(recall_score(y_test_cv, pred_cv, zero_division=0)),
        "threshold": float(args.thr_conv),
    }

    # LTV
    ltv_model = Pipeline([
        ("pre", preprocess),
        ("rf", RandomForestRegressor(
            n_estimators=140, max_depth=16, min_samples_leaf=14,
            random_state=args.seed, n_jobs=-1
        ))
    ])
    y_train_ltv, y_test_ltv = train_df["ltv_12m"], test_df["ltv_12m"]
    ltv_model.fit(X_train, y_train_ltv)
    ltv_hat = ltv_model.predict(X_test)

    ltv_metrics = {"rmse": float(mean_squared_error(y_test_ltv, ltv_hat, squared=False))}

    # segmentation
    scaler = StandardScaler()
    seg_X = scaler.fit_transform(train_df[SEG_FEATURES].to_numpy())
    km = MiniBatchKMeans(n_clusters=5, batch_size=512, max_iter=60, n_init=3, random_state=args.seed)
    km.fit(seg_X)
    seg_test = km.predict(scaler.transform(test_df[SEG_FEATURES].to_numpy()))

    # NBA policy
    policy = {"churn_high": 0.45, "ltv_high": 1200.0, "conv_high": 0.35, "conv_medium": 0.25}

    scored = test_df[["customer_id", "ref_date"] + FEATURES].copy()
    scored["p_churn_90d"] = p_ch
    scored["p_convert_30d"] = p_cv
    scored["ltv_12m_hat"] = ltv_hat
    scored["segmento_kmeans"] = seg_test

    decisions = [decide_nba(scored.iloc[i],
                            float(scored.iloc[i]["p_convert_30d"]),
                            float(scored.iloc[i]["p_churn_90d"]),
                            float(scored.iloc[i]["ltv_12m_hat"]),
                            policy)
                 for i in range(len(scored))]
    scored["nba_decision"] = [d[0] for d in decisions]
    scored["decision_reason"] = [d[1] for d in decisions]

    scored_path = os.path.join(args.outdir, "scored_with_nba.csv")
    scored.to_csv(scored_path, index=False)

    metrics = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "cutoff": str(cutoff.date()),
        "churn": churn_metrics,
        "conversion": conv_metrics,
        "ltv": ltv_metrics,
        "policy": policy,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("OK")
    print(" -", scored_path)
    print(" -", os.path.join(args.outdir, "metrics.json"))

if __name__ == "__main__":
    main()
