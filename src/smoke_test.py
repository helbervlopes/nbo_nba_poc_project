import os, tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from subprocess import check_call
import sys

def gen_small_csv(path, n=800, seed=7):
    rng = np.random.default_rng(seed)
    start_date = datetime(2024,1,1)
    end_date = datetime(2025,12,31)
    days_range = (end_date-start_date).days

    is_new = rng.binomial(1, 0.25, size=n)
    days_relationship = np.where(is_new==1, rng.integers(1, 90, size=n), rng.integers(90, 1500, size=n))
    score_band = rng.choice(['<600','600-650','650-700','700-800','800+'], size=n, p=[0.12,0.18,0.28,0.30,0.12])
    recency_days = np.where(is_new==1, rng.integers(1, 30, size=n), rng.integers(1, 210, size=n))
    freq_90d = np.clip(np.where(is_new==1, rng.poisson(3.0, size=n), rng.poisson(26, size=n)), 0, 180)
    monetary_90d = np.clip(np.where(is_new==1, rng.normal(160, 55, size=n), rng.normal(1100, 420, size=n)), 0, None)
    weekend_ratio = np.clip(rng.normal(0.30, 0.12, size=n), 0.0, 0.95)
    spend_std_90d = np.clip(rng.normal(120, 60, size=n), 0, None)
    app_interactions_30d = np.where(is_new==1, rng.poisson(1.2, size=n), rng.poisson(2.6, size=n))
    engagement_rate = np.clip(app_interactions_30d / np.maximum(days_relationship/30, 1), 0, None)
    multimodal_90d = np.clip(np.where(is_new==1, rng.poisson(0.5, size=n), rng.poisson(3.2, size=n)), 0, 40)
    plan_active = np.where(is_new==1, rng.binomial(1, 0.10, size=n), rng.binomial(1, 0.50, size=n))
    plan_price = np.where(plan_active==1, rng.choice([49, 79, 99, 129], size=n, p=[0.25,0.35,0.30,0.10]), 0)
    reactivations_12m = np.where(is_new==1, 0, rng.binomial(2, 0.18, size=n))
    inadimplencia = np.where(is_new==1, rng.binomial(1, 0.03, size=n), rng.binomial(1, 0.07, size=n))

    score_map = {'<600':0.30,'600-650':0.22,'650-700':0.16,'700-800':0.12,'800+':0.08}
    base_churn = np.vectorize(score_map.get)(score_band)
    latent_churn = np.clip(base_churn + 0.12*inadimplencia + 0.05*np.tanh(recency_days/70) - 0.05*plan_active, 0.01, 0.85)
    churn_90d = rng.binomial(1, latent_churn)

    latent_conv = np.clip(0.10 + 0.06*np.tanh((freq_90d-15)/30) + 0.04*(plan_active==0) - 0.06*latent_churn, 0.01, 0.85)
    convert_30d = rng.binomial(1, latent_conv)

    monthly_rev = (monetary_90d/3) + plan_price
    survival = np.exp(-2.0*latent_churn)
    ltv_12m = np.clip(12*monthly_rev*survival + rng.normal(0, 120, size=n), 0, None)

    ref_offsets = rng.integers(0, days_range+1, size=n)
    ref_date = [start_date + timedelta(days=int(x)) for x in ref_offsets]

    df = pd.DataFrame({
        "customer_id": np.arange(1, n+1),
        "ref_date": pd.to_datetime(ref_date),
        "is_new": is_new,
        "days_relationship": days_relationship,
        "score_band": score_band,
        "recency_days": recency_days,
        "freq_90d": freq_90d,
        "monetary_90d": np.round(monetary_90d,2),
        "weekend_ratio": np.round(weekend_ratio,3),
        "spend_std_90d": np.round(spend_std_90d,2),
        "app_interactions_30d": app_interactions_30d,
        "engagement_rate": np.round(engagement_rate,3),
        "multimodal_90d": multimodal_90d,
        "plan_active": plan_active,
        "plan_price": plan_price,
        "reactivations_12m": reactivations_12m,
        "inadimplencia": inadimplencia,
        "churn_90d": churn_90d,
        "convert_30d": convert_30d,
        "ltv_12m": np.round(ltv_12m,2),
    })
    df.to_csv(path, index=False)

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with tempfile.TemporaryDirectory() as td:
        data = os.path.join(td, "smoke.csv")
        outdir = os.path.join(td, "out")
        gen_small_csv(data)
        check_call([sys.executable, os.path.join(root, "src", "train_and_score.py"), "--data", data, "--outdir", outdir])
        print("Smoke test OK. metrics.json exists:", os.path.exists(os.path.join(outdir, "metrics.json")))

if __name__ == "__main__":
    main()
