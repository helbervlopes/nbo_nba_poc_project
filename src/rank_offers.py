import argparse, os, yaml
import pandas as pd

def score_band_geq(sb: str, min_sb: str) -> bool:
    order = ["<600","600-650","650-700","700-800","800+"]
    return order.index(sb) >= order.index(min_sb)

def eligible(row, elig: dict) -> bool:
    if not elig:
        return True
    if "plan_active" in elig and int(row["plan_active"]) != int(elig["plan_active"]):
        return False
    if "min_score_band" in elig and not score_band_geq(str(row["score_band"]), str(elig["min_score_band"])):
        return False
    if "min_freq_90d" in elig and float(row["freq_90d"]) < float(elig["min_freq_90d"]):
        return False
    if "min_monetary_90d" in elig and float(row["monetary_90d"]) < float(elig["min_monetary_90d"]):
        return False
    if "min_churn" in elig and float(row["p_churn_90d"]) < float(elig["min_churn"]):
        return False
    if "max_churn" in elig and float(row["p_churn_90d"]) > float(elig["max_churn"]):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--offers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--alpha_ltv", type=float, default=0.04)
    args = ap.parse_args()

    df = pd.read_csv(args.scored, parse_dates=["ref_date"])
    cat = yaml.safe_load(open(args.offers, "r", encoding="utf-8"))
    offers = cat.get("offers", [])
    if not offers:
        raise ValueError("offers.yaml sem ofertas.")

    rows = []
    for _, r in df.iterrows():
        p = float(r["p_convert_30d"])
        churn = float(r.get("p_churn_90d", 0.0))
        ltv = float(r.get("ltv_12m_hat", 0.0))
        for off in offers:
            elig = off.get("eligibility", {})
            if not eligible(r, elig):
                continue
            margin = float(off.get("margin", 0.0))
            cost = float(off.get("cost", 0.0))
            ev = p * (margin + args.alpha_ltv * ltv) - cost
            rows.append({
                "customer_id": int(r["customer_id"]),
                "ref_date": r["ref_date"],
                "offer_id": off.get("offer_id"),
                "offer_name": off.get("name"),
                "p_convert": p,
                "p_churn": churn,
                "ltv_hat": ltv,
                "expected_value": ev,
            })

    cand = pd.DataFrame(rows)
    cand.sort_values(["customer_id","expected_value"], ascending=[True, False], inplace=True)
    cand["rank"] = cand.groupby("customer_id").cumcount() + 1
    top = cand[cand["rank"] <= args.topn].copy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    top.to_csv(args.out, index=False)
    print("OK:", args.out)

if __name__ == "__main__":
    main()
