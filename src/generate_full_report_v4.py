import json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_roc(y_true, y_score, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={roc_auc:.3f})")
    savefig(outpath)

def plot_pr(y_true, y_score, title, outpath):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(7,5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AUC_PR={pr_auc:.3f})")
    savefig(outpath)

def pdf_add_title(c, text, y):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, text)
    return y-1.0*cm

def pdf_add_paragraph(c, text, y, size=10, max_width=17*cm):
    c.setFont("Helvetica", size)
    words = text.split()
    line = ""
    for w in words:
        test_line = (line + " " + w).strip()
        if c.stringWidth(test_line, "Helvetica", size) <= max_width:
            line = test_line
        else:
            c.drawString(2*cm, y, line)
            y -= 0.55*cm
            line = w
            if y < 2*cm:
                c.showPage()
                y = 28*cm
                c.setFont("Helvetica", size)
    if line:
        c.drawString(2*cm, y, line)
        y -= 0.55*cm
    return y

def pdf_add_image(c, img_path, y, caption=None, max_w=17*cm, max_h=8.5*cm):
    img_path = Path(img_path)
    if not img_path.exists():
        return y
    ir = ImageReader(str(img_path))
    iw, ih = ir.getSize()
    scale = min(max_w/iw, max_h/ih)
    w, h = iw*scale, ih*scale
    x = 2*cm
    c.drawImage(ir, x, y-h, width=w, height=h)
    y = y - h - 0.4*cm
    if caption:
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(2*cm, y, caption)
        y -= 0.6*cm
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_sintetica_mobilidade.csv")
    ap.add_argument("--scored", default="outputs/scored_with_nba.csv")
    ap.add_argument("--top3", default="outputs/top3_offers.csv")
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--pdf", default="outputs/Relatorio_V4_Completo.pdf")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8")) if Path(args.metrics).exists() else {}
    data = pd.read_csv(args.data, parse_dates=["ref_date"]) if Path(args.data).exists() else pd.DataFrame()
    scored = pd.read_csv(args.scored, parse_dates=["ref_date"]) if Path(args.scored).exists() else pd.DataFrame()
    top3 = pd.read_csv(args.top3, parse_dates=["ref_date"]) if Path(args.top3).exists() else pd.DataFrame()

    cutoff = pd.Timestamp(metrics.get("cutoff","2025-06-30"))
    test = data[data["ref_date"] > cutoff].copy() if not data.empty else pd.DataFrame()

    merged = scored
    if not scored.empty and not test.empty:
        lab = test[["customer_id","ref_date","churn_90d","convert_30d","ltv_12m"]].copy()
        merged = scored.merge(lab, on=["customer_id","ref_date"], how="left")

    if "churn_90d" in merged.columns and "p_churn_90d" in merged.columns and merged["churn_90d"].notna().any():
        y = merged["churn_90d"].astype(int)
        s = merged["p_churn_90d"].astype(float)
        plot_roc(y, s, "ROC – Churn 90d", plots/"roc_churn.png")
        plot_pr(y, s, "Precision-Recall – Churn 90d", plots/"pr_churn.png")

    if "convert_30d" in merged.columns and "p_convert_30d" in merged.columns and merged["convert_30d"].notna().any():
        y = merged["convert_30d"].astype(int)
        s = merged["p_convert_30d"].astype(float)
        plot_roc(y, s, "ROC – Conversão 30d", plots/"roc_conv.png")
        plot_pr(y, s, "Precision-Recall – Conversão 30d", plots/"pr_conv.png")

    if "p_churn_90d" in merged.columns and not merged.empty:
        plt.figure(figsize=(7,5))
        plt.hist(merged["p_churn_90d"].astype(float), bins=30)
        plt.xlabel("p_churn_90d"); plt.ylabel("count")
        plt.title("Distribuição – p_churn_90d")
        savefig(plots/"dist_p_churn.png")

    if "p_convert_30d" in merged.columns and not merged.empty:
        plt.figure(figsize=(7,5))
        plt.hist(merged["p_convert_30d"].astype(float), bins=30)
        plt.xlabel("p_convert_30d"); plt.ylabel("count")
        plt.title("Distribuição – p_convert_30d")
        savefig(plots/"dist_p_conv.png")

    if "ltv_12m_hat" in merged.columns and not merged.empty:
        plt.figure(figsize=(7,5))
        plt.hist(merged["ltv_12m_hat"].astype(float), bins=30)
        plt.xlabel("ltv_12m_hat"); plt.ylabel("count")
        plt.title("Distribuição – ltv_12m_hat")
        savefig(plots/"dist_ltv_hat.png")

    if not top3.empty and "offer_id" in top3.columns:
        cnt = top3["offer_id"].value_counts()
        plt.figure(figsize=(7,5))
        plt.bar(cnt.index.astype(str), cnt.values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Recomendações por oferta (Top-N)")
        savefig(plots/"offer_counts.png")

        if "expected_value" in top3.columns:
            plt.figure(figsize=(7,5))
            plt.hist(top3["expected_value"].astype(float), bins=30)
            plt.title("Histograma – EV (Top-N)")
            savefig(plots/"ev_hist.png")

    # PDF
    c = canvas.Canvas(str(args.pdf), pagesize=A4)
    y = 28.5*cm
    y = pdf_add_title(c, "Relatório – PoC NBO/NBA (V4)", y)
    y = pdf_add_paragraph(c, "Relatório gerado a partir das saídas do pipeline V4.", y)

    y = pdf_add_title(c, "Métricas", y)
    if metrics:
        churn = metrics.get("churn", {})
        conv = metrics.get("conversion", {})
        ltv = metrics.get("ltv", {})
        y = pdf_add_paragraph(c, f"Cutoff: {metrics.get('cutoff','-')} | Treino: {metrics.get('rows_train','-')} | Teste: {metrics.get('rows_test','-')}", y)
        if isinstance(churn, dict) and 'auc' in churn:
            y = pdf_add_paragraph(c, f"Churn: AUC={float(churn.get('auc',0)):.3f}", y)
        if isinstance(conv, dict) and 'auc' in conv:
            y = pdf_add_paragraph(c, f"Conversão: AUC={float(conv.get('auc',0)):.3f}", y)
        if isinstance(ltv, dict) and 'rmse' in ltv:
            y = pdf_add_paragraph(c, f"LTV: RMSE={float(ltv.get('rmse',0)):.2f}", y)

    y = pdf_add_title(c, "Gráficos", y)
    for img, cap in [
        (plots/"roc_churn.png","ROC – churn"),
        (plots/"pr_churn.png","PR – churn"),
        (plots/"roc_conv.png","ROC – conversão"),
        (plots/"pr_conv.png","PR – conversão"),
        (plots/"dist_p_churn.png","Distribuição – p_churn"),
        (plots/"dist_p_conv.png","Distribuição – p_convert"),
        (plots/"dist_ltv_hat.png","Distribuição – LTV_hat"),
        (plots/"offer_counts.png","Recomendações por oferta"),
        (plots/"ev_hist.png","Histograma – EV"),
    ]:
        if y < 10*cm:
            c.showPage(); y = 28*cm
        y = pdf_add_image(c, img, y, cap)
    c.save()
    print("OK:", args.pdf)

if __name__ == "__main__":
    main()
