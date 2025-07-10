import re, sys, json
from pathlib import Path
from datetime import datetime
from dateutil import parser as dateparse
from fuzzywuzzy import process
import pandas as pd, numpy as np
import shap, joblib
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import seaborn as sns

BASE        = Path(__file__).resolve().parents[1]
DATA_DIR    = BASE / "Data" / "ML Data"
MODEL_DIR   = BASE / "Prediction" / "Models"
MODEL_PATH  = MODEL_DIR / "flight_price_xgb.joblib"
DEFAULTS_JS = MODEL_DIR / "defaults.json"
MONTH_JS = MODEL_DIR / "month_counts.json"
MONTH_COUNTS = json.load(open(MONTH_JS))
MONTH_MEDIAN = int(np.median(list(MONTH_COUNTS.values())))

PIPE = joblib.load(MODEL_PATH)
enc  = PIPE.named_steps["prep"].named_transformers_["cat"]

CATS = {n: cats.tolist() for n, cats in zip(
        ["Airline","Source","Destination","Additional_Info"], enc.categories_)}

if DEFAULTS_JS.exists():
    DEFAULTS = json.load(open(DEFAULTS_JS))
else:
    feat = pd.read_pickle(DATA_DIR/"train_features.pkl")
    DEFAULTS = {
        "Airline":          feat["Airline"].mode()[0],
        "Source":           feat["Source"].mode()[0],
        "Destination":      feat["Destination"].mode()[0],
        "Additional_Info":  feat["Additional_Info"].mode()[0],
        "dep_hour":         int(feat["dep_hour"].median()),
        "arr_hour":         int(feat["arr_hour"].median()),
        "duration_mins":    float(feat["duration_mins"].median()),
        "stops":            int(feat["stops"].median()),
        "journey_day":      15,
        "journey_month":    6,
        "journey_dow":      2,
        "route_len":        int(feat["route_len"].median()),
    }
    json.dump(DEFAULTS, open(DEFAULTS_JS,"w"), indent=2)

train_raw = pd.read_pickle(DATA_DIR/"train_features.pkl")[["Source","Destination","Price"]]
range_tbl = train_raw.groupby(["Source","Destination"])["Price"].agg(["min","max"]).to_dict("index")

PAT = re.compile(
    r"from\s+(?P<src>.+?)\s+to\s+(?P<dst>.+?)\s+on\s+(?P<date>[^ ]+\s+[^ ]+)(?:\s+with\s+(?P<air>.+))?",
    flags=re.I)

def fuzzy(tok, choices):  
    hit, score = process.extractOne(tok, choices)
    return hit if score >= 70 else None

def query_to_row(q:str)->pd.DataFrame:
    m = PAT.search(q.lower())
    if not m:
        raise ValueError("Use: from X to Y on <date> [with Airline]")
    gd = m.groupdict()
    src = fuzzy(gd["src"], CATS["Source"])          or DEFAULTS["Source"]
    dst = fuzzy(gd["dst"], CATS["Destination"])     or DEFAULTS["Destination"]
    air = fuzzy((gd["air"] or ""), CATS["Airline"]) or DEFAULTS["Airline"]
    dt  = dateparse.parse(gd["date"], dayfirst=True, default=datetime.now())

    row = {
        "Airline": air, "Source": src, "Destination": dst,
        "Additional_Info": DEFAULTS["Additional_Info"],

        "dep_hour": DEFAULTS["dep_hour"],
        "arr_hour": DEFAULTS["arr_hour"],
        "duration_mins": DEFAULTS["duration_mins"],
        "stops": DEFAULTS["stops"],
        "journey_day": dt.day,
        "journey_month": dt.month,
        "journey_dow": dt.weekday(),
        "route_len": DEFAULTS["route_len"],

        "flights_in_month": MONTH_COUNTS.get(dt.month, MONTH_MEDIAN),
        "fuel_proxy": DEFAULTS["duration_mins"] * (DEFAULTS["stops"] + 1),
    }
    return pd.DataFrame([row]), src, dst

def explain(row: pd.DataFrame, top=10, plot_top=25) -> tuple[str, float]:
    EXPLAINER = shap.TreeExplainer(PIPE.named_steps["xgb"])
    x_enc         = PIPE.named_steps["prep"].transform(row)
    shap_vals_all = EXPLAINER.shap_values(x_enc)[0]                 
    names_all     = PIPE.named_steps["prep"].get_feature_names_out()
    baseline      = EXPLAINER.expected_value
    pred_exact    = baseline + shap_vals_all.sum()

    # ---- identifying active one-hots -------------------------
    air, src, dst, add = (row["Airline"].iat[0],
                          row["Source"].iat[0],
                          row["Destination"].iat[0],
                          row["Additional_Info"].iat[0])

    visible_mask = [
        n.endswith(f"_{air}")  if n.startswith("cat__Airline_")        else
        n.endswith(f"_{src}")  if n.startswith("cat__Source_")         else
        n.endswith(f"_{dst}")  if n.startswith("cat__Destination_")    else
        n.endswith(f"_{add}")  if n.startswith("cat__Additional_Info_") else
        True
        for n in names_all
    ]

    shap_vals_vis = shap_vals_all[visible_mask]
    names_vis     = names_all[visible_mask]

    order_vis   = np.argsort(np.abs(shap_vals_vis))[::-1]
    top_idx_vis = order_vis[:top]

    top_lines = [
        f"  {names_vis[i]:<45} {shap_vals_vis[i]:+,.0f}"
        for i in top_idx_vis
    ]
    top_sum = shap_vals_vis[top_idx_vis].sum()

    other_sum = shap_vals_all.sum() - top_sum
    top_lines.append(
        f"  other features                               {other_sum:+,.0f}"
    )

    console = "\n".join([
        "Breakdown (top drivers):",
        f"  Baseline (E[f(X)]){'':<28} {baseline:,.0f}",
        *top_lines,
        "  ─" * 25,
        f"  Sum = Predicted   : {pred_exact:,.0f}",
    ])

    order_all = np.argsort(np.abs(shap_vals_all))[::-1]
    plot_idx  = order_all[:plot_top]
    exp_plot  = shap.Explanation(values=shap_vals_all,
                                 base_values=baseline,
                                 feature_names=names_all)
    plt.style.use("dark_background")
    shap.plots.waterfall(exp_plot, max_display=plot_top, show=False)
    plt.gcf().set_size_inches(10, 8); plt.tight_layout()
    plt.savefig("shap_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    return console, baseline

def predict(q: str) -> str:
    row, src, dst = query_to_row(q)
    fare = PIPE.predict(row)[0]

    rng = range_tbl.get((src, dst))
    if rng:
        lo, hi = rng["min"], rng["max"]
    else:
        lo, hi = fare * 0.9, fare * 1.1
        hdr = "(no historical data for that route — using ±10 %)"
    breakdown, base = explain(row)

    parts = [
        f"### Predicted fare: **₹{fare:,.0f}**",
        f"Global baseline : ₹{base:,.0f}",
        f"Historical range: ₹{lo:,.0f} – ₹{hi:,.0f}",
    ]
    if not rng:
        parts.append(hdr)
    parts.append("```text\n" + breakdown + "\n```")
    parts.append("_Full SHAP waterfall saved as **shap_plot.png**_")
    return "\n\n".join(parts)

if __name__ == "__main__":
    print("   Inference ready. Example:\n"
          "   Predict flight price from Bangalore to Delhi on 15 August with SpiceJet")
    while True:
        try:
            txt = input("\n> ").strip()
            if not txt:
                break
            predict(txt)
        except Exception as e:
            print("⚠️ ", e, file=sys.stderr)