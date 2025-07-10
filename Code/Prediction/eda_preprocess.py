from pathlib import Path
import pandas as pd, numpy as np

BASE     = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "Data" / "ML Data"

raw = pd.read_excel(DATA_DIR / "Data_Train.xlsx")
raw["RowID"] = np.arange(len(raw))          

print("\n🔎 Missing-value count:")
print(raw.isna().sum())

d_journey = pd.to_datetime(raw["Date_of_Journey"], format="%d/%m/%Y")
# monthly demand: number of flights in each calendar month (1-12)
month_counts = d_journey.dt.month.value_counts().to_dict()
dep_hr    = pd.to_datetime(raw["Dep_Time"].str[:5],  format="%H:%M").dt.hour
arr_hr    = pd.to_datetime(raw["Arrival_Time"].str[:5], format="%H:%M").dt.hour

# duration to minutes
dur_mins  = (
    raw["Duration"].str.extract(r"(?P<h>\d+)h").fillna(0).astype(float)["h"] * 60
    + raw["Duration"].str.extract(r"(?P<m>\d+)m").fillna(0).astype(float)["m"]
)

# stops to int
stops = (
    raw["Total_Stops"]
    .replace({"non-stop": 0})
    .str.extract(r"(\d+)")
    .fillna(0)
    .astype(int)
    .squeeze()
)

# ── feature set -------------------------------------------------
feat = pd.DataFrame({
    "RowID":          raw["RowID"],
    "Airline":        raw["Airline"],
    "Source":         raw["Source"],
    "Destination":    raw["Destination"],
    "Additional_Info":raw["Additional_Info"],

    "dep_hour":       dep_hr,
    "arr_hour":       arr_hr,
    "duration_mins":  dur_mins,
    "stops":          stops,
    "journey_day":    d_journey.dt.day,
    "journey_month":  d_journey.dt.month,
    "journey_dow":    d_journey.dt.dayofweek,
    "route_len":      raw["Route"].str.count(r"→|:") + 1,

    "flights_in_month": d_journey.dt.month.map(month_counts),
    "fuel_proxy":       dur_mins * (stops + 1),

    "Price":          raw["Price"],
})

num_cols = ["dep_hour","arr_hour","duration_mins","stops",
            "journey_day","journey_month","journey_dow","route_len"]
corr = feat[num_cols + ["Price"]].corr()["Price"].sort_values(ascending=False)
print("\n📈 Pearson correlation with Price:")
print(corr)

feat.to_pickle(DATA_DIR / "train_features.pkl")
import json
(META := (DATA_DIR.parent / "Prediction" / "Models")).mkdir(exist_ok=True, parents=True)
json.dump(month_counts, open(META / "month_counts.json", "w"))
print("\ntrain_features.pkl written — shape", feat.shape)